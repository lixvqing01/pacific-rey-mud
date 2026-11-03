from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MechanismAwareREYRegressor(nn.Module):
    """CNN encoder with mechanism-gated residual regression heads.

    The model predicts normalized log1p(REY). A 3-class mechanism gate
    regularizes the representation, while residual experts adjust a global
    continuous REY prediction:

        mu = mu_global + sum_k softmax(mech_logits)_k * residual_k
    """

    def __init__(
        self,
        *,
        input_channels: int,
        n_mechanisms: int = 3,
        n_thresholds: int = 4,
        cnn_filters: list[int] | None = None,
        kernel_size: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        use_multiscale: bool = True,
        multiscale_kernel_size: int = 5,
        use_core_aux_head: bool = False,
        use_tail_residual_head: bool = False,
        n_domain_experts: int = 0,
        regression_head_mode: str = "mechanism_gated",
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        self.input_channels = int(input_channels)
        self.n_mechanisms = int(n_mechanisms)
        self.n_thresholds = int(n_thresholds)
        self.use_multiscale = bool(use_multiscale)
        self.use_core_aux_head = bool(use_core_aux_head)
        self.use_tail_residual_head = bool(use_tail_residual_head)
        self.n_domain_experts = int(n_domain_experts or 0)
        self.regression_head_mode = str(regression_head_mode or "mechanism_gated")
        if self.regression_head_mode not in ("mechanism_gated", "global"):
            raise ValueError(
                "regression_head_mode must be 'mechanism_gated' or 'global', "
                f"got {self.regression_head_mode!r}"
            )
        if self.n_mechanisms < 2:
            raise ValueError("n_mechanisms must be >= 2")
        if self.n_thresholds < 1:
            raise ValueError("n_thresholds must be >= 1")

        if cnn_filters is None:
            cnn_filters = [64, 128, 128]
        cnn_filters = [int(x) for x in cnn_filters]

        self.encoder = self._make_encoder(
            in_channels=self.input_channels,
            cnn_filters=cnn_filters,
            kernel_size=int(kernel_size),
        )
        self.encoder_ms = (
            self._make_encoder(
                in_channels=self.input_channels,
                cnn_filters=cnn_filters,
                kernel_size=int(multiscale_kernel_size),
            )
            if self.use_multiscale
            else None
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = int(cnn_filters[-1]) * (2 if self.use_multiscale else 1)
        hidden = int(hidden_dim)
        drop = float(dropout)
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.global_mu = nn.Linear(hidden, 1)
        self.log_sigma = nn.Linear(hidden, 1)
        self.mechanism_head = nn.Linear(hidden, self.n_mechanisms)
        self.residual_head = nn.Linear(hidden, self.n_mechanisms)
        self.threshold_head = nn.Linear(hidden, self.n_thresholds)
        self.core_peak_head = nn.Linear(hidden, 1) if self.use_core_aux_head else None
        self.tail_gate_head = nn.Linear(hidden, 1) if self.use_tail_residual_head else None
        self.tail_residual_head = nn.Linear(hidden, 1) if self.use_tail_residual_head else None
        self.domain_head = nn.Linear(hidden, self.n_domain_experts) if self.n_domain_experts > 1 else None
        self.domain_residual_head = nn.Linear(hidden, self.n_domain_experts) if self.n_domain_experts > 1 else None


        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.domain_residual_head is not None:
            nn.init.zeros_(self.domain_residual_head.weight)
            nn.init.zeros_(self.domain_residual_head.bias)
        if self.tail_residual_head is not None:
            nn.init.zeros_(self.tail_residual_head.weight)
            nn.init.zeros_(self.tail_residual_head.bias)

    @staticmethod
    def _make_encoder(*, in_channels: int, cnn_filters: list[int], kernel_size: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        current = int(in_channels)
        k = int(kernel_size)
        for idx, out_channels in enumerate(cnn_filters):
            layers.extend(
                [
                    nn.Conv2d(current, int(out_channels), k, padding=k // 2),
                    nn.BatchNorm2d(int(out_channels)),
                    nn.LeakyReLU(inplace=True),
                ]
            )
            if idx < len(cnn_filters) - 1:
                layers.append(nn.MaxPool2d(2, 2))
            current = int(out_channels)
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        main = torch.flatten(self.pool(self.encoder(x)), 1)
        if self.encoder_ms is not None:
            ms = torch.flatten(self.pool(self.encoder_ms(x)), 1)
            main = torch.cat([main, ms], dim=1)
        return self.shared_mlp(main)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        mechanism_logits = self.mechanism_head(z)
        mechanism_prob = F.softmax(mechanism_logits, dim=1)
        residuals = self.residual_head(z)
        mu_global = self.global_mu(z)
        mechanism_mu = mu_global + torch.sum(mechanism_prob * residuals, dim=1, keepdim=True)
        mu = mu_global if self.regression_head_mode == "global" else mechanism_mu
        domain_logits = None
        domain_prob = None
        domain_residuals = None
        if self.domain_head is not None and self.domain_residual_head is not None:
            domain_logits = self.domain_head(z)
            domain_prob = F.softmax(domain_logits, dim=1)
            domain_residuals = self.domain_residual_head(z)
            mu = mu + torch.sum(domain_prob * domain_residuals, dim=1, keepdim=True)
        tail_gate_logit = None
        tail_residual = None
        if self.tail_gate_head is not None and self.tail_residual_head is not None:
            tail_gate_logit = self.tail_gate_head(z)
            tail_residual = self.tail_residual_head(z)
            mu = mechanism_mu + torch.sigmoid(tail_gate_logit) * tail_residual

        log_sigma = torch.clamp(self.log_sigma(z), min=-4.0, max=3.0)
        threshold_logits = self.threshold_head(z)
        out = {
            "mu": mu,
            "mu_global": mu_global,
            "mu_mechanism": mechanism_mu,
            "log_sigma": log_sigma,
            "mechanism_logits": mechanism_logits,
            "mechanism_prob": mechanism_prob,
            "mechanism_residuals": residuals,
            "threshold_logits": threshold_logits,
        }
        if domain_logits is not None and domain_prob is not None and domain_residuals is not None:
            out["domain_logits"] = domain_logits
            out["domain_prob"] = domain_prob
            out["domain_residuals"] = domain_residuals
        if self.core_peak_head is not None:
            out["core_peak_mu"] = self.core_peak_head(z)
        if tail_gate_logit is not None and tail_residual is not None:
            out["tail_gate_logit"] = tail_gate_logit
            out["tail_residual"] = tail_residual
        return out


def student_t_nll(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    *,
    nu: float = 4.0,
) -> torch.Tensor:
    """Per-sample Student-t negative log likelihood."""

    y_true = y_true.view_as(mu)
    sigma = torch.exp(log_sigma).clamp_min(1e-5)
    v = torch.as_tensor(float(nu), dtype=mu.dtype, device=mu.device)
    z = (y_true - mu) / sigma
    log_norm = (
        torch.lgamma((v + 1.0) / 2.0)
        - torch.lgamma(v / 2.0)
        - 0.5 * torch.log(v * torch.pi)
        - log_sigma
    )
    log_prob = log_norm - ((v + 1.0) / 2.0) * torch.log1p((z * z) / v)
    return -log_prob.squeeze(1)


def ordinal_monotonic_penalty(threshold_logits: torch.Tensor) -> torch.Tensor:
    """Penalty for violating P(t_low) >= P(t_high)."""

    if threshold_logits.shape[1] <= 1:
        return threshold_logits.new_tensor(0.0)
    violations = torch.relu(threshold_logits[:, 1:] - threshold_logits[:, :-1])
    return torch.mean(violations * violations)
