
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSLELoss(nn.Module):
    """
    Computes the Mean Squared Logarithmic Error loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred_clamped = torch.clamp(y_pred, min=0.0)
        y_true_clamped = torch.clamp(y_true, min=0.0)

        log_pred = torch.log1p(y_pred_clamped)
        log_true = torch.log1p(y_true_clamped)
        return F.mse_loss(log_pred, log_true)

class WeightedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error with sample-wise weights.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, weights):

        y_pred_s = y_pred.squeeze()
        y_true_s = y_true.squeeze()
        weights_s = weights.squeeze()

        squared_errors = F.mse_loss(y_pred_s, y_true_s, reduction='none')
        weighted_squared_errors = squared_errors * weights_s
        return torch.mean(weighted_squared_errors)

class WeightedMSLELoss(nn.Module):
    """
    Computes the Mean Squared Logarithmic Error with sample-wise weights.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, weights):
        y_pred_clamped = torch.clamp(y_pred, min=0.0)
        y_true_clamped = torch.clamp(y_true, min=0.0)

        log_pred = torch.log1p(y_pred_clamped.squeeze())
        log_true = torch.log1p(y_true_clamped.squeeze())
        weights_s = weights.squeeze()

        log_squared_errors = F.mse_loss(log_pred, log_true, reduction='none')
        weighted_log_squared_errors = log_squared_errors * weights_s
        return torch.mean(weighted_log_squared_errors)


class HuberLoss(nn.Module):
    """
    Huber (SmoothL1) loss with configurable delta (beta).
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        delta = self.delta
        if delta <= 0:
            return abs_error.mean()
        quadratic = 0.5 * (error ** 2) / delta
        linear = abs_error - 0.5 * delta
        loss = torch.where(abs_error <= delta, quadratic, linear)
        return loss.mean()


class WeightedHuberLoss(nn.Module):
    """
    Weighted Huber loss with sample-wise weights.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, y_pred, y_true, weights):
        y_pred_s = y_pred.squeeze()
        y_true_s = y_true.squeeze()
        weights_s = weights.squeeze()

        error = y_true_s - y_pred_s
        abs_error = torch.abs(error)
        delta = self.delta
        if delta <= 0:
            loss = abs_error
        else:
            quadratic = 0.5 * (error ** 2) / delta
            linear = abs_error - 0.5 * delta
            loss = torch.where(abs_error <= delta, quadratic, linear)
        return torch.mean(loss * weights_s)


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted BCEWithLogits loss with sample-wise weights (and optional pos_weight).

    - y_pred: logits (unbounded)
    - y_true: binary labels in {0,1}
    - weights: sample-wise weights (e.g., for rare positives / curriculum weighting)
    """

    def __init__(self, *, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.base = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, y_pred, y_true, weights):
        y_pred_s = y_pred.squeeze()
        y_true_s = y_true.squeeze()
        weights_s = weights.squeeze()
        loss = self.base(y_pred_s, y_true_s)
        return torch.mean(loss * weights_s)


class WeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss with optional sample-wise weights and class weights.

    - logits: [B, C]
    - targets: [B] int64 class indices
    - weights: [B] sample-wise weights (optional, set to ones if unused)
    """

    def __init__(self, *, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.base = nn.CrossEntropyLoss(reduction="none", weight=class_weights)

    def forward(self, logits, targets, weights):
        logits_s = logits
        targets_s = targets.long().view(-1)
        weights_s = weights.view(-1).float()
        loss = self.base(logits_s, targets_s)
        w_sum = torch.clamp(weights_s.sum(), min=1e-12)
        return torch.sum(loss * weights_s) / w_sum


class FocalCrossEntropyLoss(nn.Module):
    """
    Focal loss for multiclass classification.

    This is useful for class-imbalanced problems where the model tends to
    over-predict the majority class.

    - logits: [B, C]
    - targets: [B] int64 class indices
    - weights: optional [B] sample-wise weights (defaults to 1)
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.base = nn.CrossEntropyLoss(
            reduction="none",
            weight=class_weights,
            label_smoothing=float(label_smoothing or 0.0),
        )

    def forward(self, logits, targets, weights=None):
        targets_s = targets.long().view(-1)
        loss_ce = self.base(logits, targets_s)

        p_t = torch.exp(-loss_ce)
        focal = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = focal * loss_ce
        if weights is None:
            return torch.mean(loss)
        weights_s = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype).view(-1)
        w_sum = torch.clamp(weights_s.sum(), min=1e-12)
        return torch.sum(loss * weights_s) / w_sum


class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized Cross Entropy loss for noisy-label multiclass classification.

    q=0 approaches CE, q=1 approaches MAE-like behavior. Values around 0.7 are
    common for noisy labels.
    """

    def __init__(
        self,
        *,
        q: float = 0.7,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.q = float(q)
        self.register_buffer("class_weights", class_weights.detach().clone() if class_weights is not None else None)

    def forward(self, logits, targets, weights=None):
        targets_s = targets.long().view(-1)
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets_s.view(-1, 1)).view(-1).clamp(min=1e-8, max=1.0)
        if abs(self.q) < 1e-8:
            loss = -torch.log(p_t)
        else:
            loss = (1.0 - torch.pow(p_t, self.q)) / self.q
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device, dtype=loss.dtype)[targets_s]
        if weights is not None:
            weights_s = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype).view(-1)
            return torch.sum(loss * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
        return torch.mean(loss)


class BootstrapCrossEntropyLoss(nn.Module):
    """
    Bootstrap loss for noisy labels.

    The target is mixed with the model's current prediction:
    - soft mode: beta * one_hot + (1-beta) * prob
    - hard mode: beta * one_hot + (1-beta) * one_hot(argmax(prob))
    """

    def __init__(
        self,
        *,
        beta: float = 0.8,
        mode: str = "soft",
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.mode = str(mode or "soft").lower()
        self.register_buffer("class_weights", class_weights.detach().clone() if class_weights is not None else None)

    def forward(self, logits, targets, weights=None):
        targets_s = targets.long().view(-1)
        n_classes = int(logits.shape[1])
        log_prob = F.log_softmax(logits, dim=1)
        hard = F.one_hot(targets_s, num_classes=n_classes).to(logits.device, dtype=logits.dtype)
        with torch.no_grad():
            prob = torch.softmax(logits, dim=1)
            if self.mode == "hard":
                pseudo = F.one_hot(torch.argmax(prob, dim=1), num_classes=n_classes).to(logits.device, dtype=logits.dtype)
            else:
                pseudo = prob
            target = (self.beta * hard) + ((1.0 - self.beta) * pseudo)
        loss = -torch.sum(target * log_prob, dim=1)
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device, dtype=loss.dtype)[targets_s]
        if weights is not None:
            weights_s = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype).view(-1)
            return torch.sum(loss * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
        return torch.mean(loss)


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax loss using train-set class counts."""

    def __init__(
        self,
        *,
        class_counts: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        counts = torch.clamp(class_counts.detach().clone().float(), min=1.0)
        self.register_buffer("log_counts", torch.log(counts))
        self.label_smoothing = float(label_smoothing or 0.0)

    def forward(self, logits, targets, weights=None):
        adjusted = logits + self.log_counts.to(logits.device, dtype=logits.dtype).view(1, -1)
        loss = F.cross_entropy(
            adjusted,
            targets.long().view(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if weights is not None:
            weights_s = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype).view(-1)
            return torch.sum(loss * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
        return torch.mean(loss)


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Cross entropy with train-time logit adjustment from class priors."""

    def __init__(
        self,
        *,
        class_counts: torch.Tensor,
        tau: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        counts = torch.clamp(class_counts.detach().clone().float(), min=1.0)
        priors = counts / torch.clamp(counts.sum(), min=1.0)
        self.register_buffer("log_prior", torch.log(torch.clamp(priors, min=1e-8)))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing or 0.0)

    def forward(self, logits, targets, weights=None):
        adjusted = logits + (self.tau * self.log_prior.to(logits.device, dtype=logits.dtype).view(1, -1))
        loss = F.cross_entropy(
            adjusted,
            targets.long().view(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if weights is not None:
            weights_s = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype).view(-1)
            return torch.sum(loss * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
        return torch.mean(loss)


class ClassBalancedEffectiveNumberLoss(nn.Module):
    """Class-balanced effective-number reweighting from long-tailed learning."""

    def __init__(
        self,
        *,
        class_counts: torch.Tensor,
        beta: float = 0.999,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        counts = torch.clamp(class_counts.detach().clone().float(), min=1.0)
        beta = float(beta)
        effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts)
        weights = (1.0 - beta) / torch.clamp(effective, min=1e-12)
        weights = weights / torch.clamp(weights.mean(), min=1e-12)
        self.register_buffer("class_weights", weights)
        self.label_smoothing = float(label_smoothing or 0.0)

    def forward(self, logits, targets, weights=None):
        targets_s = targets.long().view(-1)
        loss = F.cross_entropy(
            logits,
            targets_s,
            weight=self.class_weights.to(logits.device, dtype=logits.dtype),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if weights is not None:
            weights_s = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype).view(-1)
            return torch.sum(loss * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
        return torch.mean(loss)
