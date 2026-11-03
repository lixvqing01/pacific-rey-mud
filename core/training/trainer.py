
import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.amp import autocast

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from core.utils.logger_config import logger
from core.models.cnn_mlp import FactorizedEnvDepthModel, MixtureOfProfilesModel, HierarchicalMixtureOfProfilesModel
from core.models.losses import (
    MSLELoss,
    WeightedMSELoss,
    WeightedMSLELoss,
    HuberLoss,
    WeightedHuberLoss,
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    FocalCrossEntropyLoss,
    GeneralizedCrossEntropyLoss,
    BootstrapCrossEntropyLoss,
    BalancedSoftmaxLoss,
    LogitAdjustedCrossEntropyLoss,
    ClassBalancedEffectiveNumberLoss,
)

from core.visualization.scatter_plots import plot_scatter_plotly, plot_scatter
from core.visualization.diagnostic_plots import plot_residuals, plot_error_histogram, plot_depth_residuals
from core.visualization.classification_diagnostics import (
    compute_high_grade_metrics,
    infer_high_grade_classes,
    save_classification_tables,
    save_grade_class_diagnostics,
    save_hierarchical_classification_diagnostics,
    plot_confusion_matrices,
    plot_confidence_histograms,
    plot_top1_reliability,
    plot_one_vs_rest_pr_curves,
    save_target_class_threshold_sweep,
)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0:
        return np.zeros((0, int(z.shape[1]) if z.ndim == 2 else 0), dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(np.clip(z, -60.0, 60.0))
    return ez / np.clip(np.sum(ez, axis=1, keepdims=True), 1e-12, None)


def _hierarchical_outputs_to_log_probs(outputs, *, grade_multitask: bool) -> torch.Tensor:
    if grade_multitask:
        end_logits, sub_logits, grade_logits = outputs
        pe = torch.softmax(end_logits, dim=1)
        ps = torch.softmax(sub_logits, dim=1)
        pg = torch.softmax(grade_logits, dim=1)
        p_non = pe[:, 1:2]
        p_full = torch.cat(
            [
                pe[:, 0:1],
                p_non * ps[:, 0:1] * pg[:, 0:1],
                p_non * ps[:, 0:1] * pg[:, 1:2],
                p_non * ps[:, 1:2] * pg[:, 0:1],
                p_non * ps[:, 1:2] * pg[:, 1:2],
            ],
            dim=1,
        )
    else:
        end_logits, sub_logits = outputs
        pe = torch.softmax(end_logits, dim=1)
        ps = torch.softmax(sub_logits, dim=1)
        p_full = torch.cat([pe[:, 0:1], pe[:, 1:2] * ps], dim=1)
    return torch.log(torch.clamp(p_full, 1e-12, 1.0))


def _hierarchical_targets(y_full: torch.Tensor, *, grade_multitask: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    y_full = y_full.view(-1).long()
    y_end = (y_full > 0).long()
    mask = y_full > 0
    if grade_multitask:
        y_non = y_full[mask] - 1
        y_sub = torch.div(y_non, 2, rounding_mode="floor").long()
        y_grade = torch.remainder(y_non, 2).long()
    else:
        y_sub = (y_full[mask] - 1).long()
        y_grade = None
    return y_end, y_sub, y_grade, mask


def _head_ce_loss(
    criterion,
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    use_sample_weights: bool,
    label_smoothing: float,
) -> torch.Tensor:
    targets = targets.view(-1).long()
    if targets.numel() == 0:
        return logits.sum() * 0.0
    if use_sample_weights and weights is not None:
        weights_s = weights.view(-1).float()
        try:
            return criterion(logits, targets, weights_s)
        except TypeError:
            class_weights = getattr(criterion, "weight", None)
            loss_vec = F.cross_entropy(
                logits,
                targets,
                weight=class_weights,
                reduction="none",
                label_smoothing=float(label_smoothing or 0.0),
            )
            w_sum = torch.clamp(weights_s.sum(), min=1e-12)
            return torch.sum(loss_vec * weights_s) / w_sum
    return criterion(logits, targets)


def _hierarchical_full_nll_loss(
    outputs,
    targets: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    use_sample_weights: bool,
    class_weights: torch.Tensor | None,
    grade_multitask: bool,
) -> torch.Tensor:
    y = targets.view(-1).long()
    log_probs = _hierarchical_outputs_to_log_probs(outputs, grade_multitask=grade_multitask)
    loss_vec = F.nll_loss(log_probs, y, reduction="none")
    if class_weights is not None:
        try:
            loss_vec = loss_vec * class_weights.to(device=loss_vec.device, dtype=loss_vec.dtype)[y]
        except Exception:
            pass
    if use_sample_weights and weights is not None:
        weights_s = weights.view(-1).to(device=loss_vec.device, dtype=loss_vec.dtype)
        return torch.sum(loss_vec * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
    return loss_vec.mean()


def _soft_target_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = soft_targets.to(logits.device, dtype=logits.dtype)
    targets = targets / torch.clamp(targets.sum(dim=1, keepdim=True), min=1e-12)
    if label_smoothing and label_smoothing > 0.0:
        n_classes = int(targets.shape[1])
        targets = ((1.0 - float(label_smoothing)) * targets) + (float(label_smoothing) / float(max(n_classes, 1)))
    log_prob = F.log_softmax(logits, dim=1)
    loss = -torch.sum(targets * log_prob, dim=1)
    if class_weights is not None:
        cw = class_weights.to(logits.device, dtype=logits.dtype).view(1, -1)
        loss = loss * torch.sum(targets * cw, dim=1)
    if weights is not None:
        weights_s = weights.to(logits.device, dtype=logits.dtype).view(-1)
        return torch.sum(loss * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
    return torch.mean(loss)


def _one_vs_rest_logits(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    """Convert multiclass logits to a binary logit for one target class."""

    n_classes = int(logits.shape[1]) if logits.ndim == 2 else 0
    if n_classes <= 1:
        return logits.view(-1)
    target_class = int(np.clip(int(target_class), 0, n_classes - 1))
    pos = logits[:, target_class]
    if n_classes == 2:
        other_idx = 1 - target_class
        return pos - logits[:, other_idx]
    keep = [i for i in range(n_classes) if i != target_class]
    neg = torch.logsumexp(logits[:, keep], dim=1)
    return pos - neg


def _multi_vs_rest_logits(logits: torch.Tensor, positive_classes: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Convert multiclass logits to a binary logit for a set of positive classes."""

    n_classes = int(logits.shape[1]) if logits.ndim == 2 else 0
    if n_classes <= 1:
        return logits.view(-1)
    pos = sorted({int(i) for i in positive_classes if 0 <= int(i) < n_classes})
    if not pos:
        raise ValueError("positive_classes must include at least one valid class index.")
    neg = [i for i in range(n_classes) if i not in pos]
    pos_logit = torch.logsumexp(logits[:, pos], dim=1)
    if not neg:
        return pos_logit
    neg_logit = torch.logsumexp(logits[:, neg], dim=1)
    return pos_logit - neg_logit


def _binary_logit_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Penalize binary multiclass logits that do not separate the true class by a margin."""

    if logits.ndim != 2 or int(logits.shape[1]) != 2:
        return logits.sum() * 0.0
    y = targets.view(-1).long().clamp(0, 1)
    signed = torch.where(y == 1, logits[:, 1] - logits[:, 0], logits[:, 0] - logits[:, 1])
    loss = F.relu(float(margin) - signed)
    if weights is not None:
        weights_s = weights.to(logits.device, dtype=logits.dtype).view(-1)
        return torch.sum(loss * weights_s) / torch.clamp(weights_s.sum(), min=1e-12)
    return torch.mean(loss)


def _exploration4_main_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map 4-class exploration logits to 3 main classes: 0, 1, class2."""

    if logits.ndim != 2 or int(logits.shape[1]) != 4:
        raise ValueError(f"Expected exploration 4-class logits [B,4], got {tuple(logits.shape)}.")
    return torch.stack(
        [
            logits[:, 0],
            logits[:, 1],
            torch.logsumexp(logits[:, 2:4], dim=1),
        ],
        dim=1,
    )


def _exploration4_main_targets(targets: torch.Tensor) -> torch.Tensor:
    """Map exploration labels 0,1,2,3 to main-class labels 0,1,2."""

    y = targets.view(-1).long()
    return torch.where(y >= 2, torch.full_like(y, 2), y)


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_vec = F.binary_cross_entropy_with_logits(
        logits.view(-1),
        targets.float().view(-1),
        reduction="none",
        pos_weight=pos_weight,
    )
    if weights is None:
        return torch.mean(loss_vec)
    weights_s = weights.view(-1).float()
    w_sum = torch.clamp(weights_s.sum(), min=1e-12)
    return torch.sum(loss_vec * weights_s) / w_sum


def _top_fraction_capture_from_binary(
    y_true_binary: np.ndarray,
    score: np.ndarray,
    fraction: float,
) -> float:
    yb = np.asarray(y_true_binary, dtype=int).reshape(-1)
    sc = np.asarray(score, dtype=float).reshape(-1)
    if yb.size == 0 or sc.size != yb.size:
        return float("nan")
    pos = int(np.sum(yb == 1))
    if pos <= 0:
        return float("nan")
    frac = float(np.clip(float(fraction), 1e-6, 1.0))
    k = max(1, int(np.ceil(sc.size * frac)))
    order = np.argsort(-sc, kind="mergesort")
    top_idx = order[:k]
    captured = int(np.sum(yb[top_idx] == 1))
    return float(captured / max(pos, 1))


def _collect_logits_and_targets(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    scaler_enabled: bool,
    task_type: str,
    hierarchical_multiclass: bool = False,
    hierarchical_grade_multitask: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    logits_list, targets_list = [], []
    ranking_list = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            x, y, _w = data[0].to(device), data[1].to(device), data[2].to(device)
            with autocast(device_type=device.type, enabled=scaler_enabled):
                if task_type == "multiclass_classification" and not hierarchical_multiclass:
                    out, comps = model(x, return_components=True)
                else:
                    out = model(x)
                    comps = None
            if task_type == "multiclass_classification":
                if hierarchical_multiclass:
                    logits_list.append(
                        _hierarchical_outputs_to_log_probs(
                            out,
                            grade_multitask=bool(hierarchical_grade_multitask),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    logits_list.append(out.detach().cpu().numpy())
                    rank_logit = None
                    if isinstance(comps, dict):
                        rank_logit = comps.get("high_grade_ranking_logit", None)
                    if rank_logit is not None:
                        ranking_list.append(rank_logit.view(-1).detach().cpu().numpy())
                targets_list.append(y.view(-1).detach().cpu().numpy())
            else:
                logits_list.append(out.squeeze(-1).detach().cpu().numpy())
                targets_list.append(y.squeeze(-1).detach().cpu().numpy())
    logits_np = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 0), dtype=float)
    targets_np = np.concatenate(targets_list, axis=0) if targets_list else np.array([], dtype=float)
    ranking_np = np.concatenate(ranking_list, axis=0) if ranking_list else None
    return logits_np, targets_np, ranking_np


def _loader_indices(loader) -> np.ndarray:
    dataset = getattr(loader, "dataset", None)
    idx = np.asarray(getattr(dataset, "indices", []), dtype=int)
    return idx


def _prefix_paths(paths: dict[str, str], prefix: str) -> dict[str, str]:
    return {f"{prefix}_{k}": v for k, v in (paths or {}).items()}

class ModelTrainer:
    def __init__(self, model_config, input_channels, device, output_dirs, stop_flag=None, progress_callback=None, history_callback=None):
        self.model_config = model_config
        self.input_channels = input_channels
        self.device = device
        self.output_dirs = output_dirs
        self.stop_flag = stop_flag
        self.progress_callback = progress_callback
        self.history_callback = history_callback
        self.logger = logger
        self.use_weighted_loss = self.model_config.get('use_weighted_loss', False)

        self.domain_priority_enabled = bool(self.model_config.get("domain_priority_enabled", False))
        self.domain_priority_mode = str(self.model_config.get("domain_priority_mode", "loss_weighting") or "loss_weighting").lower()
        self.domain_priority_pretrain_epochs = int(self.model_config.get("domain_priority_pretrain_epochs", 200) or 200)
        self.domain_priority_finetune_epochs = int(self.model_config.get("domain_priority_finetune_epochs", 200) or 200)
        self.domain_priority_finetune_lr_mult = float(self.model_config.get("domain_priority_finetune_lr_mult", 0.2) or 0.2)
        self.use_soft_labels = bool(self.model_config.get("use_soft_labels", False))
        self.soft_label_class_weights = None
        cw_soft = self.model_config.get("ce_class_weights", None)
        if cw_soft is not None:
            try:
                self.soft_label_class_weights = torch.tensor([float(x) for x in cw_soft], device=self.device, dtype=torch.float32)
            except Exception:
                self.soft_label_class_weights = None
        self.task_type = str(self.model_config.get("task_type", "regression") or "regression").lower()
        if self.task_type not in ("regression", "binary_classification", "multiclass_classification"):
            self.logger.warning(f"ModelTrainer: Unknown task_type='{self.task_type}', falling back to 'regression'.")
            self.task_type = "regression"
        self.classification_threshold_ppm = float(self.model_config.get("classification_threshold_ppm", 1000.0) or 1000.0)
        self.classification_label_col = self.model_config.get("classification_label_col", None)
        self.classification_class_names = self.model_config.get("classification_class_names", None)
        self.num_classes = int(self.model_config.get("num_classes", 0) or 0)
        self.early_stopping_metric = str(self.model_config.get("early_stopping_metric", "val_loss") or "val_loss").lower()
        self.hierarchical_multiclass = bool(self.model_config.get("hierarchical_multiclass", False)) and (
            self.task_type == "multiclass_classification"
        )
        self.hierarchical_grade_multitask = bool(self.model_config.get("hierarchical_grade_multitask", False)) and (
            self.hierarchical_multiclass and self.num_classes == 5
        )
        if bool(self.model_config.get("hierarchical_grade_multitask", False)) and not self.hierarchical_grade_multitask:
            self.logger.warning(
                "ModelTrainer: hierarchical_grade_multitask requires hierarchical_multiclass=True, "
                "task_type=multiclass_classification, and num_classes=5. Falling back to legacy hierarchical mode."
            )

        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.train_criterion, self.eval_criterion = self._build_criterions()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.use_ema = bool(self.model_config.get("use_ema", False))
        self.ema_decay = float(self.model_config.get("ema_decay", 0.995) or 0.995)
        self.ema_state_dict = None
        if self.use_ema:
            self.ema_state_dict = {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()
                if torch.is_floating_point(v)
            }
            self.logger.info(f"Trainer: EMA enabled (decay={self.ema_decay:g}).")
        self.use_swa = bool(self.model_config.get("use_swa", False))
        self.swa_start_epoch = int(self.model_config.get("swa_start_epoch", max(1, int(self.model_config.get("epochs", 1) or 1) // 2)) or 1)
        self.swa_n = 0
        self.swa_state_dict = None
        if self.use_swa:
            self.logger.info(f"Trainer: SWA enabled (start_epoch={self.swa_start_epoch}).")
        self.aux_high_grade_bce_weight = (
            float(self.model_config.get("aux_high_grade_bce_weight", 0.0) or 0.0)
            if self.task_type == "multiclass_classification" and not self.hierarchical_multiclass
            else 0.0
        )
        self.aux_high_grade_class_index = int(
            self.model_config.get(
                "aux_high_grade_class_index",
                self.model_config.get("explain_target_class_index", max(0, int(self.num_classes or 1) - 1)),
            )
            or 0
        )
        self.aux_high_grade_pos_weight = None
        aux_pos_weight_value = self.model_config.get("aux_high_grade_pos_weight", None)
        if aux_pos_weight_value is not None:
            try:
                self.aux_high_grade_pos_weight = torch.tensor(
                    [float(aux_pos_weight_value)],
                    device=self.device,
                    dtype=torch.float32,
                )
            except Exception:
                self.aux_high_grade_pos_weight = None
        if self.aux_high_grade_bce_weight > 0.0:
            aux_pos_weight_text = (
                f"{float(aux_pos_weight_value):g}" if aux_pos_weight_value is not None else "None"
            )
            self.logger.info(
                "Trainer: Auxiliary one-vs-rest BCE enabled "
                f"(class_index={self.aux_high_grade_class_index}, "
                f"weight={self.aux_high_grade_bce_weight:g}, "
                f"pos_weight={aux_pos_weight_text})."
            )
        self.aux_binary_tasks: list[dict[str, object]] = []
        for task_name, default_classes in [
            ("aux_high_value", [1, 3]),
            ("aux_noncarbonate", [1, 2, 3]),
            ("aux_class2", [2, 3]),
        ]:
            task_weight = float(self.model_config.get(f"{task_name}_bce_weight", 0.0) or 0.0)
            if task_weight <= 0.0:
                continue
            raw_classes = self.model_config.get(f"{task_name}_positive_classes", default_classes)
            if isinstance(raw_classes, str):
                positive_classes = [int(x.strip()) for x in raw_classes.split(",") if x.strip()]
            else:
                positive_classes = [int(x) for x in raw_classes]
            pos_weight_tensor = None
            pos_weight_value = self.model_config.get(f"{task_name}_pos_weight", None)
            if pos_weight_value is not None:
                try:
                    pos_weight_tensor = torch.tensor([float(pos_weight_value)], device=self.device, dtype=torch.float32)
                except Exception:
                    pos_weight_tensor = None
            self.aux_binary_tasks.append(
                {
                    "name": task_name,
                    "weight": float(task_weight),
                    "positive_classes": positive_classes,
                    "pos_weight": pos_weight_tensor,
                }
            )
            pos_weight_text = f"{float(pos_weight_value):g}" if pos_weight_value is not None else "None"
            self.logger.info(
                "Trainer: Auxiliary binary CE enabled "
                f"(name={task_name}, classes={positive_classes}, weight={task_weight:g}, "
                f"pos_weight={pos_weight_text})."
            )
        self.aux_main_class_ce_weight = (
            float(self.model_config.get("aux_main_class_ce_weight", 0.0) or 0.0)
            if self.task_type == "multiclass_classification"
            and not self.hierarchical_multiclass
            and int(self.num_classes or 0) == 4
            else 0.0
        )
        self.aux_main_class_ce_weights = None
        aux_main_weights = self.model_config.get("aux_main_class_ce_weights", None)
        if aux_main_weights is not None:
            try:
                weights_list = [float(x) for x in aux_main_weights]
                if len(weights_list) == 3:
                    self.aux_main_class_ce_weights = torch.tensor(
                        weights_list,
                        device=self.device,
                        dtype=torch.float32,
                    )
            except Exception:
                self.aux_main_class_ce_weights = None
        if self.aux_main_class_ce_weight > 0.0:
            self.logger.info(
                "Trainer: Auxiliary exploration main-class CE enabled "
                f"(weight={self.aux_main_class_ce_weight:g}, "
                f"class_weights={self.aux_main_class_ce_weights.tolist() if self.aux_main_class_ce_weights is not None else None})."
            )
        self.binary_logit_margin_loss_weight = (
            float(self.model_config.get("binary_logit_margin_loss_weight", 0.0) or 0.0)
            if self.task_type == "multiclass_classification"
            and not self.hierarchical_multiclass
            and int(self.num_classes or 0) == 2
            else 0.0
        )
        self.binary_logit_margin = float(self.model_config.get("binary_logit_margin", 1.0) or 1.0)
        if self.binary_logit_margin_loss_weight > 0.0:
            self.logger.info(
                "Trainer: Binary logit margin loss enabled "
                f"(weight={self.binary_logit_margin_loss_weight:g}, margin={self.binary_logit_margin:g})."
            )

        self.use_high_grade_ranking_head = bool(
            self.model_config.get("use_high_grade_ranking_head", False)
        ) and (
            self.task_type == "multiclass_classification"
            and not self.hierarchical_multiclass
        )
        self.high_grade_ranking_loss_weight = (
            float(self.model_config.get("high_grade_ranking_loss_weight", 0.0) or 0.0)
            if self.use_high_grade_ranking_head
            else 0.0
        )
        self.high_grade_ranking_target_class_index = int(
            self.model_config.get(
                "high_grade_ranking_target_class_index",
                self.model_config.get(
                    "aux_high_grade_class_index",
                    self.model_config.get("explain_target_class_index", max(0, int(self.num_classes or 1) - 1)),
                ),
            )
            or 0
        )
        self.high_grade_ranking_pos_weight = None
        ranking_pos_weight_value = self.model_config.get("high_grade_ranking_pos_weight", None)
        if ranking_pos_weight_value is not None:
            try:
                self.high_grade_ranking_pos_weight = torch.tensor(
                    [float(ranking_pos_weight_value)],
                    device=self.device,
                    dtype=torch.float32,
                )
            except Exception:
                self.high_grade_ranking_pos_weight = None
        if self.use_high_grade_ranking_head:
            ranking_pos_weight_text = (
                f"{float(ranking_pos_weight_value):g}" if ranking_pos_weight_value is not None else "None"
            )
            self.logger.info(
                "Trainer: Learned high-grade ranking head enabled "
                f"(class_index={self.high_grade_ranking_target_class_index}, "
                f"loss_weight={self.high_grade_ranking_loss_weight:g}, "
                f"pos_weight={ranking_pos_weight_text})."
            )


        self.use_quantile_loss = bool(self.model_config.get("use_quantile_loss", False)) and (self.task_type == "regression")
        self.quantile_level = float(self.model_config.get("quantile_level", 0.9))
        self.quantile_loss_weight = float(self.model_config.get("quantile_loss_weight", 0.05))

        self.training_history = []
        self.logger.info(f"ModelTrainer initialized. Device: {self.device}")
        self.logger.debug(f"ModelTrainer: Model Architecture:\n{str(self.model)}")

    def _build_model(self):
        """
        Build the underlying regression model.

        The input_channels should be env_features + 1 (for depth).
        The current training pipeline only supports the Mixture-of-Profiles
        factorized model (`model_type='mixture_profiles'`).
        """
        model_type = self.model_config.get('model_type', 'mixture_profiles').lower()

        common_kwargs = {
            "input_channels": self.input_channels,
            "device": self.device,
            "cnn_filters": self.model_config.get('cnn_filters', [64, 128, 128]),
            "kernel_size": self.model_config.get('kernel_size', 3),
            "mlp_hidden_layers": self.model_config.get('mlp_hidden_layers', [128, 64]),
            "dropout_prob": self.model_config.get('dropout_prob', 0.3),
            "linear_attention_heads": self.model_config.get('linear_attention_heads', 4),
            "use_linear_attention": bool(self.model_config.get("use_linear_attention", True)),

            "categorical_channel_indices": self.model_config.get("categorical_channel_indices", None),
            "categorical_compress_dim": self.model_config.get("categorical_compress_dim", None),
            "categorical_compress_dropout": self.model_config.get("categorical_compress_dropout", None),

            "use_multiscale_env": self.model_config.get('use_multiscale_env', False),
            "multiscale_kernel_size": self.model_config.get('multiscale_kernel_size', 5),
            "use_film": self.model_config.get('use_film', False),
            "film_embedding_dim": self.model_config.get('film_embedding_dim', 32),
            "depth_feature_order": self.model_config.get('depth_feature_order', 1),
            "combination_type": self.model_config.get('combination_type', "product"),
            "use_interaction_head": self.model_config.get('use_interaction_head', False),
            "interaction_hidden_layers": self.model_config.get('interaction_hidden_layers', None),
            "interaction_embedding_dim": self.model_config.get('interaction_embedding_dim', 32),

            "coord_feature_dim": 4 if bool(self.model_config.get("add_coord_channels", False)) else 0,
            "use_coord_branch": bool(self.model_config.get("use_coord_branch", False)),
            "coord_branch_hidden_dim": int(self.model_config.get("coord_branch_hidden_dim", 8) or 8),
            "coord_branch_dropout": float(self.model_config.get("coord_branch_dropout", 0.5) or 0.5),
            "coord_branch_weight": float(self.model_config.get("coord_branch_weight", 1.0) or 1.0),
        }

        if model_type == 'mixture_profiles':
            logger.info("ModelTrainer: Using MixtureOfProfilesModel (Mixture-of-Profiles Env×Depth).")
            num_profiles = self.model_config.get('num_profiles', 3)
            env_embedding_dim = self.model_config.get('env_embedding_dim', 64)
            profile_hidden_layers = self.model_config.get('profile_hidden_layers', [32, 16])
            if self.hierarchical_multiclass:
                n_cls = int(self.num_classes or 0)
                num_subclasses = 2 if self.hierarchical_grade_multitask else max(2, n_cls - 1)
                model = HierarchicalMixtureOfProfilesModel(
                    num_profiles=num_profiles,
                    env_embedding_dim=env_embedding_dim,
                    profile_hidden_layers=profile_hidden_layers,
                    num_subclasses=int(num_subclasses),
                    endmember_head_hidden_dim=int(self.model_config.get("endmember_head_hidden_dim", 64) or 64),
                    subclass_head_hidden_dim=int(self.model_config.get("subclass_head_hidden_dim", 64) or 64),
                    hierarchical_grade_multitask=bool(self.hierarchical_grade_multitask),
                    grade_head_hidden_dim=int(self.model_config.get("grade_head_hidden_dim", 64) or 64),
                    **common_kwargs,
                ).to(self.device)
                logger.info(
                    "ModelTrainer: Hierarchical multiclass enabled "
                    f"(mode={'grade_multitask_3head' if self.hierarchical_grade_multitask else 'legacy_2head'}, "
                    f"num_classes={n_cls}, num_subclasses={int(num_subclasses)})."
                )
            else:
                model = MixtureOfProfilesModel(
                    num_profiles=num_profiles,
                    env_embedding_dim=env_embedding_dim,
                    profile_hidden_layers=profile_hidden_layers,
                    num_classes=(self.num_classes if self.task_type == "multiclass_classification" else 1),
                    class_head_hidden_dim=int(self.model_config.get("class_head_hidden_dim", 64) or 64),
                    use_high_grade_ranking_head=bool(self.model_config.get("use_high_grade_ranking_head", False)),
                    high_grade_ranking_head_hidden_dim=int(
                        self.model_config.get("high_grade_ranking_head_hidden_dim", 64) or 64
                    ),
                    **common_kwargs,
                ).to(self.device)
        else:
            raise ValueError(
                f"ModelTrainer: Unsupported model_type='{model_type}'. "
                "Only 'mixture_profiles' is supported in the current version."
            )

        return model

    def _build_optimizer(self):
        learning_rate = self.model_config.get('learning_rate', 0.001)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.model_config.get('weight_decay', 1e-4)
        )
        return optimizer

    def _build_criterions(self):
        loss_function_type = self.model_config.get('loss_function_type', 'mse').lower()
        if self.task_type == "binary_classification":
            loss_function_type = "bce"
        elif self.task_type == "multiclass_classification":
            loss_function_type = "ce"
        huber_delta = float(self.model_config.get("huber_delta", 1.0))
        classification_loss = str(self.model_config.get("classification_loss", "ce") or "ce").lower()
        focal_gamma = float(self.model_config.get("focal_gamma", 2.0) or 2.0)
        label_smoothing = float(self.model_config.get("label_smoothing", 0.0) or 0.0)

        pos_weight_value = self.model_config.get("bce_pos_weight", None)
        pos_weight = None
        if pos_weight_value is not None:
            try:
                pos_weight = torch.tensor([float(pos_weight_value)], device=self.device, dtype=torch.float32)
            except Exception:
                pos_weight = None


        if loss_function_type == "ce":
            if self.hierarchical_multiclass:
                w_end = self.model_config.get("ce_endmember_class_weights", None)
                w_sub = self.model_config.get("ce_subclass_class_weights", None)
                w_grade = self.model_config.get("ce_grade_class_weights", None)
                tw_end = None
                tw_sub = None
                tw_grade = None
                if w_end is not None:
                    try:
                        tw_end = torch.tensor([float(x) for x in w_end], device=self.device, dtype=torch.float32)
                    except Exception:
                        tw_end = None
                if w_sub is not None:
                    try:
                        tw_sub = torch.tensor([float(x) for x in w_sub], device=self.device, dtype=torch.float32)
                    except Exception:
                        tw_sub = None
                if w_grade is not None:
                    try:
                        tw_grade = torch.tensor([float(x) for x in w_grade], device=self.device, dtype=torch.float32)
                    except Exception:
                        tw_grade = None


                end_noncarb_w = float(self.model_config.get("hierarchical_endmember_noncarbonate_weight", 1.0) or 1.0)
                if np.isfinite(end_noncarb_w) and end_noncarb_w > 0 and abs(end_noncarb_w - 1.0) > 1e-9:
                    try:
                        if tw_end is None:
                            tw_end = torch.tensor([1.0, end_noncarb_w], device=self.device, dtype=torch.float32)
                        elif int(tw_end.numel()) == 2:
                            tw_end = tw_end.clone()
                            tw_end[1] = tw_end[1] * float(end_noncarb_w)
                        self.logger.info(
                            f"Trainer: Applying hierarchical_endmember_noncarbonate_weight={end_noncarb_w:g} "
                            f"to endmember CE weights: {tw_end.tolist() if tw_end is not None else None}"
                        )
                    except Exception:
                        pass
                eval_criterion = {
                    "end": nn.CrossEntropyLoss(weight=tw_end, label_smoothing=label_smoothing),
                    "sub": nn.CrossEntropyLoss(weight=tw_sub, label_smoothing=label_smoothing),
                }
                if self.hierarchical_grade_multitask:
                    eval_criterion["grade"] = nn.CrossEntropyLoss(weight=tw_grade, label_smoothing=label_smoothing)
            else:
                class_weights = None
                cw = self.model_config.get("ce_class_weights", None)
                if cw is not None:
                    try:
                        class_weights = torch.tensor([float(x) for x in cw], device=self.device, dtype=torch.float32)
                    except Exception:
                        class_weights = None
                eval_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        elif loss_function_type == "bce":
            eval_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss_function_type == 'msle':
            eval_criterion = MSLELoss()
        elif loss_function_type == "huber":
            eval_criterion = HuberLoss(delta=huber_delta)
        else:
            eval_criterion = nn.MSELoss()


        if self.hierarchical_multiclass and self.task_type == "multiclass_classification":
            sub_loss = str(self.model_config.get("hierarchical_sub_loss", "ce") or "ce").lower()
            sub_focal_gamma = float(self.model_config.get("hierarchical_sub_focal_gamma", 3.0) or 3.0)
            if classification_loss != "ce":
                self.logger.warning(
                    f"Trainer: hierarchical_multiclass expects classification_loss='ce' (for full model), got '{classification_loss}'. "
                    "Proceeding with hierarchical head-specific settings."
                )
            required_keys = {"end", "sub"} | ({"grade"} if self.hierarchical_grade_multitask else set())
            if not isinstance(eval_criterion, dict) or not required_keys.issubset(set(eval_criterion.keys())):
                raise ValueError(f"Trainer: hierarchical_multiclass requires dict criterions {sorted(required_keys)}.")

            train_criterion = dict(eval_criterion)
            if sub_loss == "focal":

                try:
                    class_weights_sub = eval_criterion["sub"].weight
                except Exception:
                    class_weights_sub = None
                train_criterion["sub"] = FocalCrossEntropyLoss(
                    gamma=sub_focal_gamma,
                    class_weights=class_weights_sub,
                    label_smoothing=label_smoothing,
                )
                self.logger.info(
                    f"Trainer: Hierarchical training losses: end=CE, sub=FocalCE(gamma={sub_focal_gamma:g})."
                )
            else:
                self.logger.info(
                    "Trainer: Hierarchical training losses: "
                    f"end=CE, sub=CE{', grade=CE' if self.hierarchical_grade_multitask else ''}."
                )
        elif self.task_type == "multiclass_classification" and classification_loss == "focal":
            class_weights = None
            cw = self.model_config.get("ce_class_weights", None)
            if cw is not None:
                try:
                    class_weights = torch.tensor([float(x) for x in cw], device=self.device, dtype=torch.float32)
                except Exception:
                    class_weights = None
            self.logger.info(f"Trainer: Using Focal CrossEntropy loss for training (gamma={focal_gamma:g}).")
            train_criterion = FocalCrossEntropyLoss(
                gamma=focal_gamma,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
        elif self.task_type == "multiclass_classification" and classification_loss in {
            "gce",
            "generalized_ce",
            "bootstrap",
            "bootstrap_soft",
            "bootstrap_hard",
            "balanced_softmax",
            "logit_adjusted",
            "class_balanced_effective",
            "effective_number",
        }:
            class_weights = None
            cw = self.model_config.get("ce_class_weights", None)
            if cw is not None and classification_loss in {"gce", "generalized_ce", "bootstrap", "bootstrap_soft", "bootstrap_hard"}:
                try:
                    class_weights = torch.tensor([float(x) for x in cw], device=self.device, dtype=torch.float32)
                except Exception:
                    class_weights = None
            counts_raw = self.model_config.get("class_counts", None)
            class_counts = None
            if counts_raw is not None:
                try:
                    class_counts = torch.tensor([float(x) for x in counts_raw], device=self.device, dtype=torch.float32)
                except Exception:
                    class_counts = None
            if class_counts is None and self.num_classes:
                class_counts = torch.ones(int(self.num_classes), device=self.device, dtype=torch.float32)

            if classification_loss in {"gce", "generalized_ce"}:
                q = float(self.model_config.get("gce_q", 0.7) or 0.7)
                self.logger.info(f"Trainer: Using Generalized CrossEntropy loss for training (q={q:g}).")
                train_criterion = GeneralizedCrossEntropyLoss(q=q, class_weights=class_weights)
            elif classification_loss in {"bootstrap", "bootstrap_soft", "bootstrap_hard"}:
                beta = float(self.model_config.get("bootstrap_beta", 0.8) or 0.8)
                mode = "hard" if classification_loss == "bootstrap_hard" else str(self.model_config.get("bootstrap_mode", "soft") or "soft")
                self.logger.info(f"Trainer: Using Bootstrap CrossEntropy loss for training (beta={beta:g}, mode={mode}).")
                train_criterion = BootstrapCrossEntropyLoss(beta=beta, mode=mode, class_weights=class_weights)
            elif classification_loss == "balanced_softmax":
                self.logger.info(
                    "Trainer: Using Balanced Softmax loss for training "
                    f"(class_counts={class_counts.detach().cpu().tolist() if class_counts is not None else None})."
                )
                train_criterion = BalancedSoftmaxLoss(class_counts=class_counts, label_smoothing=label_smoothing)
            elif classification_loss == "logit_adjusted":
                tau = float(self.model_config.get("logit_adjustment_tau", 1.0) or 1.0)
                self.logger.info(
                    "Trainer: Using Logit-Adjusted CrossEntropy loss for training "
                    f"(tau={tau:g}, class_counts={class_counts.detach().cpu().tolist() if class_counts is not None else None})."
                )
                train_criterion = LogitAdjustedCrossEntropyLoss(class_counts=class_counts, tau=tau, label_smoothing=label_smoothing)
            else:
                beta = float(self.model_config.get("effective_number_beta", 0.999) or 0.999)
                self.logger.info(
                    "Trainer: Using Class-Balanced Effective-Number loss for training "
                    f"(beta={beta:g}, class_counts={class_counts.detach().cpu().tolist() if class_counts is not None else None})."
                )
                train_criterion = ClassBalancedEffectiveNumberLoss(
                    class_counts=class_counts,
                    beta=beta,
                    label_smoothing=label_smoothing,
                )
        elif self.use_weighted_loss or self.domain_priority_enabled:
            if loss_function_type == "ce":
                self.logger.info("Trainer: Using Weighted CrossEntropy loss for training.")
                class_weights = None
                cw = self.model_config.get("ce_class_weights", None)
                if cw is not None:
                    try:
                        class_weights = torch.tensor([float(x) for x in cw], device=self.device, dtype=torch.float32)
                    except Exception:
                        class_weights = None
                train_criterion = WeightedCrossEntropyLoss(class_weights=class_weights)
            elif loss_function_type == "bce":
                self.logger.info("Trainer: Using Weighted BCEWithLogits loss for training.")
                train_criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
            elif loss_function_type == 'msle':
                self.logger.info("Trainer: Using Weighted MSLE loss for training.")
                train_criterion = WeightedMSLELoss()
            elif loss_function_type == "huber":
                self.logger.info("Trainer: Using Weighted Huber loss for training.")
                train_criterion = WeightedHuberLoss(delta=huber_delta)
            else:
                self.logger.info("Trainer: Using Weighted MSE loss for training.")
                train_criterion = WeightedMSELoss()
        else:
            self.logger.info(f"Trainer: Using standard {loss_function_type.upper()} loss for training.")
            train_criterion = copy.deepcopy(eval_criterion)

        return train_criterion, eval_criterion

    def _build_scheduler(self):
        scheduler_patience = self.model_config.get('lr_scheduler_patience', 10)
        scheduler_factor = self.model_config.get('lr_scheduler_factor', 0.1)
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=scheduler_factor,
            patience=scheduler_patience
        )
        return scheduler

    def _update_ema_state(self) -> None:
        if not self.use_ema or self.ema_state_dict is None:
            return
        decay = float(np.clip(self.ema_decay, 0.0, 0.999999))
        with torch.no_grad():
            current = self.model.state_dict()
            for key, value in current.items():
                if key in self.ema_state_dict and torch.is_floating_point(value):
                    self.ema_state_dict[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    def _update_swa_state(self) -> None:
        if not self.use_swa:
            return
        current = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
            if torch.is_floating_point(v)
        }
        if self.swa_state_dict is None:
            self.swa_state_dict = current
            self.swa_n = 1
            return
        self.swa_n += 1
        alpha = 1.0 / float(self.swa_n)
        with torch.no_grad():
            for key, value in current.items():
                if key in self.swa_state_dict:
                    self.swa_state_dict[key].mul_(1.0 - alpha).add_(value, alpha=alpha)

    def _merged_state_dict(self, averaged_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = copy.deepcopy(self.model.state_dict())
        for key, value in averaged_state.items():
            if key in state:
                state[key] = value.detach().clone()
        return state

    def train_and_evaluate(
        self,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        early_stopping_patience,
        full_dataset_for_denorm,
        train_eval_loader=None,
    ):
        self.logger.info("Trainer: Starting model training and evaluation...")
        best_val_loss = float('inf')
        best_val_metric = -float("inf")
        best_epoch = -1
        best_epoch_val_loss = None
        epochs_no_improve = 0
        log_interval = max(1, epochs // 20) if epochs > 20 else 1

        train_losses_history = []
        val_losses_history = []
        best_model_state_dict = None
        training_interrupted = False

        try:
            effective_epochs = int(epochs)
            if self.domain_priority_enabled and self.domain_priority_mode == "two_stage_finetune":
                pre = max(0, int(self.domain_priority_pretrain_epochs))
                fin = max(0, int(self.domain_priority_finetune_epochs))
                if pre + fin > 0:
                    effective_epochs = pre + fin
            for epoch in range(effective_epochs):
                if self.stop_flag and self.stop_flag[0]:
                    self.logger.warning("Trainer: Stop requested, interrupting training.")
                    training_interrupted = True
                    break


                if (
                    self.domain_priority_enabled
                    and self.domain_priority_mode == "two_stage_finetune"
                    and epoch == int(self.domain_priority_pretrain_epochs)
                    and 0.0 < float(self.domain_priority_finetune_lr_mult) < 1.0
                ):
                    try:
                        mult = float(self.domain_priority_finetune_lr_mult)
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = float(pg.get("lr", 0.0)) * mult
                        self.logger.info(f"Trainer: Entering finetune stage; scaled lr by {mult:g}.")
                    except Exception as e_lr:
                        self.logger.warning(f"Trainer: Failed to apply finetune lr multiplier: {e_lr}")

                self.model.train()
                running_train_loss = 0.0
                processed_train_samples = 0
                for i, data in enumerate(train_loader):

                    combined_patches, target_reys, sample_weights = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                    soft_targets = data[3].to(self.device) if (len(data) > 3 and self.use_soft_labels) else None
                    current_batch_size = combined_patches.size(0)

                    self.optimizer.zero_grad()
                with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                        if (
                            self.task_type == "multiclass_classification"
                            and not self.hierarchical_multiclass
                            and self.use_high_grade_ranking_head
                        ):
                            outputs, comps_train = self.model(combined_patches, return_components=True)
                        else:
                            outputs = self.model(combined_patches)
                            comps_train = None

                        targets_for_loss = target_reys
                        weights_for_loss = sample_weights
                        if self.task_type == "multiclass_classification":
                            targets_for_loss = target_reys.view(-1).long()
                            weights_for_loss = sample_weights.view(-1)
                        elif self.task_type == "binary_classification":
                            targets_for_loss = target_reys
                            weights_for_loss = sample_weights

                        use_weights_this_epoch = bool(self.use_weighted_loss)
                        if self.domain_priority_enabled:
                            if self.domain_priority_mode == "loss_weighting":
                                use_weights_this_epoch = True
                            elif self.domain_priority_mode == "two_stage_finetune":
                                use_weights_this_epoch = epoch >= int(self.domain_priority_pretrain_epochs)

                        if self.hierarchical_multiclass and self.task_type == "multiclass_classification":
                            end_logits, sub_logits = outputs[:2]
                            grade_logits = outputs[2] if self.hierarchical_grade_multitask else None
                            y_full = targets_for_loss
                            y_end, y_sub, y_grade, mask = _hierarchical_targets(
                                y_full,
                                grade_multitask=self.hierarchical_grade_multitask,
                            )
                            label_smoothing = float(self.model_config.get("label_smoothing", 0.0) or 0.0)
                            loss_end = _head_ce_loss(
                                self.train_criterion["end"],
                                end_logits,
                                y_end,
                                weights_for_loss,
                                use_sample_weights=use_weights_this_epoch,
                                label_smoothing=label_smoothing,
                            )
                            if bool(mask.any()):
                                w_non = weights_for_loss.view(-1).float()[mask]
                                loss_sub = _head_ce_loss(
                                    self.train_criterion["sub"],
                                    sub_logits[mask],
                                    y_sub,
                                    w_non,
                                    use_sample_weights=use_weights_this_epoch,
                                    label_smoothing=label_smoothing,
                                )
                                if self.hierarchical_grade_multitask and grade_logits is not None and y_grade is not None:
                                    loss_grade = _head_ce_loss(
                                        self.train_criterion["grade"],
                                        grade_logits[mask],
                                        y_grade,
                                        w_non,
                                        use_sample_weights=use_weights_this_epoch,
                                        label_smoothing=label_smoothing,
                                    )
                                else:
                                    loss_grade = loss_end * 0.0
                            else:
                                loss_sub = loss_end * 0.0
                                loss_grade = loss_end * 0.0
                            w_end = float(self.model_config.get("hierarchical_end_loss_weight", 1.0) or 1.0)
                            w_sub = float(self.model_config.get("hierarchical_sub_loss_weight", 1.0) or 1.0)
                            w_grade = float(self.model_config.get("hierarchical_grade_loss_weight", 1.0) or 1.0)
                            loss = (w_end * loss_end) + (w_sub * loss_sub) + (w_grade * loss_grade)
                            w_full = float(self.model_config.get("hierarchical_full_ce_weight", 0.0) or 0.0)
                            if np.isfinite(w_full) and w_full > 0.0:
                                cw_full = self.model_config.get("ce_class_weights", None)
                                tw_full = None
                                if cw_full is not None:
                                    try:
                                        tw_full = torch.tensor(
                                            [float(x) for x in cw_full],
                                            device=self.device,
                                            dtype=torch.float32,
                                        )
                                    except Exception:
                                        tw_full = None
                                loss_full = _hierarchical_full_nll_loss(
                                    outputs,
                                    y_full,
                                    weights_for_loss,
                                    use_sample_weights=use_weights_this_epoch,
                                    class_weights=tw_full,
                                    grade_multitask=self.hierarchical_grade_multitask,
                                )
                                loss = loss + (w_full * loss_full)
                        elif (
                            soft_targets is not None
                            and self.task_type == "multiclass_classification"
                            and not self.hierarchical_multiclass
                        ):
                            loss = _soft_target_cross_entropy(
                                outputs,
                                soft_targets,
                                weights_for_loss if use_weights_this_epoch else None,
                                class_weights=self.soft_label_class_weights,
                                label_smoothing=float(self.model_config.get("label_smoothing", 0.0) or 0.0),
                            )
                        elif use_weights_this_epoch:
                            try:
                                loss = self.train_criterion(outputs, targets_for_loss, weights_for_loss)
                            except TypeError:
                                loss = self.train_criterion(outputs, targets_for_loss)
                        else:
                            loss = self.train_criterion(outputs, targets_for_loss)

                        if (
                            self.binary_logit_margin_loss_weight > 0.0
                            and self.task_type == "multiclass_classification"
                            and not self.hierarchical_multiclass
                            and int(outputs.shape[1]) == 2
                        ):
                            margin_loss = _binary_logit_margin_loss(
                                outputs,
                                targets_for_loss,
                                weights_for_loss if use_weights_this_epoch else None,
                                margin=self.binary_logit_margin,
                            )
                            loss = loss + (self.binary_logit_margin_loss_weight * margin_loss)

                        if (
                            self.aux_high_grade_bce_weight > 0.0
                            and self.task_type == "multiclass_classification"
                            and not self.hierarchical_multiclass
                        ):
                            target_k = int(np.clip(self.aux_high_grade_class_index, 0, int(outputs.shape[1]) - 1))
                            aux_logits = _one_vs_rest_logits(outputs, target_k)
                            aux_targets = (targets_for_loss.view(-1).long() == target_k).float()
                            aux_weights = weights_for_loss if use_weights_this_epoch else None
                            aux_loss = _weighted_bce_with_logits(
                                aux_logits,
                                aux_targets,
                                aux_weights,
                                pos_weight=self.aux_high_grade_pos_weight,
                            )
                            loss = loss + (self.aux_high_grade_bce_weight * aux_loss)

                        if (
                            self.aux_binary_tasks
                            and self.task_type == "multiclass_classification"
                            and not self.hierarchical_multiclass
                        ):
                            y_aux_base = targets_for_loss.view(-1).long()
                            aux_weights = weights_for_loss if use_weights_this_epoch else None
                            for aux_task in self.aux_binary_tasks:
                                positive_classes = [int(x) for x in aux_task["positive_classes"]]
                                aux_logits = _multi_vs_rest_logits(outputs, positive_classes)
                                aux_targets = torch.zeros_like(y_aux_base, dtype=torch.float32)
                                for cls_idx in positive_classes:
                                    aux_targets = torch.where(
                                        y_aux_base == int(cls_idx),
                                        torch.ones_like(aux_targets),
                                        aux_targets,
                                    )
                                aux_loss = _weighted_bce_with_logits(
                                    aux_logits,
                                    aux_targets,
                                    aux_weights,
                                    pos_weight=aux_task.get("pos_weight"),
                                )
                                loss = loss + (float(aux_task["weight"]) * aux_loss)

                        if (
                            self.aux_main_class_ce_weight > 0.0
                            and self.task_type == "multiclass_classification"
                            and not self.hierarchical_multiclass
                            and int(outputs.shape[1]) == 4
                        ):
                            main_logits = _exploration4_main_logits(outputs)
                            main_targets = _exploration4_main_targets(targets_for_loss)
                            loss_vec = F.cross_entropy(
                                main_logits,
                                main_targets,
                                weight=self.aux_main_class_ce_weights,
                                reduction="none",
                                label_smoothing=float(self.model_config.get("label_smoothing", 0.0) or 0.0),
                            )
                            if use_weights_this_epoch:
                                aux_weights = weights_for_loss.view(-1).float()
                                loss_main = torch.sum(loss_vec * aux_weights) / torch.clamp(aux_weights.sum(), min=1e-12)
                            else:
                                loss_main = torch.mean(loss_vec)
                            loss = loss + (self.aux_main_class_ce_weight * loss_main)

                        if (
                            self.use_high_grade_ranking_head
                            and self.high_grade_ranking_loss_weight > 0.0
                            and self.task_type == "multiclass_classification"
                            and not self.hierarchical_multiclass
                        ):
                            rank_logits = None
                            if isinstance(comps_train, dict):
                                rank_logits = comps_train.get("high_grade_ranking_logit", None)
                            if rank_logits is not None:
                                rank_targets = (
                                    targets_for_loss.view(-1).long() == int(self.high_grade_ranking_target_class_index)
                                ).float()
                                rank_weights = weights_for_loss if use_weights_this_epoch else None
                                rank_loss = _weighted_bce_with_logits(
                                    rank_logits,
                                    rank_targets,
                                    rank_weights,
                                    pos_weight=self.high_grade_ranking_pos_weight,
                                )
                                loss = loss + (self.high_grade_ranking_loss_weight * rank_loss)


                        if self.use_quantile_loss and self.quantile_loss_weight > 0.0 and current_batch_size > 1:
                            try:
                                y_pred_flat = outputs.view(-1)
                                y_true_flat = target_reys.view(-1)
                                q = self.quantile_level
                                q_true = torch.quantile(y_true_flat.detach(), q)
                                q_pred = torch.quantile(y_pred_flat, q)
                                quantile_penalty = (q_true - q_pred) ** 2
                                loss = loss + self.quantile_loss_weight * quantile_penalty
                            except Exception as e_q:
                                self.logger.debug(
                                    f"Trainer: Quantile loss computation skipped due to error: {e_q}"
                                )

                        if torch.isnan(loss):
                            self.logger.error(f"Trainer: NaN loss encountered during training epoch {epoch+1}, batch {i+1}. Stopping.")
                            raise ValueError("NaN loss encountered in trainer")

                        self.scaler.scale(loss).backward()


                        self.scaler.unscale_(self.optimizer)


                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.model_config.get('clip_grad_norm', 1.0))

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self._update_ema_state()
                        running_train_loss += loss.item() * current_batch_size
                        processed_train_samples += current_batch_size

                epoch_train_loss = running_train_loss / processed_train_samples if processed_train_samples > 0 else 0.0
                train_losses_history.append(epoch_train_loss)
                if self.use_swa and (epoch + 1) >= int(self.swa_start_epoch):
                    self._update_swa_state()

                self.model.eval()
                running_val_loss = 0.0
                processed_val_samples = 0
                val_logits = []
                val_targets = []
                val_ranking_logits = []
                with torch.no_grad():
                    for data in val_loader:

                        combined_patches, target_reys, _ = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                        current_batch_size = combined_patches.size(0)
                        with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                            if (
                                self.task_type == "multiclass_classification"
                                and not self.hierarchical_multiclass
                                and self.use_high_grade_ranking_head
                            ):
                                outputs, comps_val = self.model(combined_patches, return_components=True)
                            else:
                                outputs = self.model(combined_patches)
                                comps_val = None
                            if self.task_type == "multiclass_classification":
                                yb = target_reys.view(-1).long()
                                if self.hierarchical_multiclass:
                                    end_logits, sub_logits = outputs[:2]
                                    grade_logits = outputs[2] if self.hierarchical_grade_multitask else None
                                    y_end, y_sub, y_grade, mask = _hierarchical_targets(
                                        yb,
                                        grade_multitask=self.hierarchical_grade_multitask,
                                    )
                                    loss_end = self.eval_criterion["end"](end_logits, y_end)
                                    if bool(mask.any()):
                                        loss_sub = self.eval_criterion["sub"](sub_logits[mask], y_sub)
                                        if self.hierarchical_grade_multitask and grade_logits is not None and y_grade is not None:
                                            loss_grade = self.eval_criterion["grade"](grade_logits[mask], y_grade)
                                        else:
                                            loss_grade = loss_end * 0.0
                                    else:
                                        loss_sub = loss_end * 0.0
                                        loss_grade = loss_end * 0.0
                                    w_end = float(self.model_config.get("hierarchical_end_loss_weight", 1.0) or 1.0)
                                    w_sub = float(self.model_config.get("hierarchical_sub_loss_weight", 1.0) or 1.0)
                                    w_grade = float(self.model_config.get("hierarchical_grade_loss_weight", 1.0) or 1.0)
                                    loss = (w_end * loss_end) + (w_sub * loss_sub) + (w_grade * loss_grade)
                                    w_full = float(self.model_config.get("hierarchical_full_ce_weight", 0.0) or 0.0)
                                    if np.isfinite(w_full) and w_full > 0.0:
                                        cw_full = self.model_config.get("ce_class_weights", None)
                                        tw_full = None
                                        if cw_full is not None:
                                            try:
                                                tw_full = torch.tensor(
                                                    [float(x) for x in cw_full],
                                                    device=self.device,
                                                    dtype=torch.float32,
                                                )
                                            except Exception:
                                                tw_full = None
                                        loss_full = _hierarchical_full_nll_loss(
                                            outputs,
                                            yb,
                                            None,
                                            use_sample_weights=False,
                                            class_weights=tw_full,
                                            grade_multitask=self.hierarchical_grade_multitask,
                                        )
                                        loss = loss + (w_full * loss_full)

                                    val_logits.append(
                                        _hierarchical_outputs_to_log_probs(
                                            outputs,
                                            grade_multitask=self.hierarchical_grade_multitask,
                                        )
                                        .detach()
                                        .cpu()
                                        .numpy()
                                    )
                                else:
                                    loss = self.eval_criterion(outputs, yb)
                                    val_logits.append(outputs.detach().cpu().numpy())
                                    if isinstance(comps_val, dict):
                                        rank_logits_val = comps_val.get("high_grade_ranking_logit", None)
                                        if rank_logits_val is not None:
                                            val_ranking_logits.append(rank_logits_val.view(-1).detach().cpu().numpy())
                                val_targets.append(yb.detach().cpu().numpy())
                            else:
                                loss = self.eval_criterion(outputs, target_reys)
                        running_val_loss += loss.item() * current_batch_size
                        processed_val_samples += current_batch_size

                epoch_val_loss = running_val_loss / processed_val_samples if processed_val_samples > 0 else 0.0
                val_losses_history.append(epoch_val_loss)


                if best_epoch_val_loss is None or epoch_val_loss < float(best_epoch_val_loss):
                    best_epoch_val_loss = float(epoch_val_loss)

                current_lr = self.optimizer.param_groups[0]['lr']


                val_metric_value = None
                val_macro_f1 = None
                val_balanced_accuracy = None
                val_class1_f1 = None
                val_class1_ap = None
                val_target_class_f1 = None
                val_target_class_ap = None
                val_high_grade_ap = None
                val_high_grade_f1 = None
                val_high_grade_ranking_ap = None
                val_high_grade_ranking_top10_capture = None
                val_high_grade_ranking_top20_capture = None
                val_composite = None
                if self.task_type == "multiclass_classification" and val_logits and val_targets:
                    try:
                        logits_np = np.concatenate(val_logits, axis=0)
                        y_val = np.concatenate(val_targets, axis=0).astype(int).reshape(-1)
                        z = logits_np - np.max(logits_np, axis=1, keepdims=True)
                        ez = np.exp(np.clip(z, -60.0, 60.0))
                        prob_val = ez / np.clip(np.sum(ez, axis=1, keepdims=True), 1e-12, None)
                        yhat_val = np.argmax(prob_val, axis=1)
                        target_k = int(self.model_config.get("explain_target_class_index", 2) or 2)
                        if not (0 <= target_k < prob_val.shape[1]):
                            target_k = 1 if prob_val.shape[1] > 1 else 0
                        val_macro_f1 = float(f1_score(y_val, yhat_val, average="macro", zero_division=0))
                        val_balanced_accuracy = float(balanced_accuracy_score(y_val, yhat_val))
                        val_class1_f1 = float(
                            f1_score((y_val == 1).astype(int), (yhat_val == 1).astype(int), zero_division=0)
                        )
                        y_class1 = (y_val == 1).astype(int)
                        val_class1_ap = (
                            float(average_precision_score(y_class1, prob_val[:, 1]))
                            if prob_val.shape[1] > 1 and int(np.unique(y_class1).size) >= 2
                            else float("nan")
                        )
                        y_target = (y_val == target_k).astype(int)
                        val_target_class_f1 = float(
                            f1_score(y_target, (yhat_val == target_k).astype(int), zero_division=0)
                        )
                        val_target_class_ap = (
                            float(average_precision_score(y_target, prob_val[:, target_k]))
                            if prob_val.shape[1] > target_k and int(np.unique(y_target).size) >= 2
                            else float("nan")
                        )
                        try:
                            hg_val = compute_high_grade_metrics(
                                y_true=y_val,
                                prob=prob_val,
                                class_names=self.classification_class_names,
                            )
                            if hg_val:
                                val_high_grade_ap = float(hg_val.get("average_precision", float("nan")))
                                val_high_grade_f1 = float(hg_val.get("f1", float("nan")))
                        except Exception:
                            val_high_grade_ap = None
                            val_high_grade_f1 = None
                        if val_ranking_logits:
                            try:
                                ranking_logits_np = np.concatenate(val_ranking_logits, axis=0).reshape(-1)
                                ranking_prob_np = 1.0 / (1.0 + np.exp(-np.clip(ranking_logits_np, -60.0, 60.0)))
                                y_rank = (y_val == int(self.high_grade_ranking_target_class_index)).astype(int)
                                if int(np.unique(y_rank).size) >= 2:
                                    val_high_grade_ranking_ap = float(average_precision_score(y_rank, ranking_prob_np))
                                val_high_grade_ranking_top10_capture = _top_fraction_capture_from_binary(
                                    y_rank,
                                    ranking_prob_np,
                                    0.10,
                                )
                                val_high_grade_ranking_top20_capture = _top_fraction_capture_from_binary(
                                    y_rank,
                                    ranking_prob_np,
                                    0.20,
                                )
                            except Exception:
                                val_high_grade_ranking_ap = None
                                val_high_grade_ranking_top10_capture = None
                                val_high_grade_ranking_top20_capture = None
                        vals_for_composite = [
                            val_macro_f1,
                            val_balanced_accuracy,
                            val_high_grade_ap if val_high_grade_ap is not None else val_target_class_ap,
                        ]
                        if all(v is not None and np.isfinite(v) for v in vals_for_composite):
                            val_composite = (
                                0.45 * float(val_macro_f1)
                                + 0.35 * float(val_balanced_accuracy)
                                + 0.20 * float(vals_for_composite[2])
                            )

                        if self.early_stopping_metric == "val_macro_f1":
                            val_metric_value = val_macro_f1
                        elif self.early_stopping_metric == "val_balanced_accuracy":
                            val_metric_value = val_balanced_accuracy
                        elif self.early_stopping_metric in {"val_class1_f1", "val_target_class_f1"}:
                            val_metric_value = val_target_class_f1
                        elif self.early_stopping_metric in {"val_class1_ap", "val_target_class_ap"}:
                            val_metric_value = val_target_class_ap
                        elif self.early_stopping_metric == "val_high_grade_ap":
                            val_metric_value = val_high_grade_ap
                        elif self.early_stopping_metric == "val_high_grade_ranking_ap":
                            val_metric_value = val_high_grade_ranking_ap
                        elif self.early_stopping_metric == "val_high_grade_f1":
                            val_metric_value = val_high_grade_f1
                        elif self.early_stopping_metric in {"val_composite", "val_exploration_composite"}:
                            val_metric_value = val_composite

                    except Exception as e_vm:
                        self.logger.debug(f"Trainer: Failed to compute val metrics: {e_vm}")


                epoch_record = {
                    "epoch": epoch + 1,
                    "train_loss": float(epoch_train_loss),
                    "val_loss": float(epoch_val_loss),
                    "lr": float(current_lr),
                }
                if val_macro_f1 is not None:
                    epoch_record["val_macro_f1"] = float(val_macro_f1)
                if val_balanced_accuracy is not None:
                    epoch_record["val_balanced_accuracy"] = float(val_balanced_accuracy)
                if val_class1_f1 is not None:
                    epoch_record["val_class1_f1"] = float(val_class1_f1)
                if val_class1_ap is not None:
                    epoch_record["val_class1_ap"] = float(val_class1_ap)
                if val_target_class_f1 is not None:
                    epoch_record["val_target_class_f1"] = float(val_target_class_f1)
                if val_target_class_ap is not None:
                    epoch_record["val_target_class_ap"] = float(val_target_class_ap)
                if val_high_grade_ap is not None:
                    epoch_record["val_high_grade_ap"] = float(val_high_grade_ap)
                if val_high_grade_f1 is not None:
                    epoch_record["val_high_grade_f1"] = float(val_high_grade_f1)
                if val_high_grade_ranking_ap is not None:
                    epoch_record["val_high_grade_ranking_ap"] = float(val_high_grade_ranking_ap)
                if val_high_grade_ranking_top10_capture is not None:
                    epoch_record["val_high_grade_ranking_top10_capture"] = float(val_high_grade_ranking_top10_capture)
                if val_high_grade_ranking_top20_capture is not None:
                    epoch_record["val_high_grade_ranking_top20_capture"] = float(val_high_grade_ranking_top20_capture)
                if val_composite is not None:
                    epoch_record["val_composite"] = float(val_composite)
                self.training_history.append(epoch_record)

                log_msg = f"Trainer: Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, LR: {current_lr:.7f}"
                if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                    self.logger.debug(log_msg)

                self.scheduler.step(epoch_val_loss)

                improved = False
                if val_metric_value is not None and np.isfinite(val_metric_value):
                    if float(val_metric_value) > float(best_val_metric):
                        best_val_metric = float(val_metric_value)
                        improved = True
                else:
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss
                        improved = True

                if improved:
                    best_epoch = epoch + 1


                    epochs_no_improve = 0
                    best_model_state_dict = copy.deepcopy(self.model.state_dict())
                    self.logger.debug(
                        f"Trainer: Best model updated at epoch {best_epoch} "
                        f"(epoch_val_loss={float(epoch_val_loss):.6f}, best_val_loss={best_val_loss:.6f}, "
                        f"val_metric={best_val_metric if best_val_metric!=-float('inf') else None})."
                    )
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        self.logger.warning(
                            f"Trainer: Early stopping triggered after {early_stopping_patience} epochs with no improvement. "
                            f"Best epoch: {best_epoch}, best_val_loss: {best_val_loss:.6f}."
                        )
                        break

                if self.progress_callback:
                    progress_fraction = ((epoch + 1) / epochs) * 0.8
                    self.progress_callback(progress_fraction, f"Epoch {epoch+1}/{epochs}, Val Loss: {epoch_val_loss:.4f}")

                if self.history_callback:
                    self.history_callback(self.training_history)

            self.logger.debug("Trainer: Training loop finished.")

        except ValueError as ve:
            self.logger.error(f"Trainer: Training stopped due to error: {ve}")
            return False, {"status": "error_nan_loss_trainer", "train_losses": train_losses_history, "val_losses": val_losses_history}
        except Exception as e:
            self.logger.error(f"Trainer: Error during training loop: {e}", exc_info=True)
            return False, {"status": "error_training_loop", "train_losses": train_losses_history, "val_losses": val_losses_history, "error_message": str(e)}

        if best_model_state_dict:
            self.model.load_state_dict(best_model_state_dict)
            self.logger.debug(f"Trainer: Loaded best model state (epoch {best_epoch}) for final evaluation.")
        else:
            self.logger.warning("Trainer: No best model state saved. Evaluating with final weights from training.")
        if bool(self.model_config.get("use_swa_for_eval", False)) and self.swa_state_dict is not None:
            self.model.load_state_dict(self._merged_state_dict(self.swa_state_dict))
            self.logger.info(f"Trainer: Loaded SWA weights for final evaluation (n={self.swa_n}).")
        elif bool(self.model_config.get("use_ema_for_eval", False)) and self.ema_state_dict is not None:
            self.model.load_state_dict(self._merged_state_dict(self.ema_state_dict))
            self.logger.info("Trainer: Loaded EMA weights for final evaluation.")

        self.logger.info("Trainer: Evaluating model on test and train sets for final plots...")
        self.model.eval()


        all_test_targets_norm, all_test_preds_norm, all_test_norm_depths = [], [], []
        all_test_ranking_logits = []
        with torch.no_grad():
            for data in test_loader:

                combined_patches, target_reys, _ = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    if (
                        self.task_type == "multiclass_classification"
                        and not self.hierarchical_multiclass
                        and self.use_high_grade_ranking_head
                    ):
                        outputs, comps_test = self.model(combined_patches, return_components=True)
                    else:
                        outputs = self.model(combined_patches)
                        comps_test = None
                if self.task_type == "multiclass_classification":
                    if self.hierarchical_multiclass:
                        all_test_preds_norm.append(
                            _hierarchical_outputs_to_log_probs(
                                outputs,
                                grade_multitask=self.hierarchical_grade_multitask,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        all_test_preds_norm.append(outputs.detach().cpu().numpy())
                        if isinstance(comps_test, dict):
                            rank_logits_test = comps_test.get("high_grade_ranking_logit", None)
                            if rank_logits_test is not None:
                                all_test_ranking_logits.append(rank_logits_test.view(-1).detach().cpu().numpy())
                    all_test_targets_norm.append(target_reys.view(-1).detach().cpu().numpy())
                else:
                    all_test_preds_norm.append(outputs.squeeze(-1).cpu().numpy())
                    all_test_targets_norm.append(target_reys.squeeze(-1).cpu().numpy())


        if self.task_type == "multiclass_classification":
            all_test_preds_np_norm = np.concatenate(all_test_preds_norm, axis=0) if all_test_preds_norm else np.zeros((0, max(self.num_classes, 1)), dtype=float)
            all_test_targets_np_norm = np.concatenate(all_test_targets_norm, axis=0) if all_test_targets_norm else np.array([], dtype=float)
        else:
            all_test_preds_np_norm = np.concatenate(all_test_preds_norm) if all_test_preds_norm else np.array([])
            all_test_targets_np_norm = np.concatenate(all_test_targets_norm) if all_test_targets_norm else np.array([])
        all_test_depths_np_norm = np.concatenate(all_test_norm_depths) if all_test_norm_depths else np.array([])


        train_eval_loader = train_eval_loader or train_loader
        all_train_targets_norm, all_train_preds_norm = [], []
        all_train_ranking_logits = []
        with torch.no_grad():
            for data in train_eval_loader:

                combined_patches, target_reys, _ = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    if (
                        self.task_type == "multiclass_classification"
                        and not self.hierarchical_multiclass
                        and self.use_high_grade_ranking_head
                    ):
                        outputs, comps_train_eval = self.model(combined_patches, return_components=True)
                    else:
                        outputs = self.model(combined_patches)
                        comps_train_eval = None
                if self.task_type == "multiclass_classification":
                    if self.hierarchical_multiclass:
                        all_train_preds_norm.append(
                            _hierarchical_outputs_to_log_probs(
                                outputs,
                                grade_multitask=self.hierarchical_grade_multitask,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        all_train_preds_norm.append(outputs.detach().cpu().numpy())
                        if isinstance(comps_train_eval, dict):
                            rank_logits_train = comps_train_eval.get("high_grade_ranking_logit", None)
                            if rank_logits_train is not None:
                                all_train_ranking_logits.append(rank_logits_train.view(-1).detach().cpu().numpy())
                    all_train_targets_norm.append(target_reys.view(-1).detach().cpu().numpy())
                else:
                    all_train_preds_norm.append(outputs.squeeze(-1).cpu().numpy())
                    all_train_targets_norm.append(target_reys.squeeze(-1).cpu().numpy())

        if self.task_type == "multiclass_classification":
            all_train_preds_np_norm = np.concatenate(all_train_preds_norm, axis=0) if all_train_preds_norm else np.zeros((0, max(self.num_classes, 1)), dtype=float)
            all_train_targets_np_norm = np.concatenate(all_train_targets_norm, axis=0) if all_train_targets_norm else np.array([], dtype=float)
        else:
            all_train_preds_np_norm = np.concatenate(all_train_preds_norm) if all_train_preds_norm else np.array([])
            all_train_targets_np_norm = np.concatenate(all_train_targets_norm) if all_train_targets_norm else np.array([])


        if self.task_type == "multiclass_classification":


            from sklearn.metrics import accuracy_score, log_loss

            logits_test = np.asarray(all_test_preds_np_norm, dtype=float)
            y_test = np.asarray(all_test_targets_np_norm, dtype=int)
            logits_train = np.asarray(all_train_preds_np_norm, dtype=float)
            y_train = np.asarray(all_train_targets_np_norm, dtype=int)
            logits_val, y_val, ranking_val = _collect_logits_and_targets(
                model=self.model,
                loader=val_loader,
                device=self.device,
                scaler_enabled=self.scaler.is_enabled(),
                task_type=self.task_type,
                hierarchical_multiclass=self.hierarchical_multiclass,
                hierarchical_grade_multitask=self.hierarchical_grade_multitask,
            )
            logits_val = np.asarray(logits_val, dtype=float)
            y_val = np.asarray(y_val, dtype=int).reshape(-1)
            ranking_val = np.asarray(ranking_val, dtype=float).reshape(-1) if ranking_val is not None else None

            def _softmax(z: np.ndarray) -> np.ndarray:
                z = np.asarray(z, dtype=np.float64)
                z = z - np.max(z, axis=1, keepdims=True)
                ez = np.exp(np.clip(z, -60.0, 60.0))
                return ez / np.sum(ez, axis=1, keepdims=True)

            prob_test = _softmax(logits_test) if logits_test.size else np.zeros((0, max(self.num_classes, 1)), dtype=float)
            prob_train = _softmax(logits_train) if logits_train.size else np.zeros((0, max(self.num_classes, 1)), dtype=float)
            prob_val = _softmax(logits_val) if logits_val.size else np.zeros((0, max(self.num_classes, 1)), dtype=float)
            ranking_test_score = None
            ranking_train_score = None
            ranking_val_score = None
            if all_test_ranking_logits:
                _r = np.concatenate(all_test_ranking_logits, axis=0).reshape(-1)
                ranking_test_score = 1.0 / (1.0 + np.exp(-np.clip(_r, -60.0, 60.0)))
            if all_train_ranking_logits:
                _r = np.concatenate(all_train_ranking_logits, axis=0).reshape(-1)
                ranking_train_score = 1.0 / (1.0 + np.exp(-np.clip(_r, -60.0, 60.0)))
            if ranking_val is not None and ranking_val.size:
                ranking_val_score = 1.0 / (1.0 + np.exp(-np.clip(ranking_val, -60.0, 60.0)))
            yhat_test = np.argmax(prob_test, axis=1) if prob_test.size else np.array([], dtype=int)
            yhat_train = np.argmax(prob_train, axis=1) if prob_train.size else np.array([], dtype=int)
            yhat_val = np.argmax(prob_val, axis=1) if prob_val.size else np.array([], dtype=int)
            high_grade_classes = infer_high_grade_classes(
                self.classification_class_names,
                int(prob_test.shape[1]) if prob_test.ndim == 2 and prob_test.size else int(self.num_classes or 0),
            )
            explain_target_class = int(self.model_config.get("explain_target_class_index", 2 if 2 in high_grade_classes else 1) or 1)
            if prob_test.ndim == 2 and prob_test.size and not (0 <= explain_target_class < prob_test.shape[1]):
                explain_target_class = 1 if prob_test.shape[1] > 1 else 0

            def _recommend_threshold_from_val(
                y_true: np.ndarray,
                prob: np.ndarray,
                *,
                target_class: int,
                beta: float = 1.0,
            ) -> float | None:
                """
                Pick a 1-vs-rest threshold for a target class from validation probabilities.

                This addresses a frequent failure mode where the rare class never wins argmax even when
                p(target_class) is informative. A simple override rule ("if p_k >= t then predict k")
                can significantly improve recall/F1 for the rare class.
                """
                try:
                    from sklearn.metrics import precision_recall_curve
                except Exception:
                    return None

                y_true = np.asarray(y_true, dtype=int).reshape(-1)
                prob = np.asarray(prob, dtype=float)
                if prob.ndim != 2 or prob.shape[0] != y_true.shape[0] or prob.size == 0:
                    return None
                k = int(target_class)
                if not (0 <= k < prob.shape[1]):
                    return None

                yk = (y_true == k).astype(int)
                if int(np.sum(yk)) == 0:
                    return None
                pk = prob[:, k]
                prec, rec, thr = precision_recall_curve(yk, pk)
                thr_aligned = np.concatenate([thr, [1.0]])
                b2 = float(beta) ** 2
                eps = 1e-12
                f = (1 + b2) * (prec * rec) / np.clip((b2 * prec + rec), eps, None)
                if f.size == 0:
                    return None
                j = int(np.nanargmax(f))
                t = float(thr_aligned[j]) if thr_aligned.size else None
                if t is None or not np.isfinite(t):
                    return None
                return float(np.clip(t, 0.0, 1.0))

            def _apply_target_threshold_override(
                prob: np.ndarray,
                *,
                base_pred: np.ndarray,
                target_class: int,
                threshold: float,
            ) -> np.ndarray:
                prob = np.asarray(prob, dtype=float)
                base_pred = np.asarray(base_pred, dtype=int).reshape(-1)
                if prob.ndim != 2 or prob.shape[0] != base_pred.shape[0] or prob.size == 0:
                    return base_pred
                k = int(target_class)
                if not (0 <= k < prob.shape[1]):
                    return base_pred
                t = float(threshold)
                if not np.isfinite(t):
                    return base_pred

                other_idx = [i for i in range(prob.shape[1]) if i != k]
                if not other_idx:
                    return base_pred
                alt = np.asarray(other_idx, dtype=int)[np.argmax(prob[:, other_idx], axis=1)]
                out = np.where(prob[:, k] >= t, k, alt)
                return out.astype(int)

            threshold_override = _recommend_threshold_from_val(y_val, prob_val, target_class=explain_target_class, beta=1.0)
            yhat_test_thr = (
                _apply_target_threshold_override(
                    prob_test,
                    base_pred=yhat_test,
                    target_class=explain_target_class,
                    threshold=threshold_override,
                )
                if threshold_override is not None and prob_test.size
                else None
            )

            metrics = {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "log_loss": float("nan"),
            }
            if y_test.size > 0 and prob_test.size > 0 and prob_test.shape[1] >= 2:
                try:
                    label_set = list(range(int(prob_test.shape[1])))

                    def _balanced_accuracy_all_classes(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
                        y_true = np.asarray(y_true, dtype=int).reshape(-1)
                        y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
                        if y_true.size == 0 or n_classes <= 0:
                            return float("nan")
                        recalls = []
                        for k in range(int(n_classes)):
                            m = y_true == k
                            if not np.any(m):
                                recalls.append(0.0)
                            else:
                                recalls.append(float(np.mean((y_pred[m] == k).astype(float))))
                        return float(np.mean(recalls)) if recalls else float("nan")

                    metrics["accuracy"] = float(accuracy_score(y_test, yhat_test))


                    metrics["balanced_accuracy"] = _balanced_accuracy_all_classes(y_test, yhat_test, int(prob_test.shape[1]))
                    metrics["macro_f1"] = float(f1_score(y_test, yhat_test, labels=label_set, average="macro", zero_division=0))
                    metrics["log_loss"] = float(log_loss(y_test, prob_test, labels=label_set))
                except Exception:
                    pass
                try:
                    hg_metrics = compute_high_grade_metrics(
                        y_true=y_test,
                        prob=prob_test,
                        class_names=self.classification_class_names,
                        high_grade_classes=high_grade_classes,
                    )
                    if hg_metrics:
                        metrics["high_grade_recall"] = float(hg_metrics.get("recall", float("nan")))
                        metrics["high_grade_precision"] = float(hg_metrics.get("precision", float("nan")))
                        metrics["high_grade_f1"] = float(hg_metrics.get("f1", float("nan")))
                        metrics["high_grade_ap"] = float(hg_metrics.get("average_precision", float("nan")))
                except Exception:
                    pass
                if ranking_test_score is not None and ranking_test_score.size == y_test.size:
                    try:
                        y_rank_test = (y_test == int(self.high_grade_ranking_target_class_index)).astype(int)
                        if int(np.unique(y_rank_test).size) >= 2:
                            metrics["high_grade_ranking_ap"] = float(
                                average_precision_score(y_rank_test, ranking_test_score)
                            )
                        metrics["high_grade_ranking_top10_capture"] = _top_fraction_capture_from_binary(
                            y_rank_test,
                            ranking_test_score,
                            0.10,
                        )
                        metrics["high_grade_ranking_top20_capture"] = _top_fraction_capture_from_binary(
                            y_rank_test,
                            ranking_test_score,
                            0.20,
                        )
                    except Exception:
                        pass

            self.logger.info(
                "Trainer: Test metrics (multiclass): "
                f"Acc={metrics['accuracy']:.4f}, BalAcc={metrics['balanced_accuracy']:.4f}, "
                f"MacroF1={metrics['macro_f1']:.4f}, LogLoss={metrics['log_loss']:.4f}"
            )


            metrics_in_domain = None
            in_domain_n = None
            metrics_in_domain_error = None
            try:
                age_domain_ka = float(self.model_config.get("training_age_domain_ka", 2580.0) or 2580.0)
                test_subset = getattr(test_loader, "dataset", None)
                test_indices = np.asarray(getattr(test_subset, "indices", []), dtype=int)
                if not test_indices.size:
                    try:
                        cfg_idx = self.model_config.get("_split_test_indices", None)
                        if cfg_idx:
                            test_indices = np.asarray(cfg_idx, dtype=int)
                    except Exception:
                        pass
                if test_indices.size and hasattr(full_dataset_for_denorm, "rey_data"):
                    rey_df = getattr(full_dataset_for_denorm, "rey_data", None)
                    if rey_df is not None and "age_ka_proxy" in rey_df.columns:
                        ages = pd.to_numeric(rey_df.iloc[test_indices]["age_ka_proxy"], errors="coerce").to_numpy(dtype=float)
                        mask = np.isfinite(ages) & (ages <= age_domain_ka)

                        y_base = y_test
                        yhat_base = yhat_test
                        prob_base = prob_test
                        mask_base = mask
                        if y_base.size and mask_base.size != y_base.size:
                            min_n = int(min(mask_base.size, y_base.size))
                            if min_n > 0:
                                mask_base = mask_base[:min_n]
                                y_base = y_base[:min_n]
                                yhat_base = yhat_base[:min_n]
                                prob_base = prob_base[:min_n] if prob_base.size else prob_base

                        in_domain_n = int(np.sum(mask_base)) if mask_base.size else 0
                        if in_domain_n > 0:
                            y_dom = y_base[mask_base]
                            yhat_dom = yhat_base[mask_base]
                            prob_dom = prob_base[mask_base] if prob_base.size else np.empty((0, 0), dtype=float)
                            metrics_in_domain = {
                                "accuracy": float("nan"),
                                "balanced_accuracy": float("nan"),
                                "macro_f1": float("nan"),
                                "log_loss": float("nan"),
                            }
                            if y_dom.size > 0 and prob_dom.size > 0 and prob_dom.shape[1] >= 2:
                                try:
                                    label_set_dom = list(range(int(prob_dom.shape[1])))
                                    metrics_in_domain["accuracy"] = float(accuracy_score(y_dom, yhat_dom))
                                    metrics_in_domain["balanced_accuracy"] = _balanced_accuracy_all_classes(y_dom, yhat_dom, int(prob_dom.shape[1]))
                                    metrics_in_domain["macro_f1"] = float(
                                        f1_score(y_dom, yhat_dom, labels=label_set_dom, average="macro", zero_division=0)
                                    )
                                    metrics_in_domain["log_loss"] = float(log_loss(y_dom, prob_dom, labels=label_set_dom))
                                except Exception:
                                    pass
                            if isinstance(metrics_in_domain, dict):
                                try:
                                    self.logger.info(
                                        "Trainer: Test metrics (in-domain): "
                                        f"n={in_domain_n}, Acc={metrics_in_domain['accuracy']:.4f}, "
                                        f"BalAcc={metrics_in_domain['balanced_accuracy']:.4f}, "
                                        f"MacroF1={metrics_in_domain['macro_f1']:.4f}, "
                                        f"LogLoss={metrics_in_domain['log_loss']:.4f}"
                                    )
                                except Exception:
                                    pass
                    else:
                        metrics_in_domain_error = "age_ka_proxy_missing"
                else:
                    metrics_in_domain_error = "test_indices_unavailable"
            except Exception as e_in_domain:
                try:
                    self.logger.warning(
                        f"Trainer: Failed to compute in-domain test metrics: {e_in_domain}",
                        exc_info=True,
                    )
                except Exception:
                    pass
                metrics_in_domain = None
                in_domain_n = None
                metrics_in_domain_error = str(e_in_domain)

            final_model_path = os.path.join(self.output_dirs["models"], "final_trained_model.pth")
            torch.save(self.model.state_dict(), final_model_path)
            self.logger.info(f"Trainer: Model state saved to {final_model_path}")

            diag_plot_dir = self.output_dirs.get("classification") or self.output_dirs.get("diagnostics", None)
            tables_dir = self.output_dirs.get("tables") or os.path.join(self.output_dirs.get("reports", "."), "tables")

            plot_paths: dict[str, object] = {}
            try:
                if diag_plot_dir:
                    os.makedirs(diag_plot_dir, exist_ok=True)
                    plot_paths.update(
                        plot_confusion_matrices(
                            y_true=y_test,
                            y_pred=yhat_test,
                            class_names=self.classification_class_names,
                            output_dir=diag_plot_dir,
                        )
                        or {}
                    )
                    plot_paths.update(plot_confidence_histograms(prob=prob_test, output_dir=diag_plot_dir) or {})
                    plot_paths.update(plot_top1_reliability(y_true=y_test, prob=prob_test, output_dir=diag_plot_dir) or {})

                    plot_paths.update(
                        plot_one_vs_rest_pr_curves(
                            y_true=y_test,
                            prob=prob_test,
                            class_names=self.classification_class_names,
                            output_dir=diag_plot_dir,
                            target_class=explain_target_class,
                        )
                        or {}
                    )
            except Exception as e_plot:
                self.logger.warning(f"Trainer: Failed to generate multiclass diagnostic plots: {e_plot}")

            table_paths: dict[str, str] = {}
            try:
                os.makedirs(tables_dir, exist_ok=True)
                rey_df_for_diag = getattr(full_dataset_for_denorm, "rey_data", None)

                def _rey_for_loader(loader, n_expected: int) -> np.ndarray | None:
                    if rey_df_for_diag is None or "rey" not in rey_df_for_diag.columns:
                        return None
                    idx = _loader_indices(loader)
                    if idx.size != int(n_expected):
                        return None
                    try:
                        return pd.to_numeric(rey_df_for_diag.iloc[idx]["rey"], errors="coerce").to_numpy(dtype=float)
                    except Exception:
                        return None

                def _meta_for_loader(loader, n_expected: int) -> pd.DataFrame | None:
                    if rey_df_for_diag is None:
                        return None
                    idx = _loader_indices(loader)
                    if idx.size != int(n_expected):
                        return None
                    try:
                        meta = pd.DataFrame({"dataset_index": idx.astype(int)})
                        src_index = np.asarray(rey_df_for_diag.iloc[idx].index, dtype=int)
                        meta["source_index"] = src_index.astype(int)
                        for col in [
                            "core_id",
                            "site_group",
                            "mech_class_id",
                            "grade_geochem_class_id",
                            "class1_grade_subclass_id",
                            "class2_grade_subclass_id",
                            "cascade_global_index",
                            "cascade_source_index",
                        ]:
                            if col in rey_df_for_diag.columns:
                                meta[col] = rey_df_for_diag.iloc[idx][col].to_numpy()
                        return meta
                    except Exception:
                        return None

                table_paths.update(
                    _prefix_paths(save_classification_tables(
                        y_true=y_test,
                        y_pred=yhat_test,
                        prob=prob_test,
                        class_names=self.classification_class_names,
                        output_dir=os.path.join(tables_dir, "test"),
                    ) or {}, "test")
                )
                table_paths.update(
                    _prefix_paths(save_grade_class_diagnostics(
                        y_true=y_test,
                        y_pred=yhat_test,
                        prob=prob_test,
                        class_names=self.classification_class_names,
                        output_dir=os.path.join(tables_dir, "test"),
                        rey_values=_rey_for_loader(test_loader, len(y_test)),
                    ) or {}, "test")
                )
                table_paths.update(
                    _prefix_paths(save_hierarchical_classification_diagnostics(
                        y_true=y_test,
                        y_pred=yhat_test,
                        prob=prob_test,
                        class_names=self.classification_class_names,
                        output_dir=os.path.join(tables_dir, "test", "hierarchical"),
                    ) or {}, "test")
                )
                if yhat_test_thr is not None and yhat_test_thr.size == y_test.size:
                    table_paths.update(
                        _prefix_paths(save_classification_tables(
                            y_true=y_test,
                            y_pred=yhat_test_thr,
                            prob=prob_test,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "test_thresholded"),
                        ) or {}, "test_thresholded")
                    )
                    table_paths.update(
                        _prefix_paths(save_grade_class_diagnostics(
                            y_true=y_test,
                            y_pred=yhat_test_thr,
                            prob=prob_test,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "test_thresholded"),
                            rey_values=_rey_for_loader(test_loader, len(y_test)),
                            pred_prefix="thresholded_pred",
                        ) or {}, "test_thresholded")
                    )
                    table_paths.update(
                        _prefix_paths(save_hierarchical_classification_diagnostics(
                            y_true=y_test,
                            y_pred=yhat_test_thr,
                            prob=prob_test,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "test_thresholded", "hierarchical"),
                        ) or {}, "test_thresholded")
                    )

                try:
                    if prob_test.size > 0 and prob_test.ndim == 2 and y_test.size == prob_test.shape[0]:
                        n_cls = int(prob_test.shape[1])
                        df = pd.DataFrame(
                            {
                                "y_true": y_test.astype(int),
                                "y_pred": yhat_test.astype(int),
                                **{f"p_class{c}": prob_test[:, c].astype(float) for c in range(n_cls)},
                            }
                        )
                        meta_df = _meta_for_loader(test_loader, len(y_test))
                        if meta_df is not None and len(meta_df) == len(df):
                            df = pd.concat([meta_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
                        pmax = np.max(prob_test, axis=1)
                        entropy = -np.sum(np.clip(prob_test, 1e-12, 1.0) * np.log(np.clip(prob_test, 1e-12, 1.0)), axis=1)
                        df["pmax"] = pmax.astype(float)
                        df["entropy"] = entropy.astype(float)
                        if ranking_test_score is not None and ranking_test_score.size == len(df):
                            df["high_grade_ranking_score"] = ranking_test_score.astype(float)
                            df["high_grade_prospectivity_score"] = ranking_test_score.astype(float)
                            df["high_grade_score_formula"] = "learned_high_grade_ranking_head"
                        if yhat_test_thr is not None and yhat_test_thr.size == y_test.size:
                            df["y_pred_thresholded"] = yhat_test_thr.astype(int)
                            df[f"target_class{explain_target_class}_threshold"] = float(threshold_override)
                        pred_csv = os.path.join(tables_dir, "test", "predictions_with_probs.csv")
                        df.to_csv(pred_csv, index=False)
                        table_paths["test_predictions_with_probs_csv"] = pred_csv
                except Exception as e_predcsv:
                    self.logger.warning(f"Trainer: Failed to write per-sample probability CSV (test): {e_predcsv}")

                if y_val.size > 0 and yhat_val.size == y_val.size and prob_val.size > 0:
                    table_paths.update(_prefix_paths(save_classification_tables(
                        y_true=y_val,
                        y_pred=yhat_val,
                        prob=prob_val,
                        class_names=self.classification_class_names,
                        output_dir=os.path.join(tables_dir, "val"),
                    ) or {}, "val"))
                    table_paths.update(
                        _prefix_paths(save_grade_class_diagnostics(
                            y_true=y_val,
                            y_pred=yhat_val,
                            prob=prob_val,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "val"),
                            rey_values=_rey_for_loader(val_loader, len(y_val)),
                        ) or {}, "val")
                    )
                    table_paths.update(
                        _prefix_paths(save_hierarchical_classification_diagnostics(
                            y_true=y_val,
                            y_pred=yhat_val,
                            prob=prob_val,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "val", "hierarchical"),
                        ) or {}, "val")
                    )
                    try:
                        if prob_val.size > 0 and prob_val.ndim == 2 and y_val.size == prob_val.shape[0]:
                            n_cls = int(prob_val.shape[1])
                            dfv = pd.DataFrame(
                                {
                                    "y_true": y_val.astype(int),
                                    "y_pred": yhat_val.astype(int),
                                    **{f"p_class{c}": prob_val[:, c].astype(float) for c in range(n_cls)},
                                }
                            )
                            meta_df = _meta_for_loader(val_loader, len(y_val))
                            if meta_df is not None and len(meta_df) == len(dfv):
                                dfv = pd.concat([meta_df.reset_index(drop=True), dfv.reset_index(drop=True)], axis=1)
                            pmax = np.max(prob_val, axis=1)
                            entropy = -np.sum(np.clip(prob_val, 1e-12, 1.0) * np.log(np.clip(prob_val, 1e-12, 1.0)), axis=1)
                            dfv["pmax"] = pmax.astype(float)
                            dfv["entropy"] = entropy.astype(float)
                            if ranking_val_score is not None and ranking_val_score.size == len(dfv):
                                dfv["high_grade_ranking_score"] = ranking_val_score.astype(float)
                                dfv["high_grade_prospectivity_score"] = ranking_val_score.astype(float)
                                dfv["high_grade_score_formula"] = "learned_high_grade_ranking_head"
                            pred_csv = os.path.join(tables_dir, "val", "predictions_with_probs.csv")
                            os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
                            dfv.to_csv(pred_csv, index=False)
                            table_paths["val_predictions_with_probs_csv"] = pred_csv
                    except Exception as e_predcsv_val:
                        self.logger.warning(f"Trainer: Failed to write per-sample probability CSV (val): {e_predcsv_val}")

                if y_train.size > 0 and yhat_train.size == y_train.size:
                    table_paths.update(_prefix_paths(save_classification_tables(
                        y_true=y_train,
                        y_pred=yhat_train,
                        prob=prob_train,
                        class_names=self.classification_class_names,
                        output_dir=os.path.join(tables_dir, "train"),
                    ) or {}, "train"))
                    table_paths.update(
                        _prefix_paths(save_grade_class_diagnostics(
                            y_true=y_train,
                            y_pred=yhat_train,
                            prob=prob_train,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "train"),
                            rey_values=_rey_for_loader(train_eval_loader, len(y_train)),
                        ) or {}, "train")
                    )
                    table_paths.update(
                        _prefix_paths(save_hierarchical_classification_diagnostics(
                            y_true=y_train,
                            y_pred=yhat_train,
                            prob=prob_train,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "train", "hierarchical"),
                        ) or {}, "train")
                    )
                    try:
                        if prob_train.size > 0 and prob_train.ndim == 2 and y_train.size == prob_train.shape[0]:
                            n_cls = int(prob_train.shape[1])
                            dft = pd.DataFrame(
                                {
                                    "y_true": y_train.astype(int),
                                    "y_pred": yhat_train.astype(int),
                                    **{f"p_class{c}": prob_train[:, c].astype(float) for c in range(n_cls)},
                                }
                            )
                            meta_df = _meta_for_loader(train_eval_loader, len(y_train))
                            if meta_df is not None and len(meta_df) == len(dft):
                                dft = pd.concat([meta_df.reset_index(drop=True), dft.reset_index(drop=True)], axis=1)
                            pmax = np.max(prob_train, axis=1)
                            entropy = -np.sum(np.clip(prob_train, 1e-12, 1.0) * np.log(np.clip(prob_train, 1e-12, 1.0)), axis=1)
                            dft["pmax"] = pmax.astype(float)
                            dft["entropy"] = entropy.astype(float)
                            if ranking_train_score is not None and ranking_train_score.size == len(dft):
                                dft["high_grade_ranking_score"] = ranking_train_score.astype(float)
                                dft["high_grade_prospectivity_score"] = ranking_train_score.astype(float)
                                dft["high_grade_score_formula"] = "learned_high_grade_ranking_head"
                            pred_csv = os.path.join(tables_dir, "train", "predictions_with_probs.csv")
                            os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
                            dft.to_csv(pred_csv, index=False)
                            table_paths["train_predictions_with_probs_csv"] = pred_csv
                    except Exception as e_predcsv_train:
                        self.logger.warning(f"Trainer: Failed to write per-sample probability CSV (train): {e_predcsv_train}")
            except Exception as e_tab:
                self.logger.warning(f"Trainer: Failed to save classification tables: {e_tab}")

            train_metrics = {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "log_loss": float("nan"),
            }
            if y_train.size > 0 and prob_train.size > 0 and prob_train.shape[1] >= 2:
                try:
                    train_metrics["accuracy"] = float(accuracy_score(y_train, yhat_train))
                    train_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_train, yhat_train))
                    train_metrics["macro_f1"] = float(f1_score(y_train, yhat_train, average="macro"))
                    train_metrics["log_loss"] = float(log_loss(y_train, prob_train, labels=list(range(prob_train.shape[1]))))
                except Exception:
                    pass

            val_metrics = {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "log_loss": float("nan"),
            }
            val_target_class_ap = None
            if y_val.size > 0 and prob_val.size > 0 and prob_val.ndim == 2 and prob_val.shape[1] >= 2:
                try:
                    val_metrics["accuracy"] = float(accuracy_score(y_val, yhat_val))
                    val_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_val, yhat_val))
                    val_metrics["macro_f1"] = float(f1_score(y_val, yhat_val, average="macro"))
                    val_metrics["log_loss"] = float(log_loss(y_val, prob_val, labels=list(range(prob_val.shape[1]))))
                except Exception:
                    pass
                try:
                    yk = (y_val.astype(int) == int(explain_target_class)).astype(int)
                    if int(np.unique(yk).size) >= 2:
                        val_target_class_ap = float(
                            average_precision_score(
                                yk,
                                np.asarray(prob_val[:, int(explain_target_class)], dtype=float),
                            )
                        )
                except Exception:
                    val_target_class_ap = None
                try:
                    hg_val = compute_high_grade_metrics(
                        y_true=y_val,
                        prob=prob_val,
                        class_names=self.classification_class_names,
                        high_grade_classes=high_grade_classes,
                    )
                    if hg_val:
                        val_metrics["high_grade_recall"] = float(hg_val.get("recall", float("nan")))
                        val_metrics["high_grade_precision"] = float(hg_val.get("precision", float("nan")))
                        val_metrics["high_grade_f1"] = float(hg_val.get("f1", float("nan")))
                        val_metrics["high_grade_ap"] = float(hg_val.get("average_precision", float("nan")))
                except Exception:
                    pass
                if ranking_val_score is not None and ranking_val_score.size == y_val.size:
                    try:
                        y_rank_val = (y_val == int(self.high_grade_ranking_target_class_index)).astype(int)
                        if int(np.unique(y_rank_val).size) >= 2:
                            val_metrics["high_grade_ranking_ap"] = float(
                                average_precision_score(y_rank_val, ranking_val_score)
                            )
                        val_metrics["high_grade_ranking_top10_capture"] = _top_fraction_capture_from_binary(
                            y_rank_val,
                            ranking_val_score,
                            0.10,
                        )
                        val_metrics["high_grade_ranking_top20_capture"] = _top_fraction_capture_from_binary(
                            y_rank_val,
                            ranking_val_score,
                            0.20,
                        )
                    except Exception:
                        pass


            target_class_bootstrap = None
            try:
                if (
                    prob_test.size > 0
                    and prob_test.ndim == 2
                    and y_test.size == prob_test.shape[0]
                    and prob_test.shape[1] > 1
                ):
                    boot_cfg = dict(self.model_config.get("target_class_bootstrap", self.model_config.get("class1_bootstrap", {})) or {})
                    n_boot = int(boot_cfg.get("n_boot", 2000) or 2000)
                    seed = int(boot_cfg.get("seed", 42) or 42)
                    thr_boot = float(
                        boot_cfg.get(
                            "threshold",
                            threshold_override if threshold_override is not None else 0.5,
                        )
                        or 0.5
                    )
                    min_groups = int(boot_cfg.get("min_groups", 3) or 3)
                    if n_boot <= 0:
                        target_class_bootstrap = {"status": "disabled"}
                        raise StopIteration


                    groups_test = None
                    try:

                        subset = getattr(test_loader, "dataset", None)
                        idx = np.asarray(getattr(subset, "indices", []), dtype=int)
                        if idx.size and hasattr(full_dataset_for_denorm, "rey_data"):
                            rey_df = getattr(full_dataset_for_denorm, "rey_data", None)
                            if rey_df is not None:
                                gcol = (
                                    "site_group"
                                    if "site_group" in rey_df.columns
                                    else ("core_id" if "core_id" in rey_df.columns else ("site_hole" if "site_hole" in rey_df.columns else None))
                                )
                                if gcol is not None:
                                    groups_test = rey_df.iloc[idx][gcol].astype(str).to_numpy()
                    except Exception:
                        groups_test = None

                    if groups_test is not None and int(np.unique(groups_test).size) >= min_groups:
                        rng = np.random.default_rng(seed)
                        cores = np.unique(groups_test)
                        idx_by = {c: np.where(groups_test == c)[0] for c in cores}
                        target_for_boot = int(explain_target_class)
                        yk = (y_test.astype(int) == target_for_boot).astype(int)
                        pk = np.asarray(prob_test[:, target_for_boot], dtype=float)

                        def _prf_at(ii: np.ndarray) -> tuple[float, float, float, float]:
                            y0 = yk[ii]
                            p0 = pk[ii]
                            if int(np.sum(y0)) == 0:
                                return float("nan"), 0.0, 0.0, 0.0
                            try:
                                ap = float(average_precision_score(y0, p0))
                            except Exception:
                                ap = float("nan")
                            yhat = (p0 >= float(thr_boot)).astype(int)
                            try:
                                from sklearn.metrics import precision_recall_fscore_support

                                pr, re, f1, _ = precision_recall_fscore_support(
                                    y0,
                                    yhat,
                                    average="binary",
                                    zero_division=0,
                                )
                                return float(ap), float(pr), float(re), float(f1)
                            except Exception:
                                return float(ap), float("nan"), float("nan"), float("nan")

                        rows = []
                        for _ in range(max(100, n_boot)):
                            samp = rng.choice(cores, size=int(len(cores)), replace=True)
                            ii = np.concatenate([idx_by[c] for c in samp])
                            rows.append(_prf_at(ii))
                        arr = np.asarray(rows, dtype=float)

                        def _ci(v: np.ndarray) -> dict:
                            v = np.asarray(v, dtype=float)
                            v = v[np.isfinite(v)]
                            if v.size == 0:
                                return {"p2_5": None, "p50": None, "p97_5": None}
                            return {
                                "p2_5": float(np.percentile(v, 2.5)),
                                "p50": float(np.percentile(v, 50)),
                                "p97_5": float(np.percentile(v, 97.5)),
                            }

                        target_class_bootstrap = {
                            "method": "cluster_bootstrap",
                            "group_col": str(gcol or "group"),
                            "target_class": int(target_for_boot),
                            "n_groups": int(len(cores)),
                            "n_samples": int(len(y_test)),
                            "n_pos": int(np.sum(yk)),
                            "threshold": float(thr_boot),
                            "n_boot": int(arr.shape[0]),
                            "ap": _ci(arr[:, 0]),
                            "precision": _ci(arr[:, 1]),
                            "recall": _ci(arr[:, 2]),
                            "f1": _ci(arr[:, 3]),
                        }
            except StopIteration:
                pass
            except Exception as e_boot:
                target_class_bootstrap = {"error": str(e_boot)}


            artifacts = {
                "status": "success" if not training_interrupted else "interrupted_trainer",
                "model_path": final_model_path,
                "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
                "best_val_metric": float(best_val_metric) if best_val_metric != -float('inf') else None,
                "best_epoch_val_loss": float(best_epoch_val_loss) if best_epoch_val_loss is not None else None,
                "early_stopping_metric": str(self.early_stopping_metric),
                "best_epoch": best_epoch if best_epoch != -1 else None,
                "train_losses": [float(l) for l in train_losses_history],
                "val_losses": [float(l) for l in val_losses_history],
                "test_metrics": {
                    **{k: float(v) for k, v in metrics.items()},
                    "num_classes": int(prob_test.shape[1]) if prob_test.size else int(self.num_classes or 0),
                    "test_true": y_test.astype(int).tolist(),
                    "test_pred": yhat_test.astype(int).tolist(),
                },
                "test_metrics_in_domain": (
                    {k: float(v) for k, v in (metrics_in_domain or {}).items()}
                    if isinstance(metrics_in_domain, dict)
                    else None
                ),
                "test_in_domain_n": int(in_domain_n) if in_domain_n is not None else None,
                "test_metrics_in_domain_error": (
                    str(metrics_in_domain_error) if metrics_in_domain_error else None
                ),
                "threshold_override": {
                    "target_class": int(explain_target_class),
                    "threshold": float(threshold_override) if threshold_override is not None else None,
                    "test_pred_thresholded": yhat_test_thr.astype(int).tolist() if yhat_test_thr is not None else None,
                },
                "high_grade_classes": [int(x) for x in high_grade_classes],
                "target_class_bootstrap": target_class_bootstrap,
                "class1_bootstrap": target_class_bootstrap,
                "train_metrics": {k: float(v) for k, v in train_metrics.items()},
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "val_target_class_ap": float(val_target_class_ap) if val_target_class_ap is not None else None,
                "val_class1_ap": float(val_target_class_ap) if val_target_class_ap is not None else None,
                "plot_paths": {
                    "diagnostics_dir": diag_plot_dir,
                    "classification_diagnostics_dir": diag_plot_dir,
                    "classification_plots": plot_paths,
                },
                "table_paths": table_paths,
                "training_history": self.training_history,
            }

            try:
                if y_test.size > 0 and prob_test.size > 0 and prob_test.shape[1] > 1:
                    table_paths.update(
                        save_target_class_threshold_sweep(
                            y_true=y_test,
                            prob=prob_test,
                            class_names=self.classification_class_names,
                            output_dir=os.path.join(tables_dir, "test"),
                            target_class=explain_target_class,
                        )
                        or {}
                    )
                    for high_class in high_grade_classes:
                        if int(high_class) == int(explain_target_class):
                            continue
                        table_paths.update(
                            save_target_class_threshold_sweep(
                                y_true=y_test,
                                prob=prob_test,
                                class_names=self.classification_class_names,
                                output_dir=os.path.join(tables_dir, "test"),
                                target_class=int(high_class),
                            )
                            or {}
                        )
                if y_val.size > 0 and prob_val.size > 0 and prob_val.shape[1] > 1:
                    save_target_class_threshold_sweep(
                        y_true=y_val,
                        prob=prob_val,
                        class_names=self.classification_class_names,
                        output_dir=os.path.join(tables_dir, "val"),
                        target_class=explain_target_class,
                    )
            except Exception as e_thr:
                self.logger.warning(f"Trainer: Failed to write target-class threshold sweep: {e_thr}")
            if self.progress_callback:
                self.progress_callback(1.0, "Trainer finished.")
            return True, artifacts


        if self.task_type == "binary_classification":
            def _sigmoid_np(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -60.0, 60.0)
                return 1.0 / (1.0 + np.exp(-x))

            y_test_true = np.asarray(all_test_targets_np_norm, dtype=float) if all_test_targets_np_norm.size else np.array([], dtype=float)
            y_test_prob = _sigmoid_np(all_test_preds_np_norm) if all_test_preds_np_norm.size else np.array([], dtype=float)
            y_train_true = np.asarray(all_train_targets_np_norm, dtype=float) if all_train_targets_np_norm.size else np.array([], dtype=float)
            y_train_prob = _sigmoid_np(all_train_preds_np_norm) if all_train_preds_np_norm.size else np.array([], dtype=float)

            metrics = {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan"), "accuracy": float("nan"), "f1": float("nan")}
            if y_test_true.size >= 10 and len(np.unique(y_test_true)) >= 2:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_test_true.astype(int), y_test_prob))
                    metrics["pr_auc"] = float(average_precision_score(y_test_true.astype(int), y_test_prob))
                except Exception:
                    pass
            if y_test_true.size > 0 and np.isfinite(y_test_prob).all():
                metrics["brier"] = float(np.mean((y_test_prob - y_test_true) ** 2))
                y_hat = (y_test_prob >= 0.5).astype(int)
                metrics["accuracy"] = float(np.mean(y_hat == y_test_true.astype(int)))
                try:
                    metrics["f1"] = float(f1_score(y_test_true.astype(int), y_hat))
                except Exception:
                    pass

            self.logger.info(
                "Trainer: Test metrics (binary): "
                f"ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}, "
                f"Brier={metrics['brier']:.4f}, Acc@0.5={metrics['accuracy']:.4f}, F1@0.5={metrics['f1']:.4f}"
            )


            final_model_path = os.path.join(self.output_dirs["models"], "final_trained_model.pth")
            torch.save(self.model.state_dict(), final_model_path)
            self.logger.info(f"Trainer: Model state saved to {final_model_path}")
            if self.progress_callback:
                self.progress_callback(0.9, "Test evaluation complete.")


            diag_plot_dir = self.output_dirs.get("diagnostics", None)
            try:
                if diag_plot_dir:
                    os.makedirs(diag_plot_dir, exist_ok=True)
                    import matplotlib

                    matplotlib.use("Agg", force=True)
                    import matplotlib.pyplot as plt


                    if y_test_true.size > 0:
                        fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=220)
                        pos = y_test_prob[y_test_true >= 0.5]
                        neg = y_test_prob[y_test_true < 0.5]
                        ax.hist(neg, bins=20, alpha=0.7, label=f"Negative (n={len(neg)})")
                        ax.hist(pos, bins=20, alpha=0.7, label=f"Positive (n={len(pos)})")
                        ax.set_xlabel(f"Predicted P(ΣREY ≥ {self.classification_threshold_ppm:g} ppm)")
                        ax.set_ylabel("Count")
                        ax.set_title("Test set probability distribution")
                        ax.grid(True, alpha=0.25)
                        ax.legend(frameon=True)
                        fig.tight_layout()
                        fig.savefig(os.path.join(diag_plot_dir, "test_prob_hist.png"))
                        plt.close(fig)


                    if y_test_true.size >= 10:
                        bins = np.linspace(0.0, 1.0, 11)
                        xs, ys = [], []
                        for i in range(len(bins) - 1):
                            lo, hi = float(bins[i]), float(bins[i + 1])
                            m = (y_test_prob >= lo) & (y_test_prob < (hi if i < len(bins) - 2 else hi + 1e-9))
                            if int(m.sum()) < 3:
                                continue
                            xs.append(float(np.mean(y_test_prob[m])))
                            ys.append(float(np.mean(y_test_true[m])))
                        if xs:
                            fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=220)
                            ax.plot([0, 1], [0, 1], "k--", lw=1)
                            ax.plot(xs, ys, marker="o", lw=1.6)
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            ax.set_xlabel("Mean predicted probability (bin)")
                            ax.set_ylabel("Observed positive rate (bin)")
                            ax.set_title("Calibration (test set)")
                            ax.grid(True, alpha=0.25)
                            fig.tight_layout()
                            fig.savefig(os.path.join(diag_plot_dir, "test_calibration.png"))
                            plt.close(fig)
            except Exception as e_plot:
                self.logger.warning(f"Trainer: Failed to generate classification plots: {e_plot}")

            artifacts = {
                "status": "success" if not training_interrupted else "interrupted_trainer",
                "model_path": final_model_path,
                "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
                "best_epoch": best_epoch if best_epoch != -1 else None,
                "train_losses": [float(l) for l in train_losses_history],
                "val_losses": [float(l) for l in val_losses_history],
                "test_metrics": {
                    "roc_auc": float(metrics["roc_auc"]),
                    "pr_auc": float(metrics["pr_auc"]),
                    "brier": float(metrics["brier"]),
                    "accuracy": float(metrics["accuracy"]),
                    "f1": float(metrics["f1"]),
                    "threshold_ppm": float(self.classification_threshold_ppm),
                    "test_true": y_test_true.astype(float).tolist(),
                    "test_prob": y_test_prob.astype(float).tolist(),
                },
                "plot_paths": {
                    "diagnostics_dir": diag_plot_dir,
                },
                "training_history": self.training_history,
            }
            if self.progress_callback:
                self.progress_callback(1.0, "Trainer finished.")
            return True, artifacts


        test_preds_denorm, test_targets_denorm, test_depths_denorm, residuals = [], [], [], []
        train_preds_denorm, train_targets_denorm = [], []
        test_mse, test_mae, test_r2 = np.nan, np.nan, np.nan

        if len(all_test_preds_np_norm) > 0:
            try:
                test_preds_denorm = full_dataset_for_denorm._denormalize_rey(all_test_preds_np_norm)
                test_targets_denorm = full_dataset_for_denorm._denormalize_rey(all_test_targets_np_norm)


                test_preds_denorm = np.clip(test_preds_denorm, 0.0, None)


                if len(all_test_depths_np_norm) > 0:
                    if hasattr(full_dataset_for_denorm, '_denormalize_depth'):
                        test_depths_denorm = full_dataset_for_denorm._denormalize_depth(all_test_depths_np_norm)
                    elif hasattr(full_dataset_for_denorm, 'data_config') and \
                         'normalization_params' in full_dataset_for_denorm.data_config and \
                         'depth' in full_dataset_for_denorm.data_config['normalization_params']:

                        depth_params = full_dataset_for_denorm.data_config['normalization_params']['depth']
                        if 'mean' in depth_params and 'std' in depth_params and depth_params['std'] != 0:
                            mean_depth = depth_params['mean']
                            std_depth = depth_params['std']
                            test_depths_denorm = (all_test_depths_np_norm * std_depth) + mean_depth
                        elif 'min' in depth_params and 'max' in depth_params and (depth_params['max'] - depth_params['min']) != 0 :
                            min_depth = depth_params['min']
                            max_depth = depth_params['max']
                            test_depths_denorm = all_test_depths_np_norm * (max_depth - min_depth) + min_depth
                        else:
                            self.logger.warning("Trainer: Depth normalization parameters (mean/std or min/max) incomplete or std is zero. Using normalized depths for plot.")
                            test_depths_denorm = all_test_depths_np_norm
                    else:
                        self.logger.warning("Trainer: Could not retrieve depth normalization parameters or method. Using normalized depths for plot.")
                        test_depths_denorm = all_test_depths_np_norm
                else:
                    test_depths_denorm = np.array([])

                test_mse = mean_squared_error(test_targets_denorm, test_preds_denorm)
                test_mae = mean_absolute_error(test_targets_denorm, test_preds_denorm)
                test_r2 = r2_score(test_targets_denorm, test_preds_denorm)
                residuals = test_targets_denorm - test_preds_denorm
            except Exception as e_denorm:
                self.logger.error(f"Trainer: Error de-normalizing test results: {e_denorm}.")
                test_preds_denorm = all_test_preds_np_norm 
                test_targets_denorm = all_test_targets_np_norm
        else:
            self.logger.warning("Trainer: Test set evaluation skipped as no test data was processed.")


        if len(all_train_preds_np_norm) > 0:
            train_preds_denorm = full_dataset_for_denorm._denormalize_rey(all_train_preds_np_norm)
            train_targets_denorm = full_dataset_for_denorm._denormalize_rey(all_train_targets_np_norm)


            train_preds_denorm = np.clip(train_preds_denorm, 0.0, None)

        metrics_log_msg = f"Trainer: Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}"
        self.logger.info(metrics_log_msg)

        final_model_path = os.path.join(self.output_dirs["models"], "final_trained_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        self.logger.info(f"Trainer: Model state saved to {final_model_path}")

        if self.progress_callback: self.progress_callback(0.9, "Test evaluation complete.")

        plot_params = self.model_config.get("plot_params", {})
        scatter_plot_path_base = None 
        plotly_scatter_path = None 
        diag_plot_dir = None 

        if len(test_preds_denorm) > 0 or len(train_preds_denorm) > 0:

            scatter_plot_path_base = os.path.join(self.output_dirs["scatter"], "train_test_scatter_comparison")
            y_true_list = []
            y_pred_list = []
            labels_list = []

            if len(train_targets_denorm) > 0:
                y_true_list.append(train_targets_denorm)
                y_pred_list.append(train_preds_denorm)
                labels_list.append("Train")
            if len(test_targets_denorm) > 0:
                y_true_list.append(test_targets_denorm)
                y_pred_list.append(test_preds_denorm)
                labels_list.append("Test")

            plot_scatter(
                y_true_list, y_pred_list, "Train vs. Test Set",
                scatter_plot_path_base, data_set_labels=labels_list, plot_params_dict=plot_params
            )

            plotly_scatter_path = os.path.join(self.output_dirs["plotly_interactive"], "train_test_scatter_comparison.html")
            plot_scatter_plotly(
                y_true_list, y_pred_list, "Train vs. Test Set",
                plotly_scatter_path, data_set_labels=labels_list, plot_params_dict=plot_params.get("plotly", {})
            )


            if len(residuals) > 0:
                diag_plot_dir = self.output_dirs["diagnostics"]
                plot_residuals(residuals, test_preds_denorm, "Test Set", diag_plot_dir, plot_params_dict=plot_params)
                plot_error_histogram(residuals, "Test Set", diag_plot_dir, plot_params_dict=plot_params)


                if len(test_depths_denorm) > 0:
                    plot_depth_residuals(test_depths_denorm, residuals, "Test Set", diag_plot_dir, plot_params_dict=plot_params)

        artifacts = {
            "status": "success" if not training_interrupted else "interrupted_trainer",
            "model_path": final_model_path,
            "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
            "best_epoch": best_epoch if best_epoch != -1 else None,
            "train_losses": [float(l) for l in train_losses_history],
            "val_losses": [float(l) for l in val_losses_history],
            "test_metrics": {
                "mse": float(test_mse),
                "mae": float(test_mae),
                "r2": float(test_r2),
                "test_targets_denorm": test_targets_denorm.tolist() if isinstance(test_targets_denorm, np.ndarray) else test_targets_denorm,
                "test_predictions_denorm": test_preds_denorm.tolist() if isinstance(test_preds_denorm, np.ndarray) else test_preds_denorm,
                "true_labels": test_targets_denorm.tolist() if isinstance(test_targets_denorm, np.ndarray) else test_targets_denorm,
                "pred_labels": test_preds_denorm.tolist() if isinstance(test_preds_denorm, np.ndarray) else test_preds_denorm
            },
            "plot_paths": { 
                "scatter_plot_base": scatter_plot_path_base,
                "plotly_scatter_html": plotly_scatter_path,
                "diagnostics_dir": diag_plot_dir
            },
            "training_history": self.training_history
        }
        if self.progress_callback: self.progress_callback(1.0, "Trainer finished.")
        return True, artifacts
