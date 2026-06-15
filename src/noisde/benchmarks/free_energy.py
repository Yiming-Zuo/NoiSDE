"""PyMBAR-backed free-energy utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from noisde.utility.metrics import coefficient_of_variation_squared, effective_sample_size


@dataclass(frozen=True)
class MBARResult:
    delta_f: np.ndarray
    d_delta_f: np.ndarray


@dataclass(frozen=True)
class ReweightingResult:
    delta_f: float
    ess_n: float
    cv2_w: float
    log_weight_995: float
    sample_count: int


class PFReweightingEvaluator:
    """Computes reduced FEP diagnostics from PF companion log weights."""

    def evaluate(self, log_weights: torch.Tensor) -> ReweightingResult:
        if log_weights.ndim != 1:
            raise ValueError("log_weights must be one-dimensional")
        delta_f = -torch.logsumexp(log_weights, dim=0) + torch.log(torch.tensor(log_weights.numel(), device=log_weights.device, dtype=log_weights.dtype))
        quantile = torch.quantile(log_weights.detach(), 0.995)
        return ReweightingResult(
            delta_f=float(delta_f.detach().cpu()),
            ess_n=float(effective_sample_size(log_weights).detach().cpu()),
            cv2_w=float(coefficient_of_variation_squared(log_weights).detach().cpu()),
            log_weight_995=float(quantile.cpu()),
            sample_count=int(log_weights.numel()),
        )


def pf_audit_row(
    result: ReweightingResult,
    audit_id: str,
    window_id: str,
    seed: str,
    variant: str,
    pf_mode: str,
    hutchinson_m: int,
    overlap_min: float | None = None,
    r_high_e: float | None = None,
    delta_f_abs_err_kcal_mol: float | None = None,
    ess_threshold: float = 0.05,
    log_weight_995_max: float = 6.0,
) -> dict[str, object]:
    pf_gate_pass = result.ess_n >= ess_threshold and result.log_weight_995 <= log_weight_995_max
    return {
        "audit_id": audit_id,
        "window_id": window_id,
        "seed": seed,
        "variant": variant,
        "pf_mode": pf_mode,
        "hutchinson_M": hutchinson_m,
        "ess_n": result.ess_n,
        "log_weight_995": result.log_weight_995,
        "overlap_min": "" if overlap_min is None else overlap_min,
        "r_highE": "" if r_high_e is None else r_high_e,
        "deltaF_abs_err_kcal_mol": "" if delta_f_abs_err_kcal_mol is None else delta_f_abs_err_kcal_mol,
        "pf_gate_pass": pf_gate_pass,
        "mbar_bootstrap_gate_pass": "",
        "reference_quality_gate_pass": "",
        "overall_pass_fail": "PASS" if pf_gate_pass else "FAIL",
        "downgrade_flag": not pf_gate_pass,
        "allowed_claim": "density_dependent" if pf_gate_pass else "proposal_quality_only",
    }


class MBARFreeEnergyEstimator:
    """Thin adapter around pymbar.MBAR."""

    def __init__(self) -> None:
        try:
            from pymbar import MBAR, timeseries
        except ImportError as exc:
            raise ImportError("PyMBAR is required for MBAR free-energy estimation. Install pymbar in the project environment.") from exc
        self._MBAR = MBAR
        self.timeseries = timeseries

    def compute(self, u_kn: np.ndarray, n_k: np.ndarray) -> MBARResult:
        mbar = self._MBAR(u_kn, n_k)
        result = mbar.compute_free_energy_differences()
        return MBARResult(delta_f=result["Delta_f"], d_delta_f=result["dDelta_f"])

    def decorrelate(self, values: np.ndarray) -> np.ndarray:
        t0, g, _ = self.timeseries.detect_equilibration(values)
        production = values[t0:]
        indices = self.timeseries.subsample_correlated_data(production, g=g)
        return production[indices]
