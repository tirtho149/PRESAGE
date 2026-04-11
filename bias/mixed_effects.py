"""
bias/mixed_effects.py
=====================
Mixed-effects regression for RDS confounder control (§8, Appendix D).

Eq. (5):
    L_ij = α + β_g · group_ij + γ_1 · complexity_ij
           + γ_2 · quality_ij + u_sector + u_region + ε_ij

Appendix D:
    "Estimated with REML (lme4/statsmodels MixedLM)."
    "Fixed effects: group indicator (man and Americas as reference levels),
     scene complexity (edge density, z-scored), image quality score (z-scored)."
    "Random effects: intercepts for sector and region."
    "Sensitivity analyses remove one confounder at a time."
    "All p-values are two-tailed, Bonferroni-corrected for the number of
     group comparisons."

Paper §8:
    "Significance of β_g after controlling for all covariates is the key test."
    "All causal language is avoided: we report associations, not causal effects."
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------

def build_regression_dataframe(
    traces: List[Dict],
    group_labels: List[str],
    sector_labels: List[str],
    region_labels: List[str],
    complexity_scores: Optional[np.ndarray] = None,
    quality_scores: Optional[np.ndarray] = None,
    reference_gender: str = "male",
    reference_region: str = "Americas",
) -> pd.DataFrame:
    """
    Build the regression DataFrame for Eq. (5).

    Parameters
    ----------
    traces            : list of trace dicts
    group_labels      : demographic group per trace (gender or region)
    sector_labels     : economic sector per trace (random effect)
    region_labels     : world region per trace (random effect)
    complexity_scores : edge density z-scored (N,) — if None, filled with 0
    quality_scores    : image quality z-scored (N,) — if None, filled with 0
    """
    n = len(traces)

    path_lengths = np.array([
        t.get("path_length", len(t.get("path", [])))
        for t in traces
    ], dtype=float)

    if complexity_scores is None:
        complexity_scores = np.zeros(n)
    if quality_scores is None:
        quality_scores = np.zeros(n)

    # Z-score covariates (Appendix D)
    def zscore(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x - x.mean()

    complexity_z = zscore(complexity_scores)
    quality_z = zscore(quality_scores)

    df = pd.DataFrame({
        "path_length": path_lengths,
        "group": group_labels,
        "sector": sector_labels,
        "region": region_labels,
        "complexity": complexity_z,
        "quality": quality_z,
        "image_id": [t.get("image_id", str(i)) for i, t in enumerate(traces)],
    })

    # Set reference levels (Appendix D)
    df["group"] = pd.Categorical(
        df["group"],
        categories=[reference_gender] + [g for g in df["group"].unique() if g != reference_gender]
    )
    return df


# ---------------------------------------------------------------------------
# Mixed-effects regression (Eq. 5, Appendix D)
# ---------------------------------------------------------------------------

def fit_mixed_effects(
    df: pd.DataFrame,
    outcome_col: str = "path_length",
    group_col: str = "group",
    fixed_covariates: Optional[List[str]] = None,
    random_effects_col: str = "sector",  # primary grouping variable for MixedLM
    bonferroni_n: Optional[int] = None,
    reml: bool = True,
) -> Dict[str, Any]:
    """
    Fit Eq. (5) using REML mixed-effects regression (Appendix D).

    Parameters
    ----------
    df                : DataFrame from build_regression_dataframe()
    outcome_col       : dependent variable (path_length)
    group_col         : group indicator column
    fixed_covariates  : list of fixed-effect covariate column names
    random_effects_col: column for random effect grouping
    bonferroni_n      : if set, correct p-values by this factor
    reml              : use REML estimation (Appendix D)

    Returns
    -------
    dict with: summary_df, beta_g_table, significant_groups
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for mixed-effects regression. "
            "Install with: pip install statsmodels"
        )

    covariates = fixed_covariates or ["complexity", "quality"]
    cov_str = " + ".join(covariates) if covariates else "1"

    # Build formula (Eq. 5 — group is the key fixed effect, Appendix D)
    formula = f"{outcome_col} ~ C({group_col}) + {cov_str}"

    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df[random_effects_col],
        )
        result = model.fit(reml=reml, method="lbfgs")

        # Extract β_g coefficients and p-values
        summary = result.summary()
        params = result.params
        pvalues = result.pvalues
        conf_int = result.conf_int()

        beta_rows = []
        for param_name in params.index:
            if f"C({group_col})" in param_name:
                group_name = param_name.replace(f"C({group_col})[T.", "").rstrip("]")
                p_raw = pvalues[param_name]
                p_corrected = min(p_raw * bonferroni_n, 1.0) if bonferroni_n else p_raw
                beta_rows.append({
                    "group": group_name,
                    "beta_g": float(params[param_name]),
                    "p_raw": float(p_raw),
                    "p_corrected": float(p_corrected),
                    "ci_lo": float(conf_int.loc[param_name, 0]),
                    "ci_hi": float(conf_int.loc[param_name, 1]),
                    "significant": p_corrected < 0.05,
                })

        beta_df = pd.DataFrame(beta_rows)

        return {
            "result": result,
            "summary": str(summary),
            "beta_g_table": beta_df,
            "significant_groups": beta_df[beta_df["significant"]]["group"].tolist(),
            "aic": float(result.aic),
            "bic": float(result.bic),
        }

    except Exception as e:
        return {
            "result": None,
            "error": str(e),
            "beta_g_table": pd.DataFrame(),
            "significant_groups": [],
        }


# ---------------------------------------------------------------------------
# Sensitivity analyses (Appendix D)
# ---------------------------------------------------------------------------

def sensitivity_analyses(
    df: pd.DataFrame,
    confounders: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Remove one confounder at a time and refit (Appendix D).
    Returns dict {removed_confounder: fit_result}.
    """
    confounders = confounders or ["complexity", "quality"]
    results = {}

    # Full model
    results["full"] = fit_mixed_effects(df, fixed_covariates=confounders, **kwargs)

    # Remove one at a time
    for removed in confounders:
        reduced = [c for c in confounders if c != removed]
        results[f"no_{removed}"] = fit_mixed_effects(df, fixed_covariates=reduced, **kwargs)

    # Intercept only
    results["intercept_only"] = fit_mixed_effects(df, fixed_covariates=[], **kwargs)

    return results
