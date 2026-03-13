"""
Ensemble Stacking Model.

Combines outputs from all pipeline branches (traditional PRS, CATN, TWAS)
into a final risk prediction using ridge regression or gradient boosting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


class EnsembleStacker:
    """
    Stacking ensemble that combines multiple PRS/score branches.

    Branches:
        A: Traditional PRS weights (PRS-CS, LDpred2, PRS-CSx, BridgePRS)
        B: Functionally-refined PRS weights (PolyFun + SuSiE-inf → PRS refinement)
        C: CATN deep learning risk scores
        D: TWAS gene-level scores

    The stacker learns optimal branch weights via cross-validated ridge/logistic
    regression on available validation data (or leave-one-study-out).

    Args:
        method: Stacking method — "ridge", "logistic", or "xgboost".
        cv_folds: Number of cross-validation folds for weight estimation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        method: str = "ridge",
        cv_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.method = method
        self.cv_folds = cv_folds
        self.seed = seed
        self.scaler = StandardScaler()
        self.model: Any = None
        self.branch_names: list[str] = []
        self.is_fitted = False

    def fit(
        self,
        branch_scores: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> EnsembleStacker:
        """
        Fit stacking model on branch-level scores and binary labels.

        Args:
            branch_scores: Dict mapping branch name → array of scores [n_samples].
            labels: Binary outcome array [n_samples] (0/1 for OA case/control).

        Returns:
            self (fitted stacker).
        """
        self.branch_names = sorted(branch_scores.keys())
        X = np.column_stack([branch_scores[b] for b in self.branch_names])
        y = labels.astype(int)

        logger.info(
            "Fitting ensemble stacker: method=%s, branches=%d, samples=%d",
            self.method, len(self.branch_names), len(y),
        )

        X_scaled = self.scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)

        if self.method == "ridge":
            self.model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 20), cv=cv)
            self.model.fit(X_scaled, y)
        elif self.method == "logistic":
            self.model = LogisticRegressionCV(
                Cs=20, cv=cv, penalty="l2", solver="lbfgs",
                max_iter=2000, random_state=self.seed,
            )
            self.model.fit(X_scaled, y)
        elif self.method == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError("xgboost required for method='xgboost'") from exc
            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.seed, use_label_encoder=False,
                eval_metric="logloss",
            )
            self.model.fit(X_scaled, y)
        else:
            raise ValueError(f"Unknown stacking method: {self.method}")

        self.is_fitted = True
        logger.info("Stacker fitted. Branch weights: %s", self.get_branch_weights())
        return self

    def predict_proba(self, branch_scores: dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict risk probabilities from branch scores.

        Args:
            branch_scores: Dict mapping branch name → score array [n_samples].

        Returns:
            Risk probability array [n_samples].
        """
        if not self.is_fitted:
            raise RuntimeError("Stacker not fitted. Call fit() first.")

        X = np.column_stack([branch_scores[b] for b in self.branch_names])
        X_scaled = self.scaler.transform(X)

        if self.method == "ridge":
            # RidgeClassifierCV doesn't have predict_proba; use decision function
            decision = self.model.decision_function(X_scaled)
            from scipy.special import expit
            return expit(decision)
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            decision = self.model.decision_function(X_scaled)
            from scipy.special import expit
            return expit(decision)

    def get_branch_weights(self) -> dict[str, float]:
        """Return learned weights for each branch (from model coefficients)."""
        if not self.is_fitted:
            return {}

        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_.flatten()
        elif hasattr(self.model, "feature_importances_"):
            coefs = self.model.feature_importances_
        else:
            return {b: np.nan for b in self.branch_names}

        return {name: float(w) for name, w in zip(self.branch_names, coefs)}

    def save(self, path: str | Path) -> None:
        """Save fitted stacker to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "branch_names": self.branch_names,
            "method": self.method,
        }, path)
        logger.info("Saved stacker to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> EnsembleStacker:
        """Load a fitted stacker from disk."""
        data = joblib.load(path)
        stacker = cls(method=data["method"])
        stacker.model = data["model"]
        stacker.scaler = data["scaler"]
        stacker.branch_names = data["branch_names"]
        stacker.is_fitted = True
        return stacker
