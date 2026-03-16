"""
NSGIS Neural Perception Module + Neuro-Symbolic Fusion
=======================================================
Neural module: Gradient Boosting ensemble trained to predict
  feature predicate activation probabilities from raw band values.
  (In production: replace with Clay foundation model fine-tuned head)

Symbolic module: IAKG rule evaluator (see iakg/knowledge_graph.py)

Fusion: The neural module's outputs are used as input features to
  the symbolic layer, enabling end-to-end neuro-symbolic inference.

Baseline comparison: Pure neural classifier (no symbolic reasoning)
  is also trained for performance benchmarking.
"""

import numpy as np
import json
import sys
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
import pickle

sys.path.insert(0, '/home/claude/nsgis')
from data.sentinel_simulator import SPECTRAL_SIGNATURES, ALL_FEATURES
from iakg.knowledge_graph import IAKG

CLASS_NAMES = {k: v["name"] for k, v in SPECTRAL_SIGNATURES.items()}
N_CLASSES = len(SPECTRAL_SIGNATURES)


class NSGISNeuralModule:
    """
    Neural perception module using a calibrated gradient boosting ensemble.
    Outputs calibrated posterior probabilities for each activity class.
    This approximates the feature→probability prediction that a Clay-based
    Vision Transformer would perform on real satellite patches.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=10, random_state=42
            ),
            method='isotonic', cv=3
        )
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"[NEURAL] Model trained on {len(X_train)} samples")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated probability matrix (N, n_classes)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/neural_model.pkl", "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model}, f)

    def load(self, path: str):
        with open(f"{path}/neural_model.pkl", "rb") as f:
            obj = pickle.load(f)
        self.scaler = obj["scaler"]
        self.model = obj["model"]
        self.is_trained = True


class NSGISBaselineModule:
    """
    Pure neural baseline for benchmarking against the neuro-symbolic system.
    Represents the state-of-the-art CNN/ViT approach WITHOUT symbolic reasoning.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        self.is_trained = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class NSGISFullSystem:
    """
    Full neuro-symbolic system combining neural + IAKG symbolic reasoning.

    Fusion strategy:
      1. Neural module predicts class probabilities from raw features
      2. IAKG symbolic reasoner independently evaluates feature predicates
      3. Final posterior = weighted geometric mean of neural and symbolic posteriors
         with weight determined by IAKG confidence (higher rule confidence → more symbolic)
    """

    def __init__(self):
        self.neural = NSGISNeuralModule()
        self.baseline = NSGISBaselineModule()
        self.iakg = IAKG()
        self.symbolic_weight = 0.55   # Symbolic reasoning weight in fusion
        self.neural_weight = 0.45

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.neural.fit(X_train, y_train)
        self.baseline.fit(X_train, y_train)

    def _fuse(self, neural_proba: np.ndarray, symbolic_result: dict) -> np.ndarray:
        """
        Fuse neural probability vector with symbolic posterior dict.
        Returns fused probability vector.
        """
        symbolic_vec = np.array([
            symbolic_result["posteriors"][c] for c in range(N_CLASSES)
        ])

        # Adaptive weighting: if IAKG is highly confident, trust it more
        sym_conf = symbolic_result["confidence"]
        if sym_conf >= 0.75:
            sw, nw = 0.65, 0.35
        elif sym_conf >= 0.40:
            sw, nw = 0.55, 0.45
        else:
            sw, nw = 0.30, 0.70   # Low symbolic confidence → trust neural more

        # Geometric mean fusion
        fused = (neural_proba ** nw) * (symbolic_vec ** sw)
        fused_sum = fused.sum()
        if fused_sum > 0:
            fused = fused / fused_sum
        else:
            fused = neural_proba  # Fallback to neural only
        return fused

    def predict_single(self, x: np.ndarray) -> dict:
        """Full inference for a single feature vector."""
        # Neural prediction
        neural_proba = self.neural.predict_proba(x.reshape(1, -1))[0]

        # Symbolic reasoning
        symbolic_result = self.iakg.evaluate(x, ALL_FEATURES)

        # Fusion
        fused_proba = self._fuse(neural_proba, symbolic_result)

        dominant = int(fused_proba.argmax())
        confidence = float(fused_proba.max())

        # TNFD tier
        if confidence >= 0.75:
            tier = 1
        elif confidence >= 0.40:
            tier = 2
        elif confidence >= 0.15:
            tier = 3
        else:
            tier = 4

        return {
            "predicted_class": dominant,
            "class_name": CLASS_NAMES[dominant],
            "confidence": confidence,
            "tier": tier,
            "fused_posteriors": fused_proba.tolist(),
            "neural_posteriors": neural_proba.tolist(),
            "symbolic_confidence": symbolic_result["confidence"],
            "symbolic_tier": symbolic_result["tier"],
            "reasoning_traces": symbolic_result["traces"],
            "ndvi": symbolic_result["ndvi"],
            "tnfd_profile": self.iakg.get_tnfd_impact_profile(dominant)
        }

    def predict_batch(self, X: np.ndarray) -> list:
        return [self.predict_single(X[i]) for i in range(len(X))]

    def evaluate_performance(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Full evaluation comparing neuro-symbolic vs baseline."""
        print("\n[EVAL] Running evaluation...")

        # Neuro-symbolic predictions
        ns_results = self.predict_batch(X_val)
        y_ns = np.array([r["predicted_class"] for r in ns_results])
        y_ns_conf = np.array([r["confidence"] for r in ns_results])

        # Baseline predictions
        y_base = self.baseline.predict(X_val)

        # Metrics
        ns_f1_macro = f1_score(y_val, y_ns, average='macro', zero_division=0)
        ns_f1_weighted = f1_score(y_val, y_ns, average='weighted', zero_division=0)
        base_f1_macro = f1_score(y_val, y_base, average='macro', zero_division=0)
        base_f1_weighted = f1_score(y_val, y_base, average='weighted', zero_division=0)

        # Tier 1 coverage
        tier1_mask = np.array([r["tier"] == 1 for r in ns_results])
        tier1_f1 = f1_score(y_val[tier1_mask], y_ns[tier1_mask],
                             average='macro', zero_division=0) if tier1_mask.sum() > 0 else 0

        # Expected Calibration Error
        ece = self._compute_ece(y_val, y_ns, y_ns_conf)

        # Tier distribution
        tiers = [r["tier"] for r in ns_results]
        tier_counts = {t: tiers.count(t) for t in [1, 2, 3, 4]}
        tier_pct = {t: 100 * c / len(tiers) for t, c in tier_counts.items()}

        results = {
            "neuro_symbolic": {
                "f1_macro": ns_f1_macro,
                "f1_weighted": ns_f1_weighted,
                "tier1_f1": tier1_f1,
                "ece": ece,
                "tier_distribution": tier_pct,
                "tier_counts": tier_counts,
            },
            "baseline_neural": {
                "f1_macro": base_f1_macro,
                "f1_weighted": base_f1_weighted,
            },
            "improvement": {
                "f1_macro_delta": ns_f1_macro - base_f1_macro,
                "f1_weighted_delta": ns_f1_weighted - base_f1_weighted,
            },
            "class_report_ns": classification_report(
                y_val, y_ns,
                target_names=[CLASS_NAMES[c] for c in range(N_CLASSES)],
                zero_division=0
            ),
            "y_ns": y_ns.tolist(),
            "y_true": y_val.tolist(),
        }
        return results

    def _compute_ece(self, y_true, y_pred, confidences, n_bins=10):
        """Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() == 0:
                continue
            bin_acc = (y_pred[mask] == y_true[mask]).mean()
            bin_conf = confidences[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)
        return float(ece)

    def save(self, path: str):
        self.neural.save(path)

    def load(self, path: str):
        self.neural.load(path)
        self.is_trained = True
