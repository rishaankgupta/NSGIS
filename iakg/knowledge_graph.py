"""
NSGIS Industrial Activity Knowledge Graph (IAKG)
==================================================
Encodes symbolic rules for 7 informal industrial activity archetypes.
Each rule is a probabilistic Datalog clause:
  confidence::activity(X, class) :- predicate_1(X), predicate_2(X), ...

The evaluate() method applies all rules to a feature vector and returns
the posterior probability distribution over activity classes.

Three-valued logic: predicates return (True, False, Unknown)
  - Unknown propagates as confidence degradation, not rule failure
  - Solves the open-world problem for occluded or sensor-limited cells
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import json


# ── Feature predicate thresholds (derived from IAKG Layer 1 specification) ──
THRESHOLDS = {
    # Spectral
    "high_swir":        0.16,   # B11 > threshold → metal/industrial surface
    "moderate_swir":    0.12,
    "high_nir":         0.15,   # B8 > threshold → some vegetation or dense structure
    "low_ndvi":         0.10,   # (B8-B4)/(B8+B4) < threshold → bare/industrial
    "very_low_ndvi":    0.05,

    # Thermal
    "extreme_thermal":  8.0,    # LST anomaly (°C) above climatological mean
    "high_thermal":     5.0,
    "moderate_thermal": 3.0,
    "low_thermal":      1.5,

    # SAR (dB — less negative = more backscatter = more metal/structure)
    "high_metal_vv":   -4.0,    # SAR_VV > threshold → metallic surfaces
    "moderate_metal_vv": -6.0,
    "high_metal_vh":   -12.0,

    # Context
    "near_water":       60,     # water_dist (m) < threshold
    "moderate_water":   150,
    "near_road":        50,
    "high_ntl":         25,     # nighttime light (NTL) > threshold
    "moderate_ntl":     15,
}


@dataclass
class Predicate:
    """A single feature-threshold test with uncertainty handling."""
    feature: str
    operator: str          # '>', '<', '>=', '<='
    threshold: float
    uncertainty_range: float = 0.0   # If |value - threshold| < range → Unknown

    def evaluate(self, features: dict) -> str:
        """Returns 'T', 'F', or 'U' (Unknown)."""
        val = features.get(self.feature)
        if val is None:
            return 'U'

        diff = abs(val - self.threshold)
        if self.uncertainty_range > 0 and diff < self.uncertainty_range:
            return 'U'

        if self.operator == '>':
            return 'T' if val > self.threshold else 'F'
        elif self.operator == '>=':
            return 'T' if val >= self.threshold else 'F'
        elif self.operator == '<':
            return 'T' if val < self.threshold else 'F'
        elif self.operator == '<=':
            return 'T' if val <= self.threshold else 'F'
        return 'U'


@dataclass
class Rule:
    """A probabilistic Datalog clause."""
    activity_class: int
    activity_name: str
    confidence: float          # Prior confidence if all predicates fire
    predicates: List[Predicate]
    required: List[str]        # Predicate indices that must be 'T' (not 'U')
    description: str = ""

    def evaluate(self, features: dict) -> Tuple[float, str]:
        """
        Returns (posterior_probability, trace_string).
        Unknown predicates degrade confidence multiplicatively.
        """
        results = [p.evaluate(features) for p in self.predicates]
        trace_parts = []
        posterior = self.confidence
        fired = True

        for i, (pred, result) in enumerate(zip(self.predicates, results)):
            if result == 'F':
                fired = False
                trace_parts.append(f"FAIL:{pred.feature}{pred.operator}{pred.threshold}")
                break
            elif result == 'U':
                # Unknown: degrade confidence by 35% per unknown predicate
                posterior *= 0.65
                trace_parts.append(f"UNK:{pred.feature}(±{pred.uncertainty_range})")
            else:
                trace_parts.append(f"OK:{pred.feature}{pred.operator}{pred.threshold}")

        if not fired:
            return 0.0, f"Rule failed at: {', '.join(trace_parts)}"

        return posterior, f"Rule fired [{posterior:.2f}]: {' ∧ '.join(trace_parts)}"


class IAKG:
    """
    Industrial Activity Knowledge Graph.
    Contains all symbolic rules and evaluates them against feature vectors.
    """

    def __init__(self):
        self.rules: List[Rule] = []
        self._build_rules()

    def _p(self, feat, op, thresh, unc=0.0):
        return Predicate(feat, op, thresh, unc)

    def _build_rules(self):
        """Encode all activity rules. Each activity has multiple rules
        representing different production configurations and regional variants."""

        # ── CLASS 0: Non-industrial (exclusion rules) ─────────────────────────
        # (Handled implicitly — low posterior across all industrial rules)

        # ── CLASS 1: Tannery ──────────────────────────────────────────────────
        self.rules.append(Rule(
            activity_class=1, activity_name="Tannery",
            confidence=0.87,
            predicates=[
                self._p("B11", ">", THRESHOLDS["high_swir"], 0.02),       # metal roof
                self._p("LST_anomaly", ">", THRESHOLDS["high_thermal"], 0.8),  # lime pits
                self._p("water_dist", "<", THRESHOLDS["near_water"], 8.0), # water dependency
                self._p("SAR_VV", ">", THRESHOLDS["high_metal_vv"], 0.5), # metal structure
                self._p("ndvi", "<", THRESHOLDS["low_ndvi"], 0.01),        # no vegetation
            ],
            required=["B11", "LST_anomaly", "water_dist"],
            description="Primary tannery rule: metal roof + thermal + near water"
        ))
        self.rules.append(Rule(
            activity_class=1, activity_name="Tannery (low-temp variant)",
            confidence=0.71,
            predicates=[
                self._p("B11", ">", THRESHOLDS["moderate_swir"], 0.02),
                self._p("LST_anomaly", ">", THRESHOLDS["moderate_thermal"], 0.8),
                self._p("water_dist", "<", THRESHOLDS["near_water"], 8.0),
                self._p("SAR_VV", ">", THRESHOLDS["moderate_metal_vv"], 0.5),
            ],
            required=["water_dist"],
            description="Vegetable tanning variant: lower thermal, still near water"
        ))

        # ── CLASS 2: Textile Dyeing ───────────────────────────────────────────
        self.rules.append(Rule(
            activity_class=2, activity_name="Textile Dyeing",
            confidence=0.83,
            predicates=[
                self._p("B11", ">", THRESHOLDS["moderate_swir"], 0.02),
                self._p("LST_anomaly", ">", THRESHOLDS["moderate_thermal"], 0.8),
                self._p("water_dist", "<", THRESHOLDS["moderate_water"], 15.0),
                self._p("ntl", ">", THRESHOLDS["moderate_ntl"], 3.0),   # night shifts
            ],
            required=["B11", "LST_anomaly"],
            description="Textile dyeing: steam + moderate water use + night activity"
        ))
        self.rules.append(Rule(
            activity_class=2, activity_name="Textile Dyeing (effluent variant)",
            confidence=0.78,
            predicates=[
                self._p("B11", ">", THRESHOLDS["moderate_swir"], 0.02),
                self._p("water_dist", "<", THRESHOLDS["near_water"], 8.0),
                self._p("ndvi", "<", THRESHOLDS["very_low_ndvi"], 0.01), # chemical burn
            ],
            required=["water_dist", "ndvi"],
            description="High-effluent dyeing: chemical discharge near water, dead vegetation"
        ))

        # ── CLASS 3: E-waste Processing ───────────────────────────────────────
        self.rules.append(Rule(
            activity_class=3, activity_name="E-waste Processing",
            confidence=0.91,
            predicates=[
                self._p("LST_anomaly", ">", THRESHOLDS["extreme_thermal"], 1.0),  # burning
                self._p("B11", ">", THRESHOLDS["high_swir"], 0.02),               # scrap metal
                self._p("SAR_VV", ">", THRESHOLDS["high_metal_vv"], 0.5),         # dense metal
                self._p("ndvi", "<", THRESHOLDS["very_low_ndvi"], 0.01),
            ],
            required=["LST_anomaly", "SAR_VV"],
            description="E-waste open burning: extreme heat + dense metallic debris"
        ))
        self.rules.append(Rule(
            activity_class=3, activity_name="E-waste (sorting, no burning)",
            confidence=0.72,
            predicates=[
                self._p("B11", ">", THRESHOLDS["high_swir"], 0.02),
                self._p("SAR_VV", ">", THRESHOLDS["high_metal_vv"], 0.5),
                self._p("SAR_VH", ">", THRESHOLDS["high_metal_vh"], 0.5),
                self._p("road_dist", "<", THRESHOLDS["near_road"], 5.0),  # roadside sorting
            ],
            required=["SAR_VV", "SAR_VH"],
            description="E-waste sorting yard: metallic scrap without active burning"
        ))

        # ── CLASS 4: Metal/Battery Recycling ──────────────────────────────────
        self.rules.append(Rule(
            activity_class=4, activity_name="Metal Recycling",
            confidence=0.84,
            predicates=[
                self._p("B11", ">", THRESHOLDS["high_swir"], 0.02),
                self._p("LST_anomaly", ">", THRESHOLDS["high_thermal"], 0.8),
                self._p("SAR_VV", ">", THRESHOLDS["high_metal_vv"], 0.5),
                self._p("ntl", ">", THRESHOLDS["high_ntl"], 3.0),        # smelting at night
            ],
            required=["B11", "SAR_VV", "LST_anomaly"],
            description="Metal smelting: high thermal + metallic roof + night operations"
        ))
        self.rules.append(Rule(
            activity_class=4, activity_name="Battery Acid Recycling",
            confidence=0.76,
            predicates=[
                self._p("B11", ">", THRESHOLDS["moderate_swir"], 0.02),
                self._p("LST_anomaly", ">", THRESHOLDS["moderate_thermal"], 0.8),
                self._p("ndvi", "<", THRESHOLDS["very_low_ndvi"], 0.01), # acid kills veg
                self._p("water_dist", "<", THRESHOLDS["moderate_water"], 15.0),
            ],
            required=["ndvi"],
            description="Lead-acid battery recycling: acid effluent, dead vegetation"
        ))

        # ── CLASS 5: Chemical Storage ─────────────────────────────────────────
        self.rules.append(Rule(
            activity_class=5, activity_name="Chemical Storage",
            confidence=0.79,
            predicates=[
                self._p("B11", ">", THRESHOLDS["moderate_swir"], 0.02),
                self._p("LST_anomaly", "<", THRESHOLDS["moderate_thermal"], 0.8),  # NOT hot
                self._p("SAR_VV", ">", THRESHOLDS["moderate_metal_vv"], 0.5),
                self._p("ntl", ">", THRESHOLDS["moderate_ntl"], 3.0),
            ],
            required=["B11"],
            description="Chemical storage: metal tanks, low thermal (no processing)"
        ))

        # ── CLASS 6: Food/Organic Processing ─────────────────────────────────
        self.rules.append(Rule(
            activity_class=6, activity_name="Food/Organic Processing",
            confidence=0.81,
            predicates=[
                self._p("B8", ">", THRESHOLDS["high_nir"], 0.02),         # organic matter
                self._p("LST_anomaly", "<", THRESHOLDS["moderate_thermal"], 0.8),
                self._p("SAR_VV", "<", THRESHOLDS["moderate_metal_vv"], 0.5), # non-metallic
                self._p("road_dist", "<", THRESHOLDS["near_road"], 5.0),  # market access
            ],
            required=[],
            description="Food processing: organic signature, low thermal, road access"
        ))

    def compute_ndvi(self, features: dict) -> float:
        b8 = features.get("B8", 0.1)
        b4 = features.get("B4", 0.1)
        denom = b8 + b4
        if denom == 0:
            return 0.0
        return (b8 - b4) / denom

    def evaluate(self, feature_vector: np.ndarray, feature_names: list) -> dict:
        """
        Evaluate all rules against a feature vector.

        Returns:
            dict with:
              - posteriors: {class_id: probability}
              - tier: TNFD confidence tier (1-4)
              - dominant_class: most probable activity
              - confidence: max posterior
              - traces: {class_id: [rule traces]}
        """
        # Build feature dict
        features = {name: float(val) for name, val in zip(feature_names, feature_vector)}
        features["ndvi"] = self.compute_ndvi(features)

        # Accumulate posteriors per class using noisy-OR combination
        class_posteriors = {c: 0.0 for c in range(7)}
        class_traces = {c: [] for c in range(7)}

        for rule in self.rules:
            prob, trace = rule.evaluate(features)
            if prob > 0:
                cls = rule.activity_class
                # Noisy-OR: P_combined = 1 - (1-P1)(1-P2)
                class_posteriors[cls] = 1.0 - (1.0 - class_posteriors[cls]) * (1.0 - prob)
                class_traces[cls].append(trace)

        # Non-industrial prior: high if all industrial posteriors are low
        max_industrial = max(class_posteriors[c] for c in range(1, 7))
        class_posteriors[0] = max(0.0, 1.0 - max_industrial * 1.5)

        # Normalize
        total = sum(class_posteriors.values())
        if total > 0:
            posteriors = {c: p / total for c, p in class_posteriors.items()}
        else:
            posteriors = {c: 1/7 for c in range(7)}

        # Dominant class
        dominant = max(posteriors, key=posteriors.get)
        confidence = posteriors[dominant]

        # TNFD Tier assignment
        if confidence >= 0.75:
            tier = 1
        elif confidence >= 0.40:
            tier = 2
        elif confidence >= 0.15:
            tier = 3
        else:
            tier = 4

        return {
            "posteriors": posteriors,
            "dominant_class": dominant,
            "confidence": confidence,
            "tier": tier,
            "traces": {c: t for c, t in class_traces.items() if t},
            "ndvi": features["ndvi"]
        }

    def batch_evaluate(self, X: np.ndarray, feature_names: list) -> list:
        """Evaluate entire feature matrix. Returns list of result dicts."""
        return [self.evaluate(X[i], feature_names) for i in range(len(X))]

    def get_tnfd_impact_profile(self, activity_class: int) -> dict:
        """Return TNFD LEAP impact driver profile for an activity class."""
        profiles = {
            0: {"impact_drivers": [], "water_intensity": "Low", "pollution_type": "None", "land_use_change": "Stable"},
            1: {"impact_drivers": ["Water pollution", "Chemical efflux (Cr, Na2S)", "Thermal discharge"],
                "water_intensity": "Very High", "pollution_type": "Heavy metals, sulfides", "land_use_change": "Degrading"},
            2: {"impact_drivers": ["Water pollution", "Chemical efflux (azo dyes)", "Thermal discharge"],
                "water_intensity": "High", "pollution_type": "Synthetic dyes, fixatives", "land_use_change": "Degrading"},
            3: {"impact_drivers": ["Air pollution (POPs, dioxins)", "Soil contamination", "Thermal anomaly"],
                "water_intensity": "Low", "pollution_type": "Heavy metals, PCBs, flame retardants", "land_use_change": "Severely Degrading"},
            4: {"impact_drivers": ["Air pollution (particulates)", "Soil contamination (Pb, Cd)", "Water table contamination"],
                "water_intensity": "Moderate", "pollution_type": "Lead, cadmium, sulfuric acid", "land_use_change": "Degrading"},
            5: {"impact_drivers": ["Soil contamination", "Groundwater risk", "Air quality (VOCs)"],
                "water_intensity": "Low", "pollution_type": "Volatile organics, solvents", "land_use_change": "Stable/Risk"},
            6: {"impact_drivers": ["Organic waste discharge", "Odor/air quality"],
                "water_intensity": "Moderate", "pollution_type": "BOD load, pathogens", "land_use_change": "Moderate"},
        }
        return profiles.get(activity_class, profiles[0])


if __name__ == "__main__":
    iakg = IAKG()
    print(f"[IAKG] Loaded {len(iakg.rules)} rules across 7 activity classes")

    # Quick self-test with a known e-waste signature
    test_vec = np.array([0.06, 0.06, 0.07, 0.08, 0.09, 0.09, 0.09, 0.09,
                          0.22, 0.19,  # high SWIR
                          9.5,         # extreme thermal
                          -2.1, -9.8,  # high SAR
                          20, 80, 35, 8])  # context
    from data.sentinel_simulator import ALL_FEATURES
    result = iakg.evaluate(test_vec, ALL_FEATURES)
    print(f"[IAKG] E-waste test → class {result['dominant_class']} "
          f"({result['confidence']:.2f} confidence, Tier {result['tier']})")
    print(f"       Posteriors: { {k: f'{v:.2f}' for k,v in result['posteriors'].items()} }")
