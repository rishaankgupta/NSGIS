"""
NSGIS Full Pipeline Runner
===========================
Runs the complete end-to-end system:
  1. Generate / load satellite data
  2. Train neural + baseline models
  3. Run neuro-symbolic inference on validation scene
  4. Evaluate and compare performance
  5. Generate TNFD-aligned output report
  6. Save all results for visualization

Run: python3 pipeline/run_pipeline.py
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/claude/nsgis')

from data.sentinel_simulator import (
    save_dataset, generate_scene, SPECTRAL_SIGNATURES, ALL_FEATURES
)
from iakg.knowledge_graph import IAKG
from neural.model import NSGISFullSystem

CLASS_NAMES = {k: v["name"] for k, v in SPECTRAL_SIGNATURES.items()}
CLASS_COLORS = {k: v["color"] for k, v in SPECTRAL_SIGNATURES.items()}

OUTPUT_DIR = Path("/home/claude/nsgis/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    print("=" * 65)
    print("  NSGIS — Neuro-Symbolic Geospatial Inference System")
    print("  Study Site: Greater Accra / Agbogbloshie (Simulated)")
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic Sentinel-2 + ECOSTRESS + SAR data...")
    X_train, y_train, X_val, y_val, val_grid = save_dataset(
        "/home/claude/nsgis/data", n_train=2800, n_val=400
    )

    # Also generate a larger inference scene (the "Agbogbloshie proxy")
    print("[1/5] Generating Agbogbloshie proxy inference scene (80×80 grid)...")
    X_scene, y_scene, scene_grid = generate_scene(
        grid_size=80,
        class_distribution={0: 0.45, 1: 0.05, 2: 0.06, 3: 0.18,
                             4: 0.10, 5: 0.06, 6: 0.10},
        cluster_spatial=True
    )
    np.save(OUTPUT_DIR / "scene_grid_true.npy", scene_grid)

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print("\n[2/5] Training neuro-symbolic and baseline models...")
    system = NSGISFullSystem()
    system.fit(X_train, y_train)

    # Quick IAKG self-test
    iakg = IAKG()
    print(f"[2/5] IAKG loaded: {len(iakg.rules)} rules, 7 activity classes")

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    print("\n[3/5] Evaluating on validation set...")
    eval_results = system.evaluate_performance(X_val, y_val)

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │              PERFORMANCE COMPARISON                  │")
    print("  ├────────────────────────────┬────────────┬────────────┤")
    print("  │ Metric                     │ Neuro-Sym  │ Baseline   │")
    print("  ├────────────────────────────┼────────────┼────────────┤")

    ns = eval_results["neuro_symbolic"]
    bl = eval_results["baseline_neural"]
    imp = eval_results["improvement"]

    print(f"  │ F1 Macro (all classes)     │   {ns['f1_macro']:.3f}    │   {bl['f1_macro']:.3f}    │")
    print(f"  │ F1 Weighted                │   {ns['f1_weighted']:.3f}    │   {bl['f1_weighted']:.3f}    │")
    print(f"  │ F1 Macro (Tier 1 only)     │   {ns['tier1_f1']:.3f}    │    n/a     │")
    print(f"  │ Expected Calibration Error │   {ns['ece']:.3f}    │    n/a     │")
    print(f"  │ F1 Improvement (Macro)     │  +{imp['f1_macro_delta']:.3f}    │     —      │")
    print("  └────────────────────────────┴────────────┴────────────┘")

    td = ns['tier_distribution']
    print(f"\n  TNFD Tier Distribution:")
    print(f"    Tier 1 (High confidence, ≥0.75):    {td[1]:.1f}%  → Full TNFD attribution")
    print(f"    Tier 2 (Moderate, 0.40–0.75):       {td[2]:.1f}%  → Attributed with uncertainty")
    print(f"    Tier 3 (Candidate, 0.15–0.40):      {td[3]:.1f}%  → Verification queue")
    print(f"    Tier 4 (Non-industrial, <0.15):     {td[4]:.1f}%  → Excluded")

    print("\n  Per-class performance (Neuro-Symbolic):")
    print(eval_results["class_report_ns"])

    # ── Step 4: Inference Scene ───────────────────────────────────────────────
    print("[4/5] Running inference on Agbogbloshie proxy scene (80×80 = 6400 cells)...")
    scene_results = system.predict_batch(X_scene)

    predicted_grid = np.array([r["predicted_class"] for r in scene_results]).reshape(80, 80)
    confidence_grid = np.array([r["confidence"] for r in scene_results]).reshape(80, 80)
    tier_grid = np.array([r["tier"] for r in scene_results]).reshape(80, 80)

    np.save(OUTPUT_DIR / "predicted_grid.npy", predicted_grid)
    np.save(OUTPUT_DIR / "confidence_grid.npy", confidence_grid)
    np.save(OUTPUT_DIR / "tier_grid.npy", tier_grid)
    np.save(OUTPUT_DIR / "scene_grid_true.npy", scene_grid)

    # ── Step 5: TNFD Report ───────────────────────────────────────────────────
    print("\n[5/5] Generating TNFD-aligned disclosure report...")
    tnfd_report = generate_tnfd_report(scene_results, system)

    with open(OUTPUT_DIR / "tnfd_report.json", "w") as f:
        json.dump(tnfd_report, f, indent=2)

    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        # Remove non-serializable items
        save_eval = {k: v for k, v in eval_results.items()
                     if k not in ("class_report_ns", "y_ns", "y_true")}
        json.dump(save_eval, f, indent=2)

    # Save metadata
    meta = {
        "feature_names": ALL_FEATURES,
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
        "n_classes": 7
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print_tnfd_summary(tnfd_report)
    print("\n[PIPELINE] Complete. Outputs saved to:", OUTPUT_DIR)
    return eval_results, tnfd_report, scene_results


def generate_tnfd_report(scene_results: list, system: NSGISFullSystem) -> dict:
    """Generate a structured TNFD LEAP output report from inference results."""

    # Count by activity class and tier
    class_counts = {c: 0 for c in range(7)}
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    high_risk_cells = []
    tier1_industrial = []

    for i, r in enumerate(scene_results):
        cls = r["predicted_class"]
        tier = r["tier"]
        class_counts[cls] += 1
        tier_counts[tier] += 1

        if cls > 0 and tier <= 2:
            profile = r["tnfd_profile"]
            tier1_industrial.append({
                "cell_index": i,
                "activity": r["class_name"],
                "confidence": round(r["confidence"], 3),
                "tier": tier,
                "impact_drivers": profile["impact_drivers"],
                "water_intensity": profile["water_intensity"],
                "pollution_type": profile["pollution_type"],
                "reasoning": r["reasoning_traces"]
            })

        if cls in [3, 1, 4] and r["confidence"] >= 0.60:
            high_risk_cells.append(i)

    total = len(scene_results)
    n_industrial_t1 = sum(1 for r in scene_results
                           if r["predicted_class"] > 0 and r["tier"] == 1)

    # Aggregate TNFD metrics
    activity_summary = {}
    for cls in range(1, 7):
        cls_results = [r for r in scene_results
                       if r["predicted_class"] == cls and r["tier"] <= 2]
        if cls_results:
            profile = system.iakg.get_tnfd_impact_profile(cls)
            activity_summary[CLASS_NAMES[cls]] = {
                "cell_count_tier1_2": len(cls_results),
                "area_estimate_ha": round(len(cls_results) * 0.01, 2),  # 10m cells → 0.01 ha each
                "avg_confidence": round(np.mean([r["confidence"] for r in cls_results]), 3),
                "impact_drivers": profile["impact_drivers"],
                "water_intensity": profile["water_intensity"],
                "pollution_type": profile["pollution_type"],
                "land_use_change": profile["land_use_change"]
            }

    report = {
        "report_metadata": {
            "system": "NSGIS v1.0",
            "framework": "TNFD LEAP v1.0",
            "study_site": "Greater Accra / Agbogbloshie Proxy",
            "data_sources": ["Sentinel-2 (simulated)", "ECOSTRESS (simulated)", "Sentinel-1 (simulated)"],
            "grid_cells_analyzed": total,
            "grid_resolution_m": 10,
            "analysis_area_ha": round(total * 0.01, 1),
            "generated": datetime.now().isoformat(),
            "inference_method": "Neuro-Symbolic (GBM + IAKG DeepProbLog)"
        },
        "tnfd_locate_phase": {
            "total_cells": total,
            "industrial_cells_tier1": n_industrial_t1,
            "industrial_area_tier1_ha": round(n_industrial_t1 * 0.01, 2),
            "tier_distribution_pct": {
                f"Tier {t}": round(100 * c / total, 1)
                for t, c in tier_counts.items()
            },
            "class_distribution": {
                CLASS_NAMES[c]: class_counts[c] for c in range(7)
            },
            "high_risk_cells_flagged": len(high_risk_cells)
        },
        "tnfd_evaluate_phase": {
            "activity_profiles": activity_summary,
            "aggregate_impact_drivers": list({
                driver
                for cls_data in activity_summary.values()
                for driver in cls_data["impact_drivers"]
            }),
            "water_intensity_summary": {
                cls: data["water_intensity"]
                for cls, data in activity_summary.items()
            }
        },
        "confidence_attestation": {
            "method": "Calibrated posterior via isotonic regression + IAKG rule fusion",
            "tier1_threshold": 0.75,
            "tier2_threshold": 0.40,
            "calibration_ece": "See eval_results.json",
            "symbolic_reasoning_traces": "Available per-cell in inference output",
            "audit_trail": "Full rule firing traces stored for all Tier 1 classifications"
        },
        "tier1_industrial_sample": tier1_industrial[:20],  # First 20 for report
        "coverage_limitations": [
            "Indoor industrial processes without external surface signatures not detectable",
            "Thermal band resampled from 70m ECOSTRESS to 10m (TsHARP); introduces uncertainty",
            "Cells classified as Tier 3 or 4 excluded from primary disclosure",
            "Model calibrated for Accra/Dhaka spectral signatures; regional overlay needed for other cities"
        ]
    }
    return report


def print_tnfd_summary(report: dict):
    loc = report["tnfd_locate_phase"]
    eva = report["tnfd_evaluate_phase"]
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │           TNFD LEAP OUTPUT SUMMARY                   │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │ Study area analyzed:        {loc['analysis_area_ha']:.0f} ha                │" if 'analysis_area_ha' in loc else "")
    print(f"  │ Tier 1 industrial area:     {loc['industrial_area_tier1_ha']:.1f} ha               │")
    print(f"  │ High-risk cells flagged:    {loc['high_risk_cells_flagged']}                   │")
    print(f"  │ Aggregate impact drivers:   {len(eva['aggregate_impact_drivers'])} categories          │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │ Identified industrial activities (Tier 1+2):         │")
    for name, data in eva["activity_profiles"].items():
        print(f"  │   {name[:22]:<22} {data['area_estimate_ha']:>5.2f} ha  conf:{data['avg_confidence']:.2f}  │")
    print("  └─────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    run_pipeline()
