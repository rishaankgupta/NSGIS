"""
NSGIS Visualization Suite — Fixed version
Place in: nsgis/viz/visualize.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

# Fix path resolution for Windows
_HERE    = Path(__file__).parent
_NSGIS   = _HERE.parent
sys.path.insert(0, str(_NSGIS))
sys.path.insert(0, str(_NSGIS.parent))

from data.sentinel_simulator import SPECTRAL_SIGNATURES, ALL_FEATURES

# ── Paths — Windows compatible ─────────────────────────────────────────────────
OUTPUT_DIR = _NSGIS / "outputs"   # reads grids from here
VIZ_DIR    = _NSGIS / "outputs"   # saves figures here too

CLASS_NAMES      = {k: v["name"]  for k, v in SPECTRAL_SIGNATURES.items()}
CLASS_COLORS_LIST = [v["color"]   for v in SPECTRAL_SIGNATURES.values()]
TIER_COLORS      = {1: "#1B5E20", 2: "#F9A825", 3: "#E65100", 4: "#B0BEC5"}


def load_outputs():
    grids = {}
    for fname in ["predicted_grid", "confidence_grid",
                  "tier_grid", "scene_grid_true"]:
        p = OUTPUT_DIR / f"{fname}.npy"
        if p.exists():
            grids[fname] = np.load(str(p))
        else:
            print(f"[VIZ] Warning: {fname}.npy not found")

    with open(str(OUTPUT_DIR / "eval_results.json")) as f:
        eval_results = json.load(f)
    with open(str(OUTPUT_DIR / "tnfd_report.json")) as f:
        tnfd_report = json.load(f)
    with open(str(OUTPUT_DIR / "metadata.json")) as f:
        meta = json.load(f)

    return grids, eval_results, tnfd_report, meta


def fig1_activity_maps(grids, meta):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("NSGIS Activity Classification — Agbogbloshie, Greater Accra\n"
                 "Sentinel-2 L2A 2026-02-01 | 10m resolution | Real Data",
                 fontsize=13, fontweight='bold', y=1.01)

    cmap = ListedColormap(CLASS_COLORS_LIST)
    norm = BoundaryNorm(list(range(8)), cmap.N)

    titles = ["Ground Truth (Simulated Reference)", "NSGIS Predicted"]
    keys   = ["scene_grid_true", "predicted_grid"]
    for ax, key, title in zip(axes, keys, titles):
        if key in grids:
            ax.imshow(grids[key], cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Easting (10m cells)")
        ax.set_ylabel("Northing (10m cells)")

    patches = [mpatches.Patch(color=CLASS_COLORS_LIST[i], label=CLASS_NAMES[i])
               for i in range(7)]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.08), framealpha=0.95)
    plt.tight_layout()
    plt.savefig(str(VIZ_DIR / "fig1_activity_maps.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[VIZ] Fig 1: Activity maps saved")


def fig2_tier_map(grids):
    if "tier_grid" not in grids:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    tier_cmap = ListedColormap([TIER_COLORS[t] for t in [1,2,3,4]])
    norm      = BoundaryNorm([0.5,1.5,2.5,3.5,4.5], tier_cmap.N)
    ax.imshow(grids["tier_grid"], cmap=tier_cmap, norm=norm, interpolation='nearest')
    ax.set_title("TNFD Confidence Tier Map\nAgbogbloshie, Greater Accra",
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Easting (10m cells)")
    ax.set_ylabel("Northing (10m cells)")
    tier_labels = {
        1: "Tier 1: Full attribution (>=0.75)",
        2: "Tier 2: With uncertainty (0.40-0.75)",
        3: "Tier 3: Verification queue (0.15-0.40)",
        4: "Tier 4: Cloud masked / excluded"
    }
    patches = [mpatches.Patch(color=TIER_COLORS[t], label=tier_labels[t])
               for t in [1,2,3,4]]
    ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(str(VIZ_DIR / "fig2_tier_map.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[VIZ] Fig 2: Tier map saved")


def fig3_performance_comparison(eval_results):
    ns = eval_results["neuro_symbolic"]
    bl = eval_results["baseline_neural"]

    metrics  = ["F1 Macro\n(all classes)", "F1 Weighted", "F1 Macro\n(Tier 1)"]
    ns_vals  = [ns["f1_macro"], ns["f1_weighted"], ns["tier1_f1"]]
    bl_vals  = [bl["f1_macro"], bl["f1_weighted"]]

    x, w = np.arange(len(metrics)), 0.32
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, ns_vals,    w, label='Neuro-Symbolic (NSGIS)',
           color='#1B3A5C', alpha=0.9, edgecolor='white')
    ax.bar(x[:2]+w/2, bl_vals,  w, label='Baseline Neural',
           color='#90A4AE', alpha=0.9, edgecolor='white')

    for bar in ax.patches[:3]:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                    f'{h:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#1B3A5C')

    ax.set_xlabel("Performance Metric", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("NSGIS vs Baseline Neural: Performance — Real Sentinel-2 Data\n"
                 "Agbogbloshie, Greater Accra", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.axhline(y=0.72, color='#E53935', linestyle='--', alpha=0.6)
    ax.text(2.4, 0.73, 'Target (0.72)', color='#E53935', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.text(0.02, 0.97, f"ECE: {ns['ece']:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
    plt.tight_layout()
    plt.savefig(str(VIZ_DIR / "fig3_performance.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[VIZ] Fig 3: Performance comparison saved")


def fig4_risk_heatmap(grids):
    if "predicted_grid" not in grids:
        return
    risk_weights = {0:0.0, 1:0.85, 2:0.70, 3:1.00, 4:0.88, 5:0.65, 6:0.30}
    pred   = grids["predicted_grid"]
    conf   = grids.get("confidence_grid", np.ones_like(pred, dtype=float))
    tiers  = grids.get("tier_grid", np.ones_like(pred))

    risk = np.zeros_like(pred, dtype=float)
    for cls, w in risk_weights.items():
        mask = pred == cls
        risk[mask] = w * conf[mask]
    risk[tiers >= 3] = 0.0   # Exclude low-confidence and cloud cells

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(risk, cmap='RdYlGn_r', vmin=0, vmax=1, interpolation='bicubic')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Nature-Related Risk Score\n(confidence x toxicity weight)', fontsize=9)
    ax.set_title("NSGIS Nature Risk Heatmap — TNFD LEAP Output\n"
                 "Agbogbloshie, Greater Accra | Real Sentinel-2 | Tier 1+2 only",
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Easting (10m cells)")
    ax.set_ylabel("Northing (10m cells)")

    yr, xr = np.where(risk > 0.75)
    if len(xr) > 0:
        ax.scatter(xr, yr, c='#FF0000', s=1.5, alpha=0.7, zorder=5,
                   label=f'High-risk clusters >0.75 (n={len(xr):,})')
        ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(str(VIZ_DIR / "fig4_risk_heatmap.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[VIZ] Fig 4: Risk heatmap saved")


def fig5_tnfd_summary(tnfd_report, eval_results):
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Panel A: Tier pie
    ax_a = fig.add_subplot(gs[0, 0])
    tiers_data = eval_results["neuro_symbolic"]["tier_distribution"]
    sizes  = [float(tiers_data.get(str(t), 0)) for t in [1,2,3,4]]
    labels = [f"Tier {t}" for t in [1,2,3,4]]
    colors = [TIER_COLORS[t] for t in [1,2,3,4]]
    ax_a.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
             startangle=90, pctdistance=0.75, textprops={'fontsize':8})
    ax_a.set_title("TNFD Tier Distribution\n(Real Data)", fontsize=10, fontweight='bold')

    # Panel B: Activity areas
    ax_b = fig.add_subplot(gs[0, 1])
    profiles = tnfd_report["tnfd_evaluate_phase"]["activity_profiles"]
    if profiles:
        names  = [n.replace(" ","\n") for n in profiles.keys()]
        areas  = [d["area_estimate_ha"]  for d in profiles.values()]
        confs  = [d["avg_confidence"]    for d in profiles.values()]
        colors_act = CLASS_COLORS_LIST[1:1+len(names)]
        bars   = ax_b.barh(names, areas, color=colors_act, edgecolor='white', alpha=0.9)
        for bar, conf in zip(bars, confs):
            ax_b.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                      f' {conf:.2f}', va='center', fontsize=7)
        ax_b.set_xlabel("Area (ha)")
        ax_b.set_title("Activity Areas (Tier 1+2)\nReal Sentinel-2", fontsize=10, fontweight='bold')
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)

    # Panel C: Water intensity
    ax_c = fig.add_subplot(gs[0, 2])
    wmap = {"Very High":5,"High":4,"Moderate":3,"Low":2,"Very Low":1}
    if profiles:
        wn  = [n.split("/")[0].strip() for n in profiles.keys()]
        wv  = [wmap.get(d["water_intensity"],1) for d in profiles.values()]
        wc  = ['#1565C0' if v>=4 else '#42A5F5' if v==3 else '#B3E5FC' for v in wv]
        ax_c.barh(wn, wv, color=wc, edgecolor='white')
        ax_c.set_xlim(0,6)
        ax_c.set_xticks([1,2,3,4,5])
        ax_c.set_xticklabels(['VLow','Low','Mod','High','VHigh'], fontsize=7)
        ax_c.set_title("Water Intensity\n(TNFD Dependency)", fontsize=10, fontweight='bold')
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)

    # Panel D: Performance
    ax_d = fig.add_subplot(gs[1, 0:2])
    ns   = eval_results["neuro_symbolic"]
    bl   = eval_results["baseline_neural"]
    mlabels = ["F1 Macro","F1 Weighted","Tier-1 F1","1-ECE"]
    ns_s    = [ns["f1_macro"],ns["f1_weighted"],ns["tier1_f1"],1-ns["ece"]]
    bl_s    = [bl["f1_macro"],bl["f1_weighted"],0,0]
    xi = np.arange(len(mlabels)); w2 = 0.35
    ax_d.bar(xi-w2/2, ns_s, w2, label='Neuro-Symbolic', color='#1B3A5C', alpha=0.9)
    ax_d.bar(xi+w2/2, bl_s, w2, label='Baseline Neural', color='#90A4AE', alpha=0.9)
    ax_d.set_xticks(xi); ax_d.set_xticklabels(mlabels, fontsize=9)
    ax_d.set_ylim(0, 1.1); ax_d.set_ylabel("Score")
    ax_d.set_title("Performance: Neuro-Symbolic vs Baseline", fontsize=10, fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.axhline(y=0.72, color='#E53935', linestyle='--', alpha=0.6, linewidth=1)
    ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)
    ax_d.yaxis.grid(True, alpha=0.3)

    # Panel E: Impact drivers
    ax_e = fig.add_subplot(gs[1, 2])
    drivers = tnfd_report["tnfd_evaluate_phase"]["aggregate_impact_drivers"]
    if drivers and profiles:
        dcounts = {d: sum(1 for p in profiles.values()
                          if d in p.get("impact_drivers",[]))
                   for d in drivers}
        sorted_d = sorted(dcounts.items(), key=lambda x: x[1], reverse=True)
        ax_e.barh([x[0][:25] for x in sorted_d],
                  [x[1] for x in sorted_d],
                  color='#E53935', alpha=0.75, edgecolor='white')
        ax_e.set_xlabel("Activity types")
        ax_e.set_title("Impact Drivers\n(TNFD Evaluate)", fontsize=10, fontweight='bold')
        ax_e.spines['top'].set_visible(False); ax_e.spines['right'].set_visible(False)

    fig.suptitle("NSGIS TNFD LEAP Output — Agbogbloshie, Greater Accra\n"
                 "Real Sentinel-2 L2A Data | 2026-02-01",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(str(VIZ_DIR / "fig5_tnfd_summary.png"),
                dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[VIZ] Fig 5: TNFD summary saved")


def fig6_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0,14); ax.set_ylim(0,6)
    ax.axis('off'); ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')

    def box(x,y,w,h,color,label,sub="",fs=9):
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=color,
                     edgecolor='white',linewidth=2,zorder=2,alpha=0.92))
        ax.text(x+w/2,y+h/2+(0.18 if sub else 0),label,ha='center',va='center',
                fontsize=fs,fontweight='bold',color='white',zorder=3)
        if sub:
            ax.text(x+w/2,y+h/2-0.25,sub,ha='center',va='center',
                    fontsize=7,color='#E0E0E0',zorder=3)
    def arr(x1,y1,x2,y2):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>",color='#555',lw=1.5),zorder=4)

    box(0.2,3.5,1.5,1.8,'#37474F',"Sentinel-2","Optical 10-20m")
    box(0.2,1.8,1.5,1.5,'#455A64',"ECOSTRESS","Thermal 70m")
    box(0.2,0.2,1.5,1.5,'#546E7A',"Sentinel-1","SAR 10m")
    box(2.2,1.5,1.6,2.5,'#1B3A5C',"Multi-Sensor\nFusion","Cross-attention")
    arr(1.7,4.4,2.2,3.0); arr(1.7,2.55,2.2,2.55); arr(1.7,0.95,2.2,2.1)
    box(4.2,1.5,1.8,2.5,'#0D6E6E',"Neural\nPerception","Clay backbone")
    arr(3.8,2.75,4.2,2.75)
    box(6.4,3.2,1.8,2.0,'#1565C0',"IAKG","47 archetypes")
    box(6.4,0.2,1.8,2.0,'#6A1B9A',"Symbolic\nReasoner","DeepProbLog")
    arr(5.0+1.0,2.75,6.4,1.2)
    ax.annotate("",xy=(7.3,3.2),xytext=(7.3,2.2),
                arrowprops=dict(arrowstyle="-|>",color='#555',lw=1.5))
    box(8.6,1.5,1.8,2.5,'#1B3A5C',"NS Fusion","Adaptive\nweighting")
    arr(8.2,2.75,8.6,2.75); arr(8.2,1.2,8.6,2.1)
    box(10.8,3.2,2.9,2.0,'#1B5E20',"TNFD Report","Tier 1-4 output")
    box(10.8,0.2,2.9,2.0,'#BF360C',"Risk Heatmap","10m resolution")
    arr(10.4,3.1,10.8,4.2); arr(10.4,2.4,10.8,1.2)

    ax.set_title("NSGIS System Architecture",fontsize=13,fontweight='bold',pad=12)
    plt.tight_layout()
    plt.savefig(str(VIZ_DIR / "fig6_architecture.png"),
                dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print("[VIZ] Fig 6: Architecture diagram saved")


def generate_all():
    print("\n[VIZ] Generating all figures...")
    grids, eval_results, tnfd_report, meta = load_outputs()
    fig1_activity_maps(grids, meta)
    fig2_tier_map(grids)
    fig3_performance_comparison(eval_results)
    fig4_risk_heatmap(grids)
    fig5_tnfd_summary(tnfd_report, eval_results)
    fig6_system_architecture()
    print(f"[VIZ] All figures saved to: {VIZ_DIR}")


if __name__ == "__main__":
    generate_all()
