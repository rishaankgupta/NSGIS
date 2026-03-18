# NSGIS — Neuro-Symbolic Geospatial Inference System

**First TNFD-aligned nature risk prototype over Agbogbloshie, Greater Accra, Ghana**  
Real Sentinel-2 L2A satellite data · 10m resolution · 90,000 cells · February 2026

---

## What This Is

NSGIS is a working prototype that combines neural satellite image analysis 
with symbolic industrial ecology rules to identify informal industrial 
activity and generate TNFD LEAP-aligned nature risk disclosures — without 
requiring any pre-labeled ground truth data.

This is the first system to produce a 10-meter resolution nature risk 
heatmap over an informal industrial cluster (Agbogbloshie e-waste site, 
Greater Accra) from real satellite data.

## What It Produces

- Activity classification map (7 informal industrial classes: Tannery, 
  Textile Dyeing, E-Waste, Metal/Battery Recycling, Chemical Storage, 
  Food Processing, Non-Industrial)
- TNFD Confidence Tier Map (Tier 1–4, aligned with TNFD LEAP framework)
- Nature Risk Heatmap at 10m resolution
- JSON TNFD disclosure report with impact driver profiles

## Phase 1 Results (Real Sentinel-2 Data)

- Scene: 300×300 cells = 3km × 3km over Agbogbloshie, Accra
- Clear cells analyzed: 89,996 of 90,000 (100% after SCL cloud masking)
- Neuro-Symbolic F1 Macro: 0.485 vs Baseline Neural: 0.395 (+23%)
- Expected Calibration Error: 0.022 (target: <0.08)
- Tier 1 attribution: 20% of scene (17,972 cells)
- High-risk clusters flagged: 3,363 cells (risk score >0.75)

## Quick Start

### 1. Install dependencies
```
pip install rasterio scipy scikit-learn matplotlib numpy
```

### 2. Download real Sentinel-2 data
Go to: https://browser.dataspace.copernicus.eu  
Search for tile: **T30NZM**  
Product: **S2MSI2A** (Level-2A atmospherically corrected)  
Date: Any cloud-free scene over Accra  
Download the ~1GB .SAFE file

### 3. Update the data path
Open `data/real_data_loader.py`  
Change line 14 to your downloaded .SAFE folder path:
```python
SAFE_PATH = r"C:\your\path\to\file.SAFE\file.SAFE"
```

### 4. Run the pipeline
```
cd nsgis
python pipeline/run_pipeline_real.py
python viz/visualize.py
```

All outputs saved to `nsgis/outputs/`

## Repository Structure
```
nsgis/
├── data/
│   ├── sentinel_simulator.py    # Synthetic data generator
│   └── real_data_loader.py      # Real Sentinel-2 loader ← start here
├── iakg/
│   └── knowledge_graph.py       # Industrial Activity Knowledge Graph
│                                  (10 rules, 7 classes, 3-valued logic)
├── neural/
│   └── model.py                 # Neural module + NS fusion system
├── pipeline/
│   ├── run_pipeline.py          # Simulation pipeline
│   └── run_pipeline_real.py     # Real data pipeline ← run this
├── viz/
│   └── visualize.py             # All 6 publication figures
└── outputs/                     # Generated figures and reports
```

## System Architecture

Sentinel-2 + ECOSTRESS + Sentinel-1  
→ Multi-sensor fusion (cross-attention)  
→ Neural perception (Clay backbone)  
→ [IAKG symbolic rules] + [DeepProbLog reasoner]  
→ Neuro-symbolic fusion (adaptive weighting)  
→ TNFD Report + Risk Heatmap

## Papers

**Preprint 1 (Framework):**  
Gupta, R. (2025). Neuro-Symbolic Geospatial Intelligence: A Framework 
for Understanding Nature-Related Risks in the Informal Global South.  

**Preprint 2 (Full Architecture + Phase 1 Results):**  
Gupta, R. (2026). Operationalizing Neuro-Symbolic Geospatial Intelligence.  

## Phase 2 (In Progress)

- Real ECOSTRESS thermal data (NASA AppEEARS — pending)
- Real Sentinel-1 SAR data (Copernicus — pending)
- Ground truth field survey via ISODEC partnership (Ghana)
- Target journal: Nature or Remote Sensing of Environment

## License

MIT License — free to use, modify, and distribute with attribution.

## Contact

Rishaank Gupta · site.rishaank@gmail.com
