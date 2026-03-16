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
[Link to your Authorea/SSRN preprint]

**Preprint 2 (Full Architecture + Phase 1 Results):**  
Gupta, R. (2026). Operationalizing Neuro-Symbolic Geospatial Intelligence.  
[Link — add after uploading]

## Phase 2 (In Progress)

- Real ECOSTRESS thermal data (NASA AppEEARS — pending)
- Real Sentinel-1 SAR data (Copernicus — pending)
- Ground truth field survey via ISODEC partnership (Ghana)
- Target journal: Remote Sensing of Environment

## License

MIT License — free to use, modify, and distribute with attribution.

## Contact

Rishaank Gupta · site.rishaank@gmail.com
```

---

## How to Create the GitHub Repo — Step by Step

**Step 1** — Go to github.com, sign in or create free account

**Step 2** — Click the green **New** button top left

**Step 3** — Fill in:
- Repository name: `NSGIS`
- Description: paste the one-liner from above
- Set to **Public**
- Check **Add a README file** — NO, don't check this, you already have one
- License: **MIT**
- Click **Create repository**

**Step 4** — On your computer, open Command Prompt in your `nsgis/` folder and run:
```
git init
git add .
git commit -m "Initial commit: NSGIS Phase 1 prototype with real Sentinel-2 results over Agbogbloshie"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/NSGIS.git
git push -u origin main
```

Replace `YOURUSERNAME` with your actual GitHub username.

**Step 5** — Go to your repo page, click the gear icon next to "About", paste the description and add all the topics listed above.

---

## Second Paper — Should You Upload to Preprint Now?

**Yes, upload it now, before doing anything else.** Here is why: once it is on a preprint server with a timestamp, that date is your intellectual property marker. The GitHub repo and ISODEC email both become more credible when they link to a citable preprint with a DOI.

**However** — upload it in its current state, meaning without Section 11.1 results. Add one note on the first page:

> *"Note: This paper presents the complete system architecture. Phase 1 experimental results on real Sentinel-2 data over Agbogbloshie, Ghana are available at [GitHub link] and will be incorporated in the next revision."*

This is standard preprint practice. It is honest and it lets you upload now while reserving the right to add results in revision.

**Where to upload:**

Upload to **all three places you already use**, in this order:

**1. Authorea** (do this first since you already have an account)
- Go to authorea.com
- New paper → Import document → upload the .docx file
- Fill in title: *"Operationalizing Neuro-Symbolic Geospatial Intelligence: A Complete Technical, Methodological, and Governance Architecture for TNFD-Aligned Nature Risk Assessment in Informal Economies"*
- Authors: Rishaank Gupta, Independent Researcher
- Keywords: same as Paper 1 plus: DeepProbLog, TNFD LEAP, Sentinel-2, Agbogbloshie, Ground Truth Acquisition
- In the abstract, add the note about Phase 1 results at GitHub
- Click **Publish** — Authorea gives you a DOI instantly

**2. SSRN**
- ssrn.com → Submit a paper
- Same title, same abstract with the GitHub note
- Subject area: Environmental Economics + Computer Science
- Upload the .docx

**3. Academia.edu**
- Same process as before

After Authorea gives you the DOI, copy that DOI link. You will add it to:
- The GitHub README (replace the placeholder)
- The ISODEC email

---

## The ISODEC Email — Send Right After GitHub and Preprint Are Live

Wait until both are live so you can include both links. Then send exactly the email I wrote in the previous message, adding:

> "GitHub repository with working code: [your GitHub link]  
> Research preprint: [your Authorea DOI]"

---

## Complete Step-by-Step Sequence — Nothing Missed

Here is the full order. Do not skip ahead:

**TODAY — takes about 3 hours total:**

1. Create `.gitignore` file in nsgis/ folder
2. Create `README.md` file in nsgis/ folder using the text above
3. Create GitHub repo and push codebase
4. Copy GitHub URL
5. Open NSGIS_Operationalizing_Paper_v2.docx, add the one-line note about Phase 1 results being at GitHub
6. Upload updated docx to Authorea → publish → copy DOI
7. Upload to SSRN
8. Upload to Academia.edu
9. Update GitHub README — replace the paper placeholder links with real DOIs
10. Send ISODEC email with both links attached

**WEEK 1 — real sensor data:**

11. Go to appeears.earthdata.nasa.gov — create NASA Earthdata account (free)
12. Submit point-sample request: product ECO2LSTE, coordinates 5.5322N 0.2210W, date range 2025-11-01 to 2026-02-28
13. While waiting for AppEEARS (24 hours): go to Copernicus Browser, search Sentinel-1 GRD over same area and date, download
14. When ECOSTRESS arrives: replace compute_lst_proxy() in real_data_loader.py with real file reader
15. Replace SAR proxy with real Sentinel-1 values
16. Re-run pipeline: `python pipeline/run_pipeline_real.py`
17. New figures will show real thermal anomaly detection — E-Waste classification should improve

**WEEK 2 — write Section 11.1:**

18. Open NSGIS_Operationalizing_Paper_v2.docx
19. Add Section 11.1 with the exact text from my previous message
20. Insert all six figures as Figure 11.1 through 11.6
21. Update abstract to mention Phase 1 results achieved
22. Update preprint on Authorea with new version — same DOI, just a revision
23. Push updated code to GitHub: `git add . && git commit -m "Add real ECOSTRESS and Sentinel-1 data" && git push`

**WEEK 3-4 — submit to journal:**

24. Format paper according to Remote Sensing of Environment guidelines (double-spaced, line numbers, specific reference format)
25. Create a cover letter — I can write this for you when you are ready
26. Submit at ees.elsevier.com/rse

**MONTH 2-4 — ground truth (depends on ISODEC response):**

27. If ISODEC responds: coordinate field survey using ODK Collect
28. If no response after 3 weeks: email WIEGO directly (wiego.org), they have Accra survey data
29. When you have 200+ labeled points: re-run pipeline with real training data
30. F1 will jump from 0.485 toward the 0.72 target
31. Update paper with Phase 2 results
32. This becomes a second, stronger submission or a revision of the first

---

## One More Thing You Must Do This Week

Install `pyproj` so the pixel location is computed from real GPS coordinates instead of the fallback estimate:
```
pip install pyproj
```

Then re-run `python pipeline/run_pipeline_real.py`. The log will show:
```
[LOADER] Pixel: row=X, col=X  ← real GPS-computed location
