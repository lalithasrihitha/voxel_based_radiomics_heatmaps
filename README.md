# Radiomic Heatmap Visualization for Prostate MRI

This repository contains a Python workflow for generating **voxel-based radiomic heatmaps** from **prostate T2-weighted MRI** and corresponding **gland segmentation masks** using **PyRadiomics**.

For each case and each selected voxel-based feature map, the script saves a **three-panel PNG**:

1. **Anatomical Image (T2W) with ROI contour**
2. **Heatmap Only** (zoomed around the ROI)
3. **Overlay (ROI Only)** showing the heatmap inside the gland region on the anatomical image

This project is intended for **radiomics interpretation, visualization, and exploratory medical imaging analysis**.

---

# Project Purpose

Radiomics extracts quantitative information from medical images, such as:

- intensity-based characteristics  
- texture patterns  
- spatial heterogeneity  

While scalar radiomics features provide one value per lesion or region, **voxel-based radiomics** produces a full feature map across the image volume. These maps help visualize **where** specific texture or intensity patterns are located inside the gland.

This repository focuses on making those voxel-wise radiomic features easier to inspect visually.

---

# What This Script Does

The script:

- loads **T2-weighted MRI volumes** (`*_t2w.nii.gz`)
- loads matching **gland segmentation masks** (`*_gland.nii.gz`)
- resamples and cleans masks
- extracts **voxel-based radiomic feature maps** using PyRadiomics
- identifies the slice with the largest gland area
- creates three-panel visualizations for each feature
- saves output PNGs in patient-specific folders

---

# Visualization Panels

Each generated image contains the following panels:

### Panel 1 — Anatomical Image (T2W) with ROI
- displays the selected T2-weighted slice
- shows the gland segmentation contour
- helps visually locate the region being analyzed

### Panel 2 — Heatmap Only
- displays the radiomic feature map
- zooms around the ROI region
- highlights spatial variation of the feature

### Panel 3 — Overlay (ROI Only)
- shows the full anatomical T2 image
- overlays the radiomic heatmap only inside the ROI
- provides anatomical context for the radiomic feature

---

# Expected Folder Structure

Place your files like this:

```
project_root/
│
├── t2w_files/
│   ├── 10001_t2w.nii.gz
│   ├── 10002_t2w.nii.gz
│   └── ...
│
├── gland_files/
│   ├── 10001_gland.nii.gz
│   ├── 10002_gland.nii.gz
│   └── ...
│
├── radiomic_heatmap_batch.py
├── requirements.txt
└── README.md
```

---

# Input File Naming Convention

The script expects image and mask pairs with the following naming format:

T2 image  
```
CASEID_t2w.nii.gz
```

Mask  
```
CASEID_gland.nii.gz
```

Example:

```
10001_t2w.nii.gz
10001_gland.nii.gz
```

The script automatically matches files using the shared **case ID**.

---

# Output Structure

The script creates an output folder:

```
radiomic_heatmaps_clean_all/
```

Inside it, each case will have its own folder:

```
radiomic_heatmaps_clean_all/
│
├── 10001/
│   ├── original_firstorder_Mean_3panel.png
│   ├── original_firstorder_Entropy_3panel.png
│   └── ...
│
├── 10002/
│   ├── original_firstorder_Mean_3panel.png
│   └── ...
```

Each image corresponds to one **voxel-based radiomic feature map**.

---

# Installation

Clone the repository:

```
git clone https://github.com/your-username/prostate-radiomic-heatmaps.git
cd prostate-radiomic-heatmaps
```

(Optional) create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Requirements

This repository requires the following Python packages:

- Python 3.10+
- numpy
- matplotlib
- SimpleITK
- scipy
- six
- pyradiomics

Install them using:

```
pip install -r requirements.txt
```

---

# How to Run

Run the script from the project root:

```
python radiomic_heatmap_batch.py
```

The script will automatically:

1. scan the **t2w_files/** folder  
2. match masks in **gland_files/**  
3. extract voxel-based radiomic feature maps  
4. generate three-panel heatmap visualizations  
5. save them to **radiomic_heatmaps_clean_all/**

---

# Main Configuration Options

These parameters can be adjusted in the script.

## Generate all radiomic features

```
GENERATE_ALL_FEATURES = True
```

If set to **False**, only selected features will be visualized.

---

## Feature subset

```
FEATURES_TO_VISUALIZE = [
    "original_firstorder_Mean",
    "original_firstorder_Entropy",
    "original_firstorder_Kurtosis",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_JointEnergy",
    "original_glrlm_RunEntropy"
]
```

---

## Heatmap zoom level

```
PANEL2_PAD = 25
```

Smaller values produce **tighter zoom** around the ROI.

---

## Overlay transparency

```
ALPHA = 0.65
```

Controls transparency of the heatmap overlay in Panel 3.

---

# Method Details

## Mask Preprocessing

The segmentation mask undergoes:

1. resampling to the T2 image grid
2. thresholding to produce a binary mask
3. largest connected component selection

This removes noise and small disconnected regions.

---

## Slice Selection

The slice containing the **largest ROI area** is selected as the representative visualization slice.

---

## Radiomics Extraction

Voxel-based radiomics extraction is performed with **PyRadiomics**.

Key parameters:

```
binWidth = 25
normalize = False
resampledPixelSpacing = None
interpolator = sitkBSpline
enableCExtensions = False
```

Feature classes enabled:

- First Order
- GLCM
- GLRLM
- GLSZM
- GLDM

---

## Numerical Stability

The radiomic feature **GLCM:MCC** is disabled because it can sometimes produce **NaN or infinite eigenvalues**, which can cause extraction failures.

---

## Progress Reporter Patch

Some PyRadiomics development builds contain a progress reporter bug when running `voxelBased=True`.

A small patch is included in this script to bypass that issue and ensure stable execution.

---

# Why Relative Paths Are Used

This repository uses:

```
BASE_DIR = Path(".")
```

instead of Google Drive paths or Colab mounting.

This makes the project:

- portable
- easier to run locally
- cleaner for GitHub
- reproducible across systems

Simply place the `t2w_files/` and `gland_files/` folders in the repository root.

---

# Example Applications

This repository can be used for:

- radiomics visualization
- prostate MRI feature analysis
- imaging biomarker exploration
- explainable radiomics research
- scientific presentations and publications
