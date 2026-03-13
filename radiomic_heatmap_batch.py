"""
Radiomic Heatmap Visualization for Prostate MRI

This script generates three-panel PNG visualizations for voxel-based PyRadiomics
features extracted from prostate T2-weighted MRI and corresponding gland masks.

Panels:
1. Anatomical T2W image with ROI contour
2. Heatmap-only view (zoomed around ROI)
3. Anatomical image with heatmap overlaid inside ROI only

Expected folder structure:
project_root/
├── t2w_files/
│   ├── 10001_t2w.nii.gz
│   ├── 10002_t2w.nii.gz
│   └── ...
├── gland_files/
│   ├── 10001_gland.nii.gz
│   ├── 10002_gland.nii.gz
│   └── ...
├── radiomic_heatmap_batch.py
├── requirements.txt
└── README.md
"""

from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy import ndimage
import six

from radiomics import featureextractor
import radiomics
import radiomics.base as rbase

# QUIET LOGS

logging.getLogger("radiomics").setLevel(logging.ERROR)


# PATCH: progress reporter bug in some PyRadiomics dev builds (voxelBased=True)

class _PatchedProgress:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _patched_getProgressReporter(*args, **kwargs):
    return _PatchedProgress(*args, **kwargs)


radiomics.getProgressReporter = _patched_getProgressReporter
rbase._DummyProgressReporter = _PatchedProgress


# CONFIG

BASE_DIR = Path(".")
T2_DIR = BASE_DIR / "t2w_files"
MASK_DIR = BASE_DIR / "gland_files"
OUT_DIR = BASE_DIR / "radiomic_heatmaps_clean_all"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# If True, save heatmaps for every voxel-based feature returned by PyRadiomics
GENERATE_ALL_FEATURES = True

# If GENERATE_ALL_FEATURES = False, only these features are saved
FEATURES_TO_VISUALIZE = [
    "original_firstorder_Mean",
    "original_firstorder_Entropy",
    "original_firstorder_Kurtosis",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_JointEnergy",
    "original_glrlm_RunEntropy",
]

# Panel 2 zoom padding: smaller = more zoom, larger = less zoom
PANEL2_PAD = 25

# Heatmap transparency in panel 3
ALPHA = 0.65


# HELPERS

def normalize_for_display(img2d: np.ndarray) -> np.ndarray:
    """Normalize a 2D image for display using robust percentile clipping."""
    vmin, vmax = np.percentile(img2d, [2, 98])
    img2d = np.clip(img2d, vmin, vmax)
    return (img2d - vmin) / (vmax - vmin + 1e-8)


def get_optimal_slice(mask_zyx: np.ndarray) -> int:
    """Return the slice index with the maximum ROI area."""
    areas = np.sum(mask_zyx > 0, axis=(1, 2))
    return int(np.argmax(areas))


def binarize_and_clean_mask(
    mask_img: sitk.Image,
    reference_img: sitk.Image,
    thr: float = 0.5
) -> sitk.Image:
    """
    Resample mask to the reference image grid, threshold to binary, and keep
    only the largest connected component.
    """
    mask_r = sitk.Resample(
        mask_img,
        reference_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    mask_r = sitk.Cast(mask_r, sitk.sitkFloat32)
    mask_bin = sitk.BinaryThreshold(mask_r, thr, 1e9, 1, 0)
    mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)

    cc = sitk.ConnectedComponent(mask_bin)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    if stats.GetNumberOfLabels() == 0:
        return mask_bin

    largest = max(stats.GetLabels(), key=lambda label: stats.GetPhysicalSize(label))
    cleaned = sitk.BinaryThreshold(cc, largest, largest, 1, 0)
    return sitk.Cast(cleaned, sitk.sitkUInt8)


def resample_feature_to_reference(
    feat_img: sitk.Image,
    reference_img: sitk.Image
) -> sitk.Image:
    """Resample a voxel feature map onto the full T2 image grid."""
    return sitk.Resample(
        feat_img,
        reference_img,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def robust_roi_vmin_vmax(
    feat2d: np.ndarray,
    roi2d: np.ndarray,
    p_low: float = 2,
    p_high: float = 98
) -> tuple[float, float]:
    """Compute robust min/max values from ROI pixels only."""
    vals = feat2d[roi2d]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0.0, 1.0

    vmin = np.percentile(vals, p_low)
    vmax = np.percentile(vals, p_high)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return float(vmin), float(vmax)


def bbox_from_mask(mask2d: np.ndarray, pad: int = 20) -> tuple[int, int, int, int]:
    """Return a padded bounding box around the 2D ROI mask."""
    ys, xs = np.where(mask2d > 0)

    if ys.size == 0:
        return 0, mask2d.shape[0], 0, mask2d.shape[1]

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    y1 = min(mask2d.shape[0], y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(mask2d.shape[1], x1 + pad)

    return y0, y1, x0, x1


def save_three_panel(
    anat2d: np.ndarray,
    feat2d_full: np.ndarray,
    roi2d: np.ndarray,
    feature_name: str,
    save_path: Path,
    alpha: float = 0.65,
    panel2_pad: int = 25
) -> None:
    """
    Save a three-panel figure:
    1. Full T2W image with ROI contour
    2. Heatmap-only view zoomed around ROI
    3. T2W image with ROI-only heatmap overlay
    """
    anat_norm = normalize_for_display(anat2d)
    roi = roi2d > 0

    contour = ndimage.binary_dilation(roi) ^ roi
    vmin, vmax = robust_roi_vmin_vmax(feat2d_full, roi, 2, 98)

    cmap = plt.cm.turbo.copy()
    cmap.set_bad(color=(0, 0, 0, 0))

    # Panel 3: ROI-only overlay
    masked_feat = np.ma.masked_where(~roi, feat2d_full)

    # Panel 2: crop around ROI for enlarged heatmap-only view
    y0, y1, x0, x1 = bbox_from_mask(roi2d, pad=panel2_pad)
    feat_crop = feat2d_full[y0:y1, x0:x1]
    roi_crop = roi[y0:y1, x0:x1]
    feat_crop_masked = np.ma.masked_where(~roi_crop, feat_crop)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1
    axes[0].imshow(anat_norm, cmap="gray")
    axes[0].contour(contour, colors="yellow", linewidths=2)
    axes[0].set_title("Anatomical Image (T2W)\nwith ROI", fontweight="bold")
    axes[0].axis("off")

    # Panel 2
    im2 = axes[1].imshow(
        feat_crop_masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[1].set_title(f"Heatmap Only\n{feature_name}", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3
    axes[2].imshow(anat_norm, cmap="gray")
    im3 = axes[2].imshow(
        masked_feat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        interpolation="nearest",
    )
    axes[2].contour(contour, colors="lime", linewidths=2)
    axes[2].set_title(f"Overlay (ROI Only)\n{feature_name}", fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# EXTRACTOR SETUP

def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    """Configure and return a PyRadiomics extractor."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["binWidth"] = 25
    extractor.settings["normalize"] = False
    extractor.settings["resampledPixelSpacing"] = None
    extractor.settings["interpolator"] = "sitkBSpline"
    extractor.settings["enableCExtensions"] = False

    extractor.enableImageTypeByName("Original")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")

    # Disable MCC because it can sometimes produce NaN / Inf eigenvalue errors
    try:
        extractor.disableFeatureByName("glcm", "MCC")
        print("Disabled GLCM:MCC to avoid numerical instability.")
    except Exception:
        print("Could not disable GLCM:MCC. Continuing anyway.")

    return extractor

# MAIN

def main() -> None:
    extractor = build_extractor()

    t2_files = sorted(T2_DIR.glob("*_t2w.nii.gz"))
    print(f"Found T2 files: {len(t2_files)}")

    ok_cases = 0
    fail_cases = 0

    for i, t2_path in enumerate(t2_files, 1):
        case_id = t2_path.name.replace("_t2w.nii.gz", "")
        mask_path = MASK_DIR / f"{case_id}_gland.nii.gz"

        print(f"\n[{i}/{len(t2_files)}] Case {case_id}")

        if not mask_path.exists():
            print(f"  Mask missing, skipping: {mask_path.name}")
            fail_cases += 1
            continue

        try:
            # Load images
            t2 = sitk.ReadImage(str(t2_path))
            mask = sitk.ReadImage(str(mask_path))

            # Clean mask on T2 grid
            mask_clean = binarize_and_clean_mask(mask, t2, thr=0.5)

            # Normalize T2 for radiomics extraction
            t2_norm = sitk.Normalize(t2)

            # Arrays and representative slice
            t2_arr = sitk.GetArrayFromImage(t2)
            mask_arr = sitk.GetArrayFromImage(mask_clean)

            z = get_optimal_slice(mask_arr)
            anat2d = t2_arr[z]
            roi2d = mask_arr[z]

            roi_vox = int(np.sum(mask_arr > 0))
            print(f"  ROI voxels: {roi_vox} | best slice: {z}")

            # Voxel-based extraction
            print("  Extracting voxel-wise feature maps...")
            feats_v = extractor.execute(t2_norm, mask_clean, label=1, voxelBased=True)
            img_feats = {k: v for k, v in six.iteritems(feats_v) if isinstance(v, sitk.Image)}
            print(f"  Voxel feature maps returned: {len(img_feats)}")

            # Feature list
            if GENERATE_ALL_FEATURES:
                feature_keys = sorted([k for k in img_feats.keys() if k.startswith("original_")])
            else:
                feature_keys = FEATURES_TO_VISUALIZE

            # Output folder per case
            case_out = OUT_DIR / case_id
            case_out.mkdir(parents=True, exist_ok=True)

            saved = 0
            for k in feature_keys:
                if k not in img_feats:
                    if k == "original_glcm_Energy" and "original_glcm_JointEnergy" in img_feats:
                        k_use = "original_glcm_JointEnergy"
                    else:
                        continue
                else:
                    k_use = k

                feat_full = resample_feature_to_reference(img_feats[k_use], t2)
                feat_vol = sitk.GetArrayFromImage(feat_full)

                zz = z if z < feat_vol.shape[0] else feat_vol.shape[0] // 2
                feat2d = feat_vol[zz]

                out_png = case_out / f"{k_use}_3panel.png"
                save_three_panel(
                    anat2d=anat2d,
                    feat2d_full=feat2d,
                    roi2d=roi2d,
                    feature_name=k_use.replace("original_", ""),
                    save_path=out_png,
                    alpha=ALPHA,
                    panel2_pad=PANEL2_PAD,
                )
                saved += 1

            print(f"  Saved {saved} three-panel PNGs to: {case_out}")
            ok_cases += 1

        except Exception as e:
            print(f"  ERROR: {repr(e)}")
            fail_cases += 1

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Successful cases: {ok_cases}")
    print(f"Failed/skipped cases: {fail_cases}")
    print(f"Output root: {OUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
