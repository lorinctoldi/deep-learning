**Team** - Sigmoid Squad

**Members**
- Armand Szokoly (VN450W)
- Balázs Kiss (NXG36N)
- Lőrinc Toldi (DORAU5)

# Project Description

This project focuses on **ship detection and segmentation in satellite imagery**, based on the **Container Ship Dataset** from the [Airbus Ship Detection Challenge on Kaggle](https://www.kaggle.com/c/airbus-ship-detection).
The dataset is designed for maritime object detection and segmentation tasks.

## Dataset Overview
- Source: Kaggle, Airbus Ship Detection Challenge
- Size of dataset: ~32 Gb
- Number of images: ~193,000
- Image resolution: 768 × 768 pixels (RGB)
- Annotations: Run-Length Encoded (RLE) masks for each ship
- Empty images: Some images contain no ships
- Multiple ships per image: Many images have more than one ship, each represented by a separate RLE (one-to-many)

The main annotation file, `segmentations.csv`, contains two columns:
`ImageId` (the image filename) and `EncodedPixels` (the RLE mask).
Images with multiple ships appear multiple times in the CSV, while an empty `EncodedPixels` field indicates an image without ships.

---

# Repository Structure and File Functions

## Main Folders and Files
- `data/images/` – Contains the raw satellite images.
- `data/original_segmentations.csv` – The original, unprocessed annotation file used for initial data exploration.
- `data/segmentations.csv` – The cleaned version of the annotation file (invalid or overlapping masks removed).
- `cache/` – Stores cached or precomputed results, such as analyzed masks or intermediate data.
- `assets/` – Contains example figures and visualization outputs used in notebooks or reports.
- `utils.py` - Contains utility functions used in the notebook.
- `constants.py` - Contains constants values used in the notebook.
- `main.ipynb` - Main file containing all logic, analysis, and description.
- `tests/` - Contains tests for the utility functions.
- `requirements.txt` - Lists all required Python packages.

> **Important Note:**
> The `data/` folder is initially empty.
> The dataset download and setup methods are included in the `main.ipynb` notebook.

## Excluded Files
The original Kaggle dataset also includes:
- `test_v2/` – Test images without labels.
- `sample_submission_v2.csv` – Template for Kaggle submissions.

These files were **excluded** from the repository because they contain no annotations and would require manual validation.

# Validation and Data Cleaning

## Validation

### Validation Criteria
1. Each RLE mask should correspond to a **single boat**, treated as a closed object. Every mask is expected to contain **exactly one connected shape** of reasonable size.
2. **Masks should not overlap**, as two boats cannot occupy the same physical space in the image.

### Methodology
1. Read in the `original_segmentations.csv` file.
2. Convert each RLE mask into a 2D binary mask.
3. For images with an RLE annotation, count the number of connected shapes in the mask and measure their sizes in pixels.
4. If a mask contains more than one shape or no shape reaches at least 12 pixels, mark it as problematic.
5. Visualize problematic images by overlaying the mask on the original image.
6. Check for overlapping shapes across all masks of the same image and flag any overlaps.

---

## Data Cleaning

### Methodology
- **Remove masks with no valid shapes:** If a mask contains no shapes larger than 12 pixels, discard the mask entirely.
- **Keep only the largest shape:** For masks with multiple shapes, remove all shapes except the largest one to ensure each mask corresponds to a single boat.

---

# Train, Validation, and Test Split

### Methodology
The dataset is randomly partitioned into the following subsets:
- **80%** training
- **10%** validation
- **10%** testing

- The data order is randomized prior to splitting to ensure an unbiased distribution.
- Although initially considered, adding a column to `segmentations.csv` for subset assignment was deemed unnecessary, as splitting is handled programmatically using the **scikit-learn** library.

# How to Run

## 1. Install Dependencies
Install all required Python packages listed in `requirements.txt`.
