# Dataset Creation Guide

Complete documentation for building the Vesaki-VTON dataset with high-quality annotations.

## Overview

The Vesaki-VTON dataset consists of person images and garment images with comprehensive annotations required for virtual try-on training. This guide covers the entire dataset creation pipeline from web scraping to final validation.

## Quick Start

Build the complete dataset with one command:

```bash
python3 main_advanced.py
```

Expected time: 2-4 hours (GPU) or 4-8 hours (CPU only)

## Dataset Structure

```
dataset/
├── train/
│   ├── person/              # Person images (768x1024)
│   ├── garment/             # Garment images (768x1024)
│   ├── person-parse/        # SCHP segmentation masks
│   ├── densepose/           # Body segmentation masks
│   ├── openpose/            # Pose keypoints (JSON + visualization)
│   ├── agnostic-mask/       # Masked person images
│   ├── pairs.txt            # Person-garment training pairs
│   ├── train_list.txt       # Training image list
│   ├── quality_scores.json  # Per-image quality metrics
│   └── dataset_info.json    # Dataset statistics
└── test/
    └── [same structure as train]
```

## Pipeline Components

### 1. Data Collection

**Script**: `scripts/scraper_improved.py`

**Purpose**: Scrape fashion product images from Google Shopping

**Features**:
- Undetected Chrome driver to avoid blocking
- Concurrent async downloads via aiohttp
- Automatic classification using Haar Cascade face detection
- Multi-pass detection for improved recall
- Rate limiting and error recovery

**Configuration**:
- URLs: Edit `product_urls.txt`
- Images per URL: 100 (default)
- Concurrent downloads: 10

**Input**: Google Shopping search URLs in `product_urls.txt`

**Output**: Classified images in `dataset/train/person/` and `dataset/train/garment/`

**Usage**:
```bash
python3 scripts/scraper_improved.py
```

**Optimizing Results**:
- Use "woman wearing dress" or "model wearing top" keywords
- These yield more person images than generic product searches
- Garment images come from product-only listings

### 2. Data Augmentation

**Script**: `scripts/augment_images.py`

**Purpose**: Expand dataset through systematic transformations

**Augmentation Techniques**:

**Geometric Transformations**:
- Horizontal flip (mirror image)
- Rotation: ±5 degrees (preserves body structure)
- Crop variations: 95-100% with recentering

**Photometric Transformations**:
- Brightness: ±20% adjustment
- Contrast: ±15% adjustment
- Color jitter: Hue/saturation variations

**Library**: Albumentations for professional-grade augmentation

**Configuration**:
```python
# In augment_images.py
AUGMENTATION_MULTIPLIER = 5-8  # Generate 5-8 variations per image
BRIGHTNESS_RANGE = 0.2
CONTRAST_RANGE = 0.15
ROTATION_LIMIT = 5
```

**Input**: Original person images from `dataset/train/person/`

**Output**: 
- Augmented images with suffixes (_aug1.jpg, _aug2.jpg, etc.)
- Metadata tracking in augmentation_map.json

**Usage**:
```bash
python3 scripts/augment_images.py --data_dir dataset/train
```

**Notes**:
- Only person images are augmented (garments stay original)
- Preserves aspect ratio
- Applied AFTER scraping, BEFORE annotation
- Tracked metadata for reproducibility

### 3. Image Preprocessing

**Script**: `scripts/preprocess.py`

**Purpose**: Standardize image dimensions and format

**Operations**:
1. Resize to 768x1024 with aspect ratio preservation
2. Center-crop or pad to target dimensions (white background)
3. Format conversion to RGB JPEG
4. Quality validation

**Configuration**:
```python
TARGET_WIDTH = 768
TARGET_HEIGHT = 1024
BACKGROUND_COLOR = (255, 255, 255)  # White
```

**Usage**:
```bash
python3 scripts/preprocess.py --data_dir dataset/train --resize
```

### 4. SCHP Parsing

**Script**: `scripts/generate_parse_schp.py`

**Purpose**: Generate semantic segmentation of human body parts

**Model**: Self-Correction Human Parsing (SCHP)
- Pre-trained on ATR and LIP datasets
- 18-class segmentation
- 90%+ mIoU accuracy

**Segmentation Classes** (0-17):
0. Background
1. Hat
2. Hair
3. Sunglasses
4. Upper-clothes
5. Skirt
6. Pants
7. Dress
8. Belt
9. Left-shoe
10. Right-shoe
11. Face
12. Left-leg
13. Right-leg
14. Left-arm
15. Right-arm
16. Bag
17. Scarf

**Output**:
- Grayscale label maps: `*_person.png` (0-17 integer labels)
- RGB visualization overlays: `*_person_vis.jpg`
- Per-class confidence scores (optional)

**Usage**:
```bash
python3 scripts/generate_parse_schp.py --data_dir dataset/train
```

**Fallback**: 
- Rule-based CV segmentation if SCHP unavailable
- Lower accuracy (~60% vs 90%)

**Performance**:
- Processing time: ~2-3 seconds per image (GPU)
- Batch processing: 10-20 images per batch
- Memory: ~2GB GPU VRAM

### 5. Pose Detection

**Script**: `scripts/generate_openpose_holistic.py`

**Purpose**: Detect human pose keypoints for spatial alignment

**Model**: MediaPipe Holistic
- 33 body landmarks (Pose)
- 21 hand landmarks per hand
- 468 face mesh points (optional, not currently used)

**Keypoint Mapping**:
- MediaPipe 33 landmarks → OpenPose 25-point format
- Body: 18 points (nose, neck, shoulders, elbows, wrists, hips, knees, ankles, eyes, ears)
- Hands: 7 points (2 wrist + 5 key finger joints per hand, simplified to 7 total)

**Output**:
- JSON keypoints: `*_person_keypoints.json`
- Format: OpenPose compatible
- Skeleton visualization: `*_person.jpg`
- Per-keypoint confidence scores

**JSON Format**:
```json
{
  "people": [{
    "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...],
    "hand_left_keypoints_2d": [...],
    "hand_right_keypoints_2d": [...]
  }],
  "canvas_height": 1024,
  "canvas_width": 768
}
```

**Usage**:
```bash
python3 scripts/generate_openpose_holistic.py --data_dir dataset/train
```

**Limitations**:
- 10-20% failure rate expected (partial crops, unusual angles, occlusions)
- Failed images still have other annotations and can be used

**Performance**:
- Processing time: ~0.5-1 second per image (CPU)
- MediaPipe is optimized for CPU inference
- No GPU required

### 6. Body Segmentation

**Script**: `scripts/generate_densepose_advanced.py`

**Purpose**: Generate person/background segmentation masks

**Model**: MediaPipe Selfie Segmentation (Model 1)

**Features**:
- High-quality person masks
- Morphological refinement (closing, opening)
- Edge smoothing with Gaussian blur
- Multi-resolution support

**Processing Steps**:
1. Run MediaPipe segmentation
2. Threshold to binary mask (threshold: 0.5)
3. Morphological closing (kernel: 15x15)
4. Morphological opening (kernel: 9x9)
5. Gaussian blur for smooth edges (kernel: 5x5)

**Output**:
- Binary masks: `*_person.png` (0=background, 255=person)
- Visualization: `*_person.jpg` (mask overlay)

**Usage**:
```bash
python3 scripts/generate_densepose_advanced.py --data_dir dataset/train
```

**Fallback**: 
- Traditional background subtraction if MediaPipe unavailable
- GrabCut algorithm for refinement

**Performance**:
- Processing time: ~0.5 seconds per image (CPU)
- Accurate for most poses
- Handles complex backgrounds

### 7. Agnostic Mask Generation

**Script**: `scripts/generate_agnostic_advanced.py`

**Purpose**: Create masked person images indicating try-on regions

**Method**:
1. Load SCHP segmentation to identify clothing labels
2. Mask upper-clothes (4), skirt (5), pants (6), dress (7), belt (8)
3. Apply 15px dilation for coverage
4. Gaussian blur (31x31, sigma=10) for smooth transitions
5. Fill masked regions with gray (128,128,128)

**Why Gray (128,128,128)?**
- Neutral value between black (0) and white (255)
- Model learns to "fill in" this region with warped garment
- Smooth transitions prevent hard boundaries

**Output**:
- Grayscale-region person images: `*_person.jpg`
- PNG masks: `*_person.png` (binary mask of clothing regions)

**Usage**:
```bash
python3 scripts/generate_agnostic_advanced.py --data_dir dataset/train
```

**Configuration**:
```python
CLOTHING_LABELS = [4, 5, 6, 7, 8]  # Labels to mask
DILATION_KERNEL = 15  # Expand mask size
BLUR_KERNEL = 31
BLUR_SIGMA = 10
GRAY_COLOR = (128, 128, 128)
```

**Performance**:
- Processing time: ~0.2 seconds per image
- Requires SCHP parsing to be generated first

### 8. Quality Filtering

**Script**: `scripts/filter_quality.py`

**Purpose**: Automatically remove low-quality images

**Quality Criteria** (4 checks):

**1. Sharpness (Laplacian Variance)**
- Threshold: > 100
- Measures image blur
- Rejects out-of-focus images

**2. Brightness (Mean Intensity)**
- Range: [50, 200]
- Rejects too dark or overexposed images
- Calculated on grayscale conversion

**3. Face Detection Confidence**
- Threshold: > 0.3
- Uses Haar Cascade detector
- Multiple passes with different scales

**4. Resolution Check**
- Minimum dimension: 300px (before resize)
- Prevents extreme distortion

**Scoring**:
- Image must pass 3/4 criteria to be kept
- Score recorded in quality_assessment.json

**Operation Modes**:
```bash
# Dry-run (preview without removing)
python3 scripts/filter_quality.py --data_dir dataset/train --dry_run

# Actually remove low-quality images
python3 scripts/filter_quality.py --data_dir dataset/train
```

**Output**:
- quality_assessment.json: Per-image quality scores
- Removes failed images and ALL corresponding annotations
- Summary statistics

**Typical Results**:
- High quality (4/4): 25-30%
- Good quality (3/4): 50-60%
- Rejected (< 3/4): 10-20%

### 9. Dataset Validation

**Script**: `scripts/validate_advanced.py`

**Purpose**: Verify dataset completeness and correctness

**Validation Checks**:
1. All required directories exist
2. Image counts per component match
3. All images are 768x1024 resolution
4. All person images have corresponding annotations
5. JSON keypoint files are valid
6. Augmentation mapping is consistent
7. No corrupted image files

**Output**:
```
Dataset Validation Report
==================================================
✓ Person images: 243
✓ Garment images: 531
✓ SCHP parsing: 243/243 (100%)
✓ Pose keypoints: 208/243 (85%)
⚠ Body masks: 243/243 (100%)
✓ Agnostic masks: 243/243 (100%)
✓ All images are 768x1024
==================================================
Status: PASSED (with warnings)
```

**Usage**:
```bash
python3 scripts/validate_advanced.py --data_dir dataset/train
```

### 10. Metadata Export

**Script**: `scripts/export_metadata.py`

**Purpose**: Generate training metadata files

**Generated Files**:

**1. pairs.txt**
- Person-garment pairings for training
- Random sampling: 3-5 garments per person
- Format: `person_image.jpg garment_image.jpg` (one pair per line)

**2. train_list.txt**
- List of all person image filenames
- One per line
- Used by data loader

**3. quality_scores.json**
- Per-image quality metrics from filter_quality.py
- Includes sharpness, brightness, face confidence, resolution scores
- Format: 
```json
{
  "image_001.jpg": {
    "sharpness": 156.3,
    "brightness": 128.5,
    "face_confidence": 0.87,
    "resolution_ok": true,
    "passed": true
  }
}
```

**4. dataset_info.json**
- Overall dataset statistics
- Total counts, augmentation info, annotation coverage

**Usage**:
```bash
python3 scripts/export_metadata.py --data_dir dataset/train
```

## Complete Pipeline

The `main_advanced.py` script orchestrates all steps:

```python
# Pipeline execution order:
1. Check dependencies
2. Download model weights (if needed)
3. Scrape images from Google Shopping
4. Pre-filter quality
5. Augment person images (5-8x)
6. Resize all images to 768x1024
7. Generate SCHP parsing
8. Generate pose keypoints
9. Generate body masks
10. Generate agnostic masks
11. Post-filter quality
12. Validate dataset
13. Export metadata
```

**Run the complete pipeline**:
```bash
python3 main_advanced.py
```

**Expected Results**:
- 500-800+ person images (after augmentation)
- 500-1000+ garment images
- 85-95% annotation accuracy
- Complete training metadata

## Customization

### Adding Custom URLs

Edit `product_urls.txt`:

```
# Use "woman wearing" or "model wearing" for person images
https://www.google.com/search?q=woman+wearing+dress&tbm=shop&udm=28
https://www.google.com/search?q=model+wearing+shirt&tbm=shop&udm=28

# Generic product searches yield more garment images
https://www.google.com/search?q=women+dress&tbm=shop&udm=28
```

### Adjusting Augmentation

Edit `scripts/augment_images.py`:

```python
# Increase augmentation multiplier
AUGMENTATION_MULTIPLIER = 10  # Generate 10 variations

# Adjust transformation ranges
BRIGHTNESS_RANGE = 0.3  # ±30% instead of ±20%
ROTATION_LIMIT = 10  # ±10 degrees instead of ±5
```

### Adjusting Quality Thresholds

Edit `scripts/filter_quality.py`:

```python
# Make filtering more lenient
SHARPNESS_THRESHOLD = 50  # Lower threshold
BRIGHTNESS_RANGE = (30, 220)  # Wider range
FACE_CONFIDENCE = 0.2  # Lower confidence requirement
MIN_REQUIRED_PASSES = 2  # Only need 2/4 criteria
```

### Adjusting Image Resolution

Edit `scripts/preprocess.py`:

```python
# Change target resolution
TARGET_WIDTH = 512   # Instead of 768
TARGET_HEIGHT = 768  # Instead of 1024
```

Note: Model training configuration must match dataset resolution.

## Dataset Statistics

### Current Dataset (as of last build)

- **Person Images**: 243
- **Garment Images**: 531
- **Training Pairs**: 972
- **Annotation Coverage**:
  - SCHP Parsing: 243/243 (100%)
  - Body Segmentation: 243/243 (100%)
  - Agnostic Masks: 243/243 (100%)
  - Pose Keypoints: 208/243 (85%)

### Quality Distribution

- **High Quality** (4/4 criteria): 63 images (26%)
- **Good Quality** (3/4 criteria): ~120 images (49%)
- **Acceptable Quality** (baseline): ~60 images (25%)

### Augmentation Potential

- Current: 243 person images
- After 5x augmentation: 1,215 person images
- After 8x augmentation: 1,944 person images

## Troubleshooting

### Low Person Image Count

**Problem**: Scraper yields mostly garment images, few person images

**Solutions**:
1. Update `product_urls.txt` with "woman wearing" or "model wearing" keywords
2. Run scraper multiple times
3. Lower face detection threshold in `scraper_improved.py`
4. Use augmentation to expand dataset (5-8x)

### High Quality Filter Rejection Rate

**Problem**: Too many images filtered out by quality checks

**Solutions**:
1. Use `--dry_run` to see which criteria are failing
2. Lower thresholds in `scripts/filter_quality.py`
3. Review `quality_assessment.json` for failure patterns
4. Adjust face detection sensitivity
5. Reduce minimum required passes to 2/4

### Missing Pose Keypoints

**Problem**: 10-20% of images lack pose detection

**Expected**: This is normal for challenging poses, partial crops, or unusual angles

**Solutions**:
1. Images still usable with other annotations
2. Model can handle some missing keypoints
3. For critical images, consider manual annotation
4. Filter dataset to only include images with complete keypoints (optional)

### Out of Memory Errors

**Problem**: GPU memory exhausted during annotation generation

**Solutions**:
1. Reduce batch size in SCHP parsing
2. Process in smaller chunks
3. Use CPU instead of GPU for annotation
4. Close other applications
5. Upgrade GPU or use cloud GPUs

### Slow Processing

**Problem**: Dataset building takes too long

**Optimizations**:
1. Use GPU for SCHP parsing (10x faster)
2. Enable batch processing where available
3. Skip augmentation initially (add later)
4. Process test set separately or skip
5. Use multi-threading for I/O operations

### Corrupted Images

**Problem**: Some images fail to load or process

**Solutions**:
1. Validation automatically detects corrupted files
2. Re-download corrupted images
3. Remove corrupted images with `filter_quality.py`
4. Check disk space and permissions

## Best Practices

### For High-Quality Dataset

1. **Use good source URLs**: Fashion model shots > product listings
2. **Run multiple scraping sessions**: Different times yield different results
3. **Apply augmentation**: Expand dataset 5-8x for better generalization
4. **Filter aggressively**: Quality over quantity
5. **Validate thoroughly**: Run validation after each major step
6. **Back up raw data**: Keep original scraped images before processing

### For Faster Development

1. **Start with small dataset**: 50 images for testing pipeline
2. **Use pre-built dataset**: Download pre-annotated data if available
3. **Skip test set initially**: Focus on train set
4. **Cache annotations**: Avoid regenerating if not needed
5. **Use CPU for annotation**: GPU not necessary for dataset building

### For Production

1. **Larger dataset**: 1000+ person images minimum
2. **Multiple domains**: Scrape from various fashion websites
3. **Manual review**: QA check random samples
4. **Version control**: Track dataset versions
5. **Documentation**: Document any custom modifications

## Advanced Topics

### Adding New Annotation Types

To add custom annotations (e.g., garment attributes, style labels):

1. Create new script in `scripts/`
2. Follow existing naming convention: `generate_*_*.py`
3. Output to new subdirectory in `dataset/train/`
4. Update validation script to check new annotations
5. Update metadata export if needed

### Custom Scraping Sources

To scrape from other websites:

1. Modify `scripts/scraper_improved.py`
2. Add URL patterns for new source
3. Adjust HTML parsing for that site's structure
4. Keep face detection classification logic
5. Test on small sample first

### Distributed Processing

For very large datasets:

1. Split URL list across multiple machines
2. Run scraping in parallel
3. Merge scraped images
4. Use array jobs for annotation generation
5. Aggregate metadata files

## References

### Tools Used

- **SCHP**: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
- **MediaPipe**: https://google.github.io/mediapipe
- **Albumentations**: https://albumentations.ai
- **OpenCV**: https://opencv.org

### Datasets Referenced

- **ATR (Apparel Transfer Recognition)**: Human parsing benchmark
- **LIP (Look Into Person)**: Human parsing benchmark  
- **VITON-HD Dataset**: Reference for try-on dataset structure

---

For questions about dataset creation, refer to the main [README.md](README.md) or open an issue.

