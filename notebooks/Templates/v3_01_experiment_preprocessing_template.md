## 📓 Notebook Manager

This cell initializes the widgets required for managing your research notebook. Please run the cell below to enable functionality for:
- Exporting cells tagged with `export` into a `clean` notebook
- Generating a dynamic Table of Contents (TOC)
- Exporting the notebook to GitHub-compatible Markdown

➡️ **Be sure to execute the next cell before continuing with any editing or exporting.**


```python
# Cell 1 - Workflow Tools
import sys
sys.path.insert(0, '../../lib')
sys.path.insert(0, '../../scripts') 

from notebook_tools import TOCWidget, ExportWidget
import ipywidgets as widgets


# Create widget instances
toc = TOCWidget()
export = ExportWidget()

# Create horizontal layout
left_side = widgets.VBox([toc.button, export.button, toc.status])
right_side = widgets.VBox([toc.output, export.output])

# Display side by side
display(widgets.HBox([left_side, right_side]))
```


    HBox(children=(VBox(children=(Button(button_style='primary', description='Generate TOC', icon='list', style=Bu…


<!-- TOC -->
# Table of Contents

- [📓 Notebook Manager](#📓-notebook-manager)
- [🎯 Purpose](#🎯-purpose)
- [📋 Context](#📋-context)
- [🔄 Workflow Overview](#🔄-workflow-overview)
- [📁 Output Structure](#📁-output-structure)
- [🎮 Next Steps](#🎮-next-steps)
- [🔧 Experiment Configuration Parameters](#🔧-experiment-configuration-parameters)
  - [Target Parameters](#target-parameters)
  - [Path Configuration](#path-configuration)
  - [Processing Settings](#processing-settings)
  - [Quality Thresholds](#quality-thresholds)
- [🔧 Environment Setup](#🔧-environment-setup)
- [📹 Video Ingestion & Selection](#📹-video-ingestion-&-selection)
  - [Process](#process)
  - [Expected File Pattern](#expected-file-pattern)
- [🎞️ Frame Extraction](#🎞️-frame-extraction)
  - [Process](#process)
  - [Frame Naming](#frame-naming)
- [🎨 Color Normalization](#🎨-color-normalization)
  - [Process](#process)
  - [Technical Details](#technical-details)
- [📁 Data Organization](#📁-data-organization)
  - [Process](#process)
  - [Output Files](#output-files)
- [💾 Export & Summary](#💾-export-&-summary)
  - [Process](#process)
  - [Next Steps](#next-steps)

<!-- /TOC -->


# 🚗 Experiment Preprocessing - Car Counting v2.01

## 🎯 Purpose
This notebook implements a simplified preprocessing workflow for car counting experiments. It extracts frames from GDOT traffic camera videos for manual annotation and YOLO model training.

## 📋 Context
- **Data Source**: GDOT traffic camera recordings
- **Video Specs**: 480p resolution, 15 fps, ~15 minute duration
- **Goal**: Extract frames for vehicle counting model training

## 🔄 Workflow Overview
1. Video selection (closest to noon from target date)
2. Frame extraction (evenly spaced samples)
3. Quality filtering (brightness/blur)
4. Color normalization (BGR→RGB)
5. Data organization for annotation

## 📁 Output Structure
```
data/experiments/[experiment_name]/[date]/[camera]/
├── frames/           # raw extracted frames
├── quality/          # quality-filtered frames  
├── normalized/       # RGB-normalized frames
└── metadata.json     # processing summary
```

## 🎮 Next Steps
1. Run preprocessing → manual annotation in CVAT → YOLO training
2. Start with simplest YOLO model for vehicle counting

**Version**: 2.01 | **Type**: Experiment Preprocessing | **Target**: Car Counting

## 🔧 Experiment Configuration Parameters

This cell defines all parameters for the experiment preprocessing workflow. Parameters are organized into categories with emoji indicators:

### Target Parameters  
- 🎯 **VIDEO_ID**: Specific camera to process (e.g., ATL-1005)
- 🎯 **BATCH_DATE**: Date from batch analysis (YYYYMMDD format)  
- 🎯 **TARGET_HOUR**: Target hour for video selection (12 = noon)
- 🎯 **EXPERIMENT_NAME**: Unique identifier for this experiment

### Path Configuration
- 📁 **INPUT_BASE**: Root directory for video recordings
- 📁 **OUTPUT_BASE**: Root directory for experiment output

### Processing Settings
- 📊 **FRAMES_TO_EXTRACT**: Total number of frames to extract
- 📊 **SAMPLE_RATE**: Extract every Nth frame from video

### Quality Thresholds
Relaxed values for experiment data collection:
- 🔍 **brightness_min/max**: Acceptable brightness range (0-255)
- 🔍 **blur_min**: Minimum blur score (Laplacian variance)

**Note**: These values are hardcoded for initial experiments. Future versions will support parameter files.



```python
# experiment configuration parameters
from pathlib import Path

CONFIG = {
    # target parameters
    'VIDEO_ID': 'ATL-1005',  # 🎯 camera to process
    'BATCH_DATE': '20250620',  # 🎯 date from batch analysis
    'TARGET_HOUR': 12,  # 🎯 target hour (noon)
    'EXPERIMENT_NAME': 'car_counting_v1',  # 🎯 experiment identifier
    
    # path configuration  
    'INPUT_BASE': Path.home() / 'traffic-recordings',  # 📁 video source
    'OUTPUT_BASE': Path('../../data/experiments'),  # 📁 experiment output
    
    # processing settings
    'FRAMES_TO_EXTRACT': 50,  # 📊 total frames to extract
    'SAMPLE_RATE': 15,  # 📊 extract every Nth frame
    
    # quality measurement (no hardcoded thresholds)
    'MEASURE_QUALITY': True,  # 📊 calculate brightness/blur metrics
    
    # video settings (480p, 15fps)
    'JPEG_QUALITY': 95  # 🎥 output quality (0-100)
}

# derived paths
date_formatted = f"{CONFIG['BATCH_DATE'][:4]}-{CONFIG['BATCH_DATE'][4:6]}-{CONFIG['BATCH_DATE'][6:8]}"
CONFIG['OUTPUT_DIR'] = CONFIG['OUTPUT_BASE'] / CONFIG['EXPERIMENT_NAME'] / date_formatted / CONFIG['VIDEO_ID']
CONFIG['VIDEO_DIR'] = CONFIG['INPUT_BASE'] / CONFIG['VIDEO_ID'] / date_formatted

print("Experiment Configuration:")
print(f"  Experiment: {CONFIG['EXPERIMENT_NAME']}")
print(f"  Processing: {CONFIG['VIDEO_ID']} from {date_formatted}")
print(f"  Output to: {CONFIG['OUTPUT_DIR']}")
```

    Experiment Configuration:
      Experiment: car_counting_v1
      Processing: ATL-1005 from 2025-06-20
      Output to: ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005


## 🔧 Environment Setup

This cell establishes the experiment preprocessing environment by:

1. **Import Required Libraries**
   - OpenCV (cv2) for video processing and frame extraction
   - NumPy for array operations and numerical computations
   - Pandas for data organization and metadata management
   - System utilities for path handling and file operations

2. **Library Verification**
   - Check OpenCV installation and version
   - Verify NumPy and Pandas availability
   - Display version information for debugging

3. **Initialize Helper Functions**
   - **calculate_brightness()**: Compute average pixel intensity (0-255)
   - **calculate_blur_score()**: Measure sharpness using Laplacian variance
   - **get_video_metadata()**: Extract video properties (fps, resolution, duration)

4. **Directory Setup**
   - Create experiment output directory structure
   - Ensure path exists before processing begins

**Note**: This cell must run successfully before proceeding with video processing.


```python
# environment setup
import cv2
import numpy as np
import pandas as pd
import os
import sys
import json
import logging
from datetime import datetime, timedelta

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# verify opencv
try:
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError:
    print("⚠️ OpenCV not installed. Install with: pip install opencv-python")
    
print(f"✓ Python version: {sys.version.split()[0]}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Pandas version: {pd.__version__}")

# helper functions
def calculate_brightness(frame):
    """Calculate average brightness of frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_blur_score(frame):
    """Calculate Laplacian variance (higher = sharper)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_video_metadata(video_path):
    """Extract video metadata"""
    metadata = {}
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
        metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        metadata['duration_seconds'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
        cap.release()
    return metadata

# create output directory
CONFIG['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)

print(f"\n✓ Environment setup complete")
print(f"  Output directory created: {CONFIG['OUTPUT_DIR']}")
```

    ✓ OpenCV version: 4.11.0
    ✓ Python version: 3.12.9
    ✓ NumPy version: 2.2.4
    ✓ Pandas version: 2.2.3
    
    ✓ Environment setup complete
      Output directory created: ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005


## 📹 Video Ingestion & Selection

This module finds the target video from the specified camera and date using the same logic as the batch processing workflow.

### Process
- Searches for videos matching the camera ID and batch date
- Parses timestamps from filenames (format: CAMERA_YYYYMMDD_HHMMSS.mp4)
- Calculates time difference from target hour (noon)
- Selects video closest to target time
- Extracts basic metadata (resolution, duration, frame count)

### Expected File Pattern
`ATL-1005_20250620_120347.mp4` (camera_date_time.mp4)

**Note**: If no videos are found, check the INPUT_BASE path and date format.


```python
# video ingestion and selection
def parse_timestamp(filename):
    """extract timestamp from filename"""
    parts = filename.stem.split('_')
    if len(parts) >= 3:
        time_str = parts[2]
        hours = int(time_str[:2])
        minutes = int(time_str[2:4])
        return hours * 60 + minutes  # minutes from midnight
    return None

# find videos
video_files = list(CONFIG['VIDEO_DIR'].glob(f"{CONFIG['VIDEO_ID']}_*.mp4"))

if not video_files:
    raise FileNotFoundError(f"No videos found for {CONFIG['VIDEO_ID']} on {CONFIG['BATCH_DATE']}")

# find closest to target hour
target_minutes = CONFIG['TARGET_HOUR'] * 60  # 720 minutes for noon
closest_video = None
min_diff = float('inf')

for video in video_files:
    minutes = parse_timestamp(video)
    if minutes is not None:
        diff = abs(minutes - target_minutes)
        if diff < min_diff:
            min_diff = diff
            closest_video = video

CONFIG['selected_video'] = closest_video
metadata = get_video_metadata(closest_video)
CONFIG['video_metadata'] = metadata

time_str = closest_video.stem.split('_')[2]
print(f"Selected: {closest_video.name}")
print(f"  Starts at: {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}")
print(f"  Resolution: {metadata['width']}x{metadata['height']}")
print(f"  Duration: {metadata['duration_seconds']:.1f}s ({metadata['duration_seconds']/60:.1f} min)")
print(f"  Frame rate: {metadata['fps']:.1f} fps")
```

    Selected: ATL-1005_20250620_120641.mp4
      Starts at: 12:06:41
      Resolution: 480x270
      Duration: 901.1s (15.0 min)
      Frame rate: 15.0 fps


## 🎞️ Frame Extraction

This module extracts frames from the selected video at specified intervals for experiment processing.

### Process
- Opens the selected video using OpenCV
- Extracts frames using the SAMPLE_RATE (every Nth frame)
- Continues until FRAMES_TO_EXTRACT is reached or video ends
- Saves frames as JPEG files with specified quality
- Creates frames directory in experiment output folder

### Frame Naming
- Format: `frame_0001.jpg`, `frame_0002.jpg`, etc.
- Sequential numbering for easy sorting and reference

**Note**: For 15-minute videos at 15fps, total frames ≈ 13,500. Sample rate of 15 gives ~900 available frames.


```python
# frame extraction
import matplotlib.pyplot as plt

print(f"Frame Extraction")
print(f"Extracting {CONFIG['FRAMES_TO_EXTRACT']} frames (every {CONFIG['SAMPLE_RATE']} frames)")

video_path = CONFIG['selected_video']
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise ValueError(f"Cannot open video: {video_path}")

# create frames directory
frames_dir = CONFIG['OUTPUT_DIR'] / 'frames'
frames_dir.mkdir(exist_ok=True)

# extract frames
frames_extracted = 0
frame_index = 0
sample_frames = []  # store first few frames for preview

while frames_extracted < CONFIG['FRAMES_TO_EXTRACT'] and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # extract every Nth frame
    if frame_index % CONFIG['SAMPLE_RATE'] == 0:
        frame_filename = f"frame_{frames_extracted:04d}.jpg"
        frame_path = frames_dir / frame_filename
        
        # save frame
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
        
        # store first 9 frames for preview
        if frames_extracted < 9:
            sample_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frames_extracted += 1
        if frames_extracted % 10 == 0:
            print(f"  Extracted {frames_extracted}/{CONFIG['FRAMES_TO_EXTRACT']} frames")
    
    frame_index += 1

cap.release()

# show sample extracted frames
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
fig.suptitle(f'Sample Extracted Frames (First 9 of {frames_extracted})', fontsize=14)

for i in range(9):
    row = i // 3
    col = i % 3
    
    if i < len(sample_frames):
        axes[row, col].imshow(sample_frames[i])
        axes[row, col].set_title(f'Frame {i+1}', fontsize=10)
    else:
        axes[row, col].text(0.5, 0.5, 'No Frame', ha='center', va='center')
        axes[row, col].set_title(f'Frame {i+1}', fontsize=10)
    
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

CONFIG['frames_dir'] = frames_dir
CONFIG['frames_extracted'] = frames_extracted

print(f"\n✓ Extracted {frames_extracted} frames to {frames_dir}")
print(f"  Sample rate: every {CONFIG['SAMPLE_RATE']} frames from {CONFIG['video_metadata']['frame_count']} total")
print(f"  Time span: ~{(frames_extracted * CONFIG['SAMPLE_RATE']) / CONFIG['video_metadata']['fps'] / 60:.1f} minutes of video")
```

    Frame Extraction
    Extracting 50 frames (every 15 frames)
      Extracted 10/50 frames
      Extracted 20/50 frames
      Extracted 30/50 frames
      Extracted 40/50 frames
      Extracted 50/50 frames



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_11_1.png)
    


    
    ✓ Extracted 50 frames to ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005/frames
      Sample rate: every 15 frames from 13516 total
      Time span: ~0.8 minutes of video



```python
# quality analysis
import matplotlib.pyplot as plt

print("Quality Analysis")
print("=" * 60)

# get all extracted frames
frame_files = sorted(CONFIG['frames_dir'].glob("frame_*.jpg"))
print(f"Analyzing quality of {len(frame_files)} frames")

quality_results = []

for frame_path in frame_files:
    # read frame
    frame = cv2.imread(str(frame_path))
    if frame is None:
        continue
    
    # calculate metrics
    brightness = calculate_brightness(frame)
    blur_score = calculate_blur_score(frame)
    
    result = {
        'frame': frame_path.name,
        'brightness': brightness,
        'blur_score': blur_score,
        'path': frame_path
    }
    quality_results.append(result)

# save quality report
quality_df = pd.DataFrame(quality_results)
quality_df.to_csv(CONFIG['OUTPUT_DIR'] / 'quality_report.csv', index=False)

# find quality extremes for visualization
brightest = quality_df.loc[quality_df['brightness'].idxmax()]
darkest = quality_df.loc[quality_df['brightness'].idxmin()]
sharpest = quality_df.loc[quality_df['blur_score'].idxmax()]
blurriest = quality_df.loc[quality_df['blur_score'].idxmin()]

# show quality examples
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# brightest frame
bright_img = cv2.imread(str(brightest['path']))
bright_rgb = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB)
axes[0, 0].imshow(bright_rgb)
axes[0, 0].set_title(f'Brightest Frame\nBrightness: {brightest["brightness"]:.1f}')
axes[0, 0].axis('off')

# darkest frame
dark_img = cv2.imread(str(darkest['path']))
dark_rgb = cv2.cvtColor(dark_img, cv2.COLOR_BGR2RGB)
axes[0, 1].imshow(dark_rgb)
axes[0, 1].set_title(f'Darkest Frame\nBrightness: {darkest["brightness"]:.1f}')
axes[0, 1].axis('off')

# sharpest frame
sharp_img = cv2.imread(str(sharpest['path']))
sharp_rgb = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
axes[1, 0].imshow(sharp_rgb)
axes[1, 0].set_title(f'Sharpest Frame\nBlur Score: {sharpest["blur_score"]:.1f}')
axes[1, 0].axis('off')

# blurriest frame
blur_img = cv2.imread(str(blurriest['path']))
blur_rgb = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
axes[1, 1].imshow(blur_rgb)
axes[1, 1].set_title(f'Blurriest Frame\nBlur Score: {blurriest["blur_score"]:.1f}')
axes[1, 1].axis('off')

plt.suptitle('Quality Analysis - Extreme Examples')
plt.tight_layout()
plt.show()

# quality distribution plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(quality_df['brightness'], bins=20, alpha=0.7, color='orange')
ax1.set_xlabel('Brightness (0-255)')
ax1.set_ylabel('Frame Count')
ax1.set_title('Brightness Distribution')
ax1.grid(True, alpha=0.3)

ax2.hist(quality_df['blur_score'], bins=20, alpha=0.7, color='blue')
ax2.set_xlabel('Blur Score (higher = sharper)')
ax2.set_ylabel('Frame Count')
ax2.set_title('Sharpness Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

CONFIG['quality_results'] = quality_df

print(f"\nQuality Metrics Summary:")
print(f"  Brightness range: {quality_df['brightness'].min():.1f} - {quality_df['brightness'].max():.1f}")
print(f"  Brightness average: {quality_df['brightness'].mean():.1f}")
print(f"  Blur score range: {quality_df['blur_score'].min():.1f} - {quality_df['blur_score'].max():.1f}")
print(f"  Blur score average: {quality_df['blur_score'].mean():.1f}")

print(f"\n✓ Quality analysis complete")
print(f"  Report saved: quality_report.csv")
```

    Quality Analysis
    ============================================================
    Analyzing quality of 50 frames



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_12_1.png)
    



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_12_2.png)
    


    
    Quality Metrics Summary:
      Brightness range: 100.4 - 103.5
      Brightness average: 102.6
      Blur score range: 3459.5 - 3794.0
      Blur score average: 3584.2
    
    ✓ Quality analysis complete
      Report saved: quality_report.csv


## 🎨 Color Normalization

This module converts frames from BGR (Blue-Green-Red) to RGB (Red-Green-Blue) color space for consistency with display systems and ML frameworks.

### Process
- OpenCV reads images in BGR format by default
- Converts BGR to RGB using cv2.cvtColor()
- Saves normalized frames for downstream processing
- Creates normalized directory in experiment output

### Technical Details
- BGR: pixel[0]=Blue, pixel[1]=Green, pixel[2]=Red
- RGB: pixel[0]=Red, pixel[1]=Green, pixel[2]=Blue
- Essential for correct color representation in training data

**Note**: This ensures traffic lights appear red (not blue) in annotated data.


```python
# color normalization
import matplotlib.pyplot as plt
from PIL import Image

print("Color Space Normalization")
print("=" * 60)

frame_files = sorted(CONFIG['frames_dir'].glob("frame_*.jpg"))
print(f"Normalizing {len(frame_files)} frames")

# create normalized directory
normalized_dir = CONFIG['OUTPUT_DIR'] / 'normalized'
normalized_dir.mkdir(exist_ok=True)

normalized_frames = []

for i, frame_path in enumerate(frame_files):
    # read frame (BGR)
    frame_bgr = cv2.imread(str(frame_path))
    
    # convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # save as true RGB using PIL
    output_path = normalized_dir / frame_path.name
    rgb_image = Image.fromarray(frame_rgb)
    rgb_image.save(str(output_path), 'JPEG', quality=CONFIG['JPEG_QUALITY'])
    normalized_frames.append(output_path)

# show color space difference using first frame
sample = cv2.imread(str(frame_files[0]))
sample_rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# BGR channels
axes[0, 0].imshow(sample[:,:,0], cmap='Blues')
axes[0, 0].set_title('BGR - Blue Channel (index 0)')
axes[0, 0].axis('off')

axes[0, 1].imshow(sample[:,:,1], cmap='Greens')
axes[0, 1].set_title('BGR - Green Channel (index 1)')
axes[0, 1].axis('off')

axes[0, 2].imshow(sample[:,:,2], cmap='Reds')
axes[0, 2].set_title('BGR - Red Channel (index 2)')
axes[0, 2].axis('off')

# RGB display comparison
axes[1, 0].imshow(sample)
axes[1, 0].set_title('Direct Display (BGR - Wrong Colors)')
axes[1, 0].axis('off')

axes[1, 1].imshow(sample_rgb)
axes[1, 1].set_title('After Conversion (RGB - Correct Colors)')
axes[1, 1].axis('off')

# show pixel values
pixel_y, pixel_x = 100, 100
bgr_pixel = sample[pixel_y, pixel_x]
rgb_pixel = sample_rgb[pixel_y, pixel_x]
axes[1, 2].text(0.1, 0.7, f"Sample pixel at ({pixel_x},{pixel_y}):", fontsize=12, weight='bold')
axes[1, 2].text(0.1, 0.5, f"BGR: [{bgr_pixel[0]}, {bgr_pixel[1]}, {bgr_pixel[2]}]", fontsize=11)
axes[1, 2].text(0.1, 0.3, f"RGB: [{rgb_pixel[0]}, {rgb_pixel[1]}, {rgb_pixel[2]}]", fontsize=11)
axes[1, 2].text(0.1, 0.1, "Note: B↔R values swapped", fontsize=10, style='italic')
axes[1, 2].axis('off')

plt.suptitle('Color Space Normalization: BGR → RGB')
plt.tight_layout()
plt.show()

CONFIG['normalized_frames'] = normalized_frames

print(f"\n✓ Color normalization complete")
print(f"  Frames processed: {len(normalized_frames)}")
print(f"  Output: {normalized_dir}")
```

    Color Space Normalization
    ============================================================
    Normalizing 50 frames



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_14_1.png)
    


    
    ✓ Color normalization complete
      Frames processed: 50
      Output: ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005/normalized


## 📁 Data Organization

This module creates comprehensive metadata and organizes processed frames for downstream use.

### Process
- Compiles experiment metadata (camera, date, parameters, processing summary)
- Creates frame inventory with quality metrics
- Saves metadata as JSON for experiment tracking
- Saves frame inventory as CSV for easy review
- Links all processing stages and file paths

### Output Files
- `metadata.json`: Complete experiment configuration and results
- `frame_inventory.csv`: Frame-by-frame quality metrics and paths
- Directory structure with clear organization

**Note**: Metadata enables reproducibility and traceability for model training.


```python
# data organization
import matplotlib.pyplot as plt

print("Data Organization")
print("=" * 60)

# compile metadata
metadata = {
    'experiment_name': CONFIG['EXPERIMENT_NAME'],
    'camera_id': CONFIG['VIDEO_ID'],
    'batch_date': CONFIG['BATCH_DATE'],
    'source_video': CONFIG['selected_video'].name,
    'processing_timestamp': datetime.now().isoformat(),
    'parameters': {
        'target_hour': CONFIG['TARGET_HOUR'],
        'frames_to_extract': CONFIG['FRAMES_TO_EXTRACT'],
        'sample_rate': CONFIG['SAMPLE_RATE'],
        'jpeg_quality': CONFIG['JPEG_QUALITY']
    },
    'video_metadata': CONFIG['video_metadata'],
    'processing_summary': {
        'frames_extracted': CONFIG['frames_extracted'],
        'frames_with_quality_data': len(CONFIG['quality_results'])
    }
}

# build frame inventory
frame_inventory = []

for stage, frame_list, stage_name in [
    ('extracted', frame_files, 'Raw extracted frames'),
    ('normalized', normalized_frames, 'RGB normalized frames')
]:
    for frame_path in frame_list:
        # get quality metrics
        quality_row = CONFIG['quality_results'][CONFIG['quality_results']['frame'] == frame_path.name]
        
        entry = {
            'frame_name': frame_path.name,
            'stage': stage,
            'stage_description': stage_name,
            'path': str(frame_path),
            'brightness': quality_row['brightness'].values[0] if not quality_row.empty else None,
            'blur_score': quality_row['blur_score'].values[0] if not quality_row.empty else None
        }
        frame_inventory.append(entry)

# show processing pipeline with sample frame
sample_frame_name = frame_files[5].name  # use 6th frame as example
original_path = CONFIG['frames_dir'] / sample_frame_name
normalized_path = normalized_dir / sample_frame_name

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# original frame (BGR as read by OpenCV)
original_bgr = cv2.imread(str(original_path))
axes[0].imshow(original_bgr)  # shows with wrong colors (BGR display)
axes[0].set_title('Step 1: Raw Extracted Frame\n(BGR - incorrect display colors)')
axes[0].axis('off')

# quality metrics overlay
quality_info = CONFIG['quality_results'][CONFIG['quality_results']['frame'] == sample_frame_name]
if not quality_info.empty:
    brightness = quality_info['brightness'].values[0]
    blur_score = quality_info['blur_score'].values[0]
    axes[0].text(10, 30, f'B: {brightness:.0f}\nS: {blur_score:.0f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10, color='black')

# normalized frame (RGB using PIL)
normalized_rgb = plt.imread(str(normalized_path))
axes[1].imshow(normalized_rgb)
axes[1].set_title('Step 2: Quality Analyzed\n(metrics calculated)')
axes[1].axis('off')

# final normalized RGB
axes[2].imshow(normalized_rgb)
axes[2].set_title('Step 3: RGB Normalized\n(ready for CVAT annotation)')
axes[2].axis('off')

plt.suptitle(f'Processing Pipeline - {sample_frame_name}')
plt.tight_layout()
plt.show()

# save metadata
metadata_path = CONFIG['OUTPUT_DIR'] / 'metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# save frame inventory
inventory_df = pd.DataFrame(frame_inventory)
inventory_path = CONFIG['OUTPUT_DIR'] / 'frame_inventory.csv'
inventory_df.to_csv(inventory_path, index=False)

print(f"\nExperiment data structure:")
print(f"  {CONFIG['OUTPUT_DIR']}/")
print(f"  ├── frames/          ({CONFIG['frames_extracted']} raw frames)")
print(f"  ├── normalized/      ({len(normalized_frames)} RGB frames)")
print(f"  ├── metadata.json")
print(f"  ├── frame_inventory.csv")
print(f"  └── quality_report.csv")

CONFIG['metadata'] = metadata
CONFIG['inventory'] = inventory_df

print(f"\n✓ Data organization complete")
print(f"  Metadata: {metadata_path}")
print(f"  Inventory: {inventory_path}")
```

    Data Organization
    ============================================================



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_16_1.png)
    


    
    Experiment data structure:
      ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005/
      ├── frames/          (50 raw frames)
      ├── normalized/      (50 RGB frames)
      ├── metadata.json
      ├── frame_inventory.csv
      └── quality_report.csv
    
    ✓ Data organization complete
      Metadata: ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005/metadata.json
      Inventory: ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005/frame_inventory.csv


## 💾 Export & Summary

This module creates a final summary of the experiment preprocessing workflow and confirms frames are ready for annotation.

### Process
- Creates preprocessing completion summary
- Documents frame counts and processing stages
- Confirms location of frames ready for CVAT annotation
- Saves final summary report

### Next Steps
1. **Manual Annotation**: Import normalized frames into CVAT
2. **Vehicle Labeling**: Annotate cars, trucks, motorcycles with bounding boxes
3. **Export Labels**: Download YOLO format annotations
4. **Model Training**: Train simplest YOLO model for vehicle counting

**Output**: Normalized RGB frames in `/normalized/` directory ready for annotation.


```python
# export and summary
import matplotlib.pyplot as plt

print("Export & Summary")
print("=" * 60)

# create final summary
summary = {
    'experiment_complete': datetime.now().isoformat(),
    'experiment_name': CONFIG['EXPERIMENT_NAME'],
    'camera': CONFIG['VIDEO_ID'],
    'video_processed': CONFIG['selected_video'].name,
    'frames_ready_for_annotation': len(normalized_frames),
    'annotation_directory': str(CONFIG['OUTPUT_DIR'] / 'normalized'),
    'processing_stages': {
        '1_extracted': CONFIG['frames_extracted'],
        '2_quality_analyzed': len(CONFIG['quality_results']),
        '3_rgb_normalized': len(normalized_frames)
    },
    'quality_metrics': {
        'avg_brightness': float(CONFIG['quality_results']['brightness'].mean()),
        'avg_blur_score': float(CONFIG['quality_results']['blur_score'].mean()),
        'brightness_range': [float(CONFIG['quality_results']['brightness'].min()), 
                           float(CONFIG['quality_results']['brightness'].max())],
        'blur_range': [float(CONFIG['quality_results']['blur_score'].min()), 
                      float(CONFIG['quality_results']['blur_score'].max())]
    }
}

# show final annotation-ready frames (first 12)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(f'Final Frames Ready for CVAT Annotation (Showing 12 of {len(normalized_frames)})', fontsize=14)

for i in range(12):
    row = i // 4
    col = i % 4
    
    if i < len(normalized_frames):
        # load RGB normalized frame
        frame_rgb = plt.imread(str(normalized_frames[i]))
        axes[row, col].imshow(frame_rgb)
        
        # get quality metrics for this frame
        frame_name = normalized_frames[i].name
        quality_info = CONFIG['quality_results'][CONFIG['quality_results']['frame'] == frame_name]
        if not quality_info.empty:
            brightness = quality_info['brightness'].values[0]
            blur_score = quality_info['blur_score'].values[0]
            axes[row, col].set_title(f'{frame_name}\nB:{brightness:.0f} S:{blur_score:.0f}', fontsize=9)
        else:
            axes[row, col].set_title(f'{frame_name}', fontsize=9)
    else:
        axes[row, col].text(0.5, 0.5, 'No Frame', ha='center', va='center')
        axes[row, col].set_title(f'Frame {i+1}', fontsize=9)
    
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Quality Distribution Analysis
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(CONFIG['quality_results']['brightness'], CONFIG['quality_results']['blur_score'], 
           alpha=0.7, s=100, c='blue')
ax.set_xlabel('Brightness (0-255)')
ax.set_ylabel('Blur Score (higher = sharper)')
ax.set_title('Frame Quality Distribution - All Extracted Frames')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("📊 Quality Distribution Analysis:")
brightness_range = CONFIG['quality_results']['brightness'].max() - CONFIG['quality_results']['brightness'].min()
blur_range = CONFIG['quality_results']['blur_score'].max() - CONFIG['quality_results']['blur_score'].min()
print(f"• Brightness range: {brightness_range:.1f} units ({CONFIG['quality_results']['brightness'].min():.1f} to {CONFIG['quality_results']['brightness'].max():.1f})")
print(f"• Blur score range: {blur_range:.0f} units ({CONFIG['quality_results']['blur_score'].min():.0f} to {CONFIG['quality_results']['blur_score'].max():.0f})")
print(f"• Frame clustering: 50% of frames between {CONFIG['quality_results']['brightness'].quantile(0.25):.1f}-{CONFIG['quality_results']['brightness'].quantile(0.75):.1f} brightness")
print(f"• Outliers detected: {len(CONFIG['quality_results'][(CONFIG['quality_results']['brightness'] < CONFIG['quality_results']['brightness'].quantile(0.1)) | (CONFIG['quality_results']['brightness'] > CONFIG['quality_results']['brightness'].quantile(0.9))])} frames outside 80% range")
print(f"• Data characteristics: Measured variation will inform annotation and training decisions")

# Quality Trends Over Time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(range(len(CONFIG['quality_results'])), CONFIG['quality_results']['brightness'], 
         'o-', alpha=0.7, color='orange')
ax1.set_ylabel('Brightness')
ax1.set_title('Quality Metrics Throughout Video Sequence')
ax1.grid(True, alpha=0.3)

ax2.plot(range(len(CONFIG['quality_results'])), CONFIG['quality_results']['blur_score'], 
         'o-', alpha=0.7, color='blue')
ax2.set_xlabel('Frame Extraction Order')
ax2.set_ylabel('Blur Score')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📈 Temporal Quality Analysis:")
brightness_std = CONFIG['quality_results']['brightness'].std()
blur_std = CONFIG['quality_results']['blur_score'].std()
print(f"• Brightness variation: {brightness_std:.2f} standard deviation over sequence")
print(f"• Blur variation: {blur_std:.0f} standard deviation over sequence")
print(f"• Temporal patterns: {brightness_range/brightness_std:.1f}x range-to-variation ratio for lighting")
print(f"• Sequence stability: {blur_range/blur_std:.1f}x range-to-variation ratio for sharpness")
print(f"• Data summary: Metrics provide baseline for annotation quality assessment")

# save summary
summary_path = CONFIG['OUTPUT_DIR'] / 'experiment_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Experiment Complete: {CONFIG['EXPERIMENT_NAME']}")
print(f"\nFrames ready for annotation: {len(normalized_frames)}")
print(f"Location: {CONFIG['OUTPUT_DIR'] / 'normalized'}")
print(f"\nProcessing pipeline: {CONFIG['frames_extracted']} → {len(normalized_frames)} frames")
print(f"Quality metrics saved: {summary_path}")

print(f"\n🎯 Next Steps:")
print(f"  1. Import frames from 'normalized/' into CVAT")
print(f"  2. Annotate vehicles with bounding boxes")
print(f"  3. Export YOLO format labels")
print(f"  4. Train vehicle counting model")

print(f"\n✓ Experiment preprocessing complete")
```

    Export & Summary
    ============================================================



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_18_1.png)
    



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_18_2.png)
    


    📊 Quality Distribution Analysis:
    • Brightness range: 3.1 units (100.4 to 103.5)
    • Blur score range: 334 units (3460 to 3794)
    • Frame clustering: 50% of frames between 102.2-103.1 brightness
    • Outliers detected: 10 frames outside 80% range
    • Data characteristics: Measured variation will inform annotation and training decisions



    
![png](v3_01_experiment_preprocessing_template_files/v3_01_experiment_preprocessing_template_18_4.png)
    


    
    📈 Temporal Quality Analysis:
    • Brightness variation: 0.69 standard deviation over sequence
    • Blur variation: 65 standard deviation over sequence
    • Temporal patterns: 4.5x range-to-variation ratio for lighting
    • Sequence stability: 5.2x range-to-variation ratio for sharpness
    • Data summary: Metrics provide baseline for annotation quality assessment
    Experiment Complete: car_counting_v1
    
    Frames ready for annotation: 50
    Location: ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005/normalized
    
    Processing pipeline: 50 → 50 frames
    Quality metrics saved: ../../data/experiments/car_counting_v1/2025-06-20/ATL-1005/experiment_summary.json
    
    🎯 Next Steps:
      1. Import frames from 'normalized/' into CVAT
      2. Annotate vehicles with bounding boxes
      3. Export YOLO format labels
      4. Train vehicle counting model
    
    ✓ Experiment preprocessing complete



```python

```


```python

```


```python

```


```python

```
