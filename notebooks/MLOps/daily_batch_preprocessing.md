# 📓 Notebook Manager

This cell initializes the widgets required for managing your research notebook. Please run the cell below to enable functionality for:
- Exporting cells tagged with `export` into a `clean` notebook
- Generating a dynamic Table of Contents (TOC)
- Exporting the notebook to GitHub-compatible Markdown

➡️ **Be sure to execute the next cell before continuing with any editing or exporting.**


```python
# Cell 1 - Workflow Tools
import sys
sys.path.insert(0, '../../lib')

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

<!-- TOC -->
# Table of Contents

- [🔧 Environment Setup](#🔧-environment-setup)
- [📐 Batch Processing Configuration](#📐-batch-processing-configuration)
  - [Parameters](#parameters)
  - [Configuration Legend](#configuration-legend)
  - [File Pattern](#file-pattern)
- [💾 Initialize Checkpoint System](#💾-initialize-checkpoint-system)
  - [Features](#features)
- [📂 Scan Video Directories](#📂-scan-video-directories)
  - [Process](#process)
- [🎯 Find Target Videos](#🎯-find-target-videos)
  - [Algorithm](#algorithm)
  - [Metadata to Extract](#metadata-to-extract)
- [💾 Save Video Manifest](#💾-save-video-manifest)
  - [Output Files](#output-files)
- [🎬 Preview Extraction Configuration](#🎬-preview-extraction-configuration)
  - [Parameters](#parameters)
- [🎥 Extract Preview Frames](#🎥-extract-preview-frames)
  - [Process](#process)
- [📊 Display Frame Previews](#📊-display-frame-previews)
- [📈 Quality Analysis & Recommendations](#📈-quality-analysis-&-recommendations)
- [🏆 Quality Samples](#🏆-quality-samples)
- [📚 Quality Metrics Reference](#📚-quality-metrics-reference)
  - [Brightness (Luminance)](#brightness-(luminance))
  - [Blur Score (Laplacian Variance)](#blur-score-(laplacian-variance))
  - [Quartile-Based Outlier Detection](#quartile-based-outlier-detection)
  - [Color Space Conversion (BGR to RGB)](#color-space-conversion-(bgr-to-rgb))
- [💾 Export Selection - Option 1: Automatic](#💾-export-selection---option-1-automatic)
- [💾 Export Selection - Option 2: Manual](#💾-export-selection---option-2-manual)
- [💾 Export Selection - Option 3: All Cameras](#💾-export-selection---option-3-all-cameras)
- [💾 Save Selection Queue](#💾-save-selection-queue)
- [📋 Batch Processing Summary](#📋-batch-processing-summary)
- [🎯 Final Video Selection](#🎯-final-video-selection)

<!-- /TOC -->


## 🔧 Environment Setup

This cell establishes the batch preprocessing environment by:

1. **Importing Required Libraries**
  - OpenCV (cv2) for video processing and frame extraction
  - NumPy for array operations
  - Pandas for organizing metadata and results
  - Pathlib for cross-platform file path handling
  - JSON for checkpoint persistence
  - Datetime for timestamp parsing and filtering
  - Logging for process tracking

2. **Setting System Paths**
  - Adding mlops_ops modules to Python path
  - Verifying access to preprocessing utilities

3. **Initializing Checkpoint System**
  - Loading any previous processing state
  - Setting up progress tracking variables
  - Establishing failure recovery mechanism

**Note**: Run this cell first to ensure all dependencies are available before proceeding with batch processing.


```python
# environment setup
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
import os
import sys

# add mlops modules
sys.path.insert(0, '../lib')

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# check opencv availability
try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError:
    print("⚠️ OpenCV not installed. Install with: pip install opencv-python")

print(f"✓ Python version: {sys.version.split()[0]}")
print(f"✓ Working directory: {os.getcwd()}")
```

## 📐 Batch Processing Configuration

Defines core parameters for the daily batch preprocessing workflow.

### Parameters
- **Target Time**: 12:00 PM (noon) in 24-hour format
- **Date Filter**: Previous calendar day only  
- **Frame Count**: Frames to extract per video
- **Input Path**: Base recordings directory
- **Output Path**: Organized output structure in `data/preprocessing/batch_analysis/YYYY-MM-DD/`

### Configuration Legend
- 🎯 = Adjustable targeting parameter
- 📊 = Data processing setting
- 📁 = Path configuration

### File Pattern
Expected format: `CAMERA_YYYYMMDD_HHMMSS.mp4`


```python
# batch processing configuration
from datetime import datetime, timedelta

CONFIG = {
    # time targeting
    'TARGET_TIME': '120000',  # 🎯 noon target time
    'TARGET_HOUR': 12,
    
    # date filtering yesterday
    'PROCESS_DATE': (datetime.now() - timedelta(days=1)).strftime('%Y%m%d'),
    
    # frame extraction
    'FRAMES_PER_VIDEO': 10,  # 📊 frames per video
    
    # paths
    'INPUT_DIR': Path.home() / 'traffic-recordings',  # 📁 source directory
    'OUTPUT_BASE': Path('../../data/preprocessing/batch_analysis'),  # 📁 output base
    
    # file pattern
    'VIDEO_PATTERN': '*_{date}_*.mp4',
    'FILENAME_FORMAT': '{camera}_{date}_{time}.mp4'
}

# create dated output with hyphens
date_formatted = f"{CONFIG['PROCESS_DATE'][:4]}-{CONFIG['PROCESS_DATE'][4:6]}-{CONFIG['PROCESS_DATE'][6:8]}"
date_dir = CONFIG['OUTPUT_BASE'] / date_formatted
date_dir.mkdir(parents=True, exist_ok=True)
CONFIG['OUTPUT_DIR'] = date_dir

# display configuration
print("Batch Processing Configuration:")
print(f"  Target Date: {CONFIG['PROCESS_DATE']}")
print(f"  Target Time: {CONFIG['TARGET_TIME']} (12:00:00)")
print(f"  Frames per video: {CONFIG['FRAMES_PER_VIDEO']}")
print(f"  Input: {CONFIG['INPUT_DIR']}")
print(f"  Output: {CONFIG['OUTPUT_DIR']}")
```

## 💾 Initialize Checkpoint System

Sets up checkpoint functionality to track processing progress and enable recovery from interruptions.

### Features
- Saves state after each video
- Detects previous runs
- Validates checkpoint date matches current processing date
- Enables resume from last successful video


```python
# checkpoint system initialization
CHECKPOINT_FILE = CONFIG['OUTPUT_DIR'] / "batch_checkpoint.json"
start_time = datetime.now()

def load_checkpoint():
    """load previous progress"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            print(f"✓ Loaded checkpoint: {len(checkpoint['processed'])} videos already processed")
            return checkpoint
    return {
        "processed": [], 
        "failed": [], 
        "last_completed": None,
        "process_date": CONFIG['PROCESS_DATE'],
        "start_time": start_time.isoformat()
    }

def save_checkpoint(checkpoint):
    """save current progress"""
    checkpoint['last_updated'] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

# initialize checkpoint
checkpoint = load_checkpoint()

# verify checkpoint date
if checkpoint.get('process_date') != CONFIG['PROCESS_DATE']:
    print(f"⚠️  Checkpoint is from {checkpoint.get('process_date')}, starting fresh for {CONFIG['PROCESS_DATE']}")
    checkpoint = {
        "processed": [], 
        "failed": [], 
        "last_completed": None,
        "process_date": CONFIG['PROCESS_DATE'],
        "start_time": start_time.isoformat()
    }

print(f"✓ Checkpoint system ready")
```

## 📂 Scan Video Directories

Enumerates camera subdirectories and counts videos from the target date.

### Process
- Identifies all camera directories (ATL-*)
- Checks for date-specific subdirectories
- Counts available videos per camera
- Reports missing recordings


```python
# scan video directories
camera_dirs = sorted([d for d in CONFIG['INPUT_DIR'].iterdir() if d.is_dir() and d.name.startswith('ATL-')])
print(f"Found {len(camera_dirs)} camera directories\n")

# count videos per camera
video_counts = {}
date_folder = CONFIG['PROCESS_DATE'][:4] + '-' + CONFIG['PROCESS_DATE'][4:6] + '-' + CONFIG['PROCESS_DATE'][6:8]
pattern = CONFIG['VIDEO_PATTERN'].format(date=CONFIG['PROCESS_DATE'])

for cam_dir in camera_dirs:
    date_dir = cam_dir / date_folder
    if date_dir.exists():
        videos = list(date_dir.glob(pattern))
        video_counts[cam_dir.name] = len(videos)
        
        if len(videos) == 0:
            print(f"⚠️  {cam_dir.name}: No videos in {date_folder}")
        else:
            print(f"✓ {cam_dir.name}: {len(videos)} videos")
    else:
        video_counts[cam_dir.name] = 0
        print(f"⚠️  {cam_dir.name}: No {date_folder} directory")

total_videos = sum(video_counts.values())
print(f"\nTotal videos available: {total_videos}")
print(f"Cameras with recordings: {sum(1 for v in video_counts.values() if v > 0)}/{len(camera_dirs)}")
```

## 🎯 Find Target Videos

Identifies the video closest to noon (12:00:00) for each camera and extracts metadata.

### Algorithm
- Parses timestamp from filename
- Calculates time difference from noon
- Selects minimum difference per camera

### Metadata to Extract
- Video duration
- Frame rate (fps)
- Resolution (width x height)
- Total frame count
- File size


```python
# find target videos
from datetime import datetime

def parse_timestamp(filename):
    """extract timestamp from filename"""
    parts = filename.stem.split('_')
    if len(parts) >= 3:
        time_str = parts[2]
        hours = int(time_str[:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])
        return hours * 60 + minutes  # minutes from midnight
    return None

def find_closest_to_noon(video_list):
    """find video closest to noon"""
    target_minutes = CONFIG['TARGET_HOUR'] * 60  # 720 minutes
    
    closest_video = None
    min_diff = float('inf')
    
    for video in video_list:
        minutes = parse_timestamp(video)
        if minutes is not None:
            diff = abs(minutes - target_minutes)
            if diff < min_diff:
                min_diff = diff
                closest_video = video
    
    return closest_video, min_diff

def get_video_metadata(video_path):
    """extract video metadata"""
    metadata = {}
    if cv2:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
            metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata['duration_seconds'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
            cap.release()
    
    metadata['file_size_mb'] = round(video_path.stat().st_size / (1024*1024), 2)
    return metadata

# find target videos
target_videos = []
date_folder = CONFIG['PROCESS_DATE'][:4] + '-' + CONFIG['PROCESS_DATE'][4:6] + '-' + CONFIG['PROCESS_DATE'][6:8]

print(f"Finding videos closest to {CONFIG['TARGET_TIME'][:2]}:{CONFIG['TARGET_TIME'][2:4]} (noon)...\n")

for cam_dir in camera_dirs:
    date_dir = cam_dir / date_folder
    if date_dir.exists():
        videos = list(date_dir.glob(f"{cam_dir.name}_*.mp4"))
        if videos:
            closest, diff_minutes = find_closest_to_noon(videos)
            if closest:
                metadata = get_video_metadata(closest)
                target_videos.append({
                    'camera': cam_dir.name,
                    'video_path': closest,
                    'time_diff_minutes': diff_minutes,
                    **metadata
                })
                time_str = closest.stem.split('_')[2]
                print(f"{cam_dir.name}:")
                print(f"  Video starts: {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}")
                if metadata.get('duration_seconds'):
                    print(f"  Resolution: {metadata['width']}x{metadata['height']}")
                    print(f"  Frame rate: {metadata['fps']:.1f} fps")
                    print(f"  Duration: {metadata['duration_seconds']:.1f}s ({metadata['duration_seconds']/60:.1f} min)")
                print(f"  File size: {metadata['file_size_mb']:.1f} MB")
                print()

print(f"Total videos selected: {len(target_videos)}")
```

## 💾 Save Video Manifest

Creates manifest files in the batch analysis directory with selected video metadata for tracking and downstream processing.

### Output Files
- JSON manifest with full metadata
- CSV summary for quick review
- Both saved to: `data/preprocessing/batch_analysis/YYYYMMDD/`


```python
# save video manifest
manifest_data = {
    'processing_date': CONFIG['PROCESS_DATE'],
    'target_time': CONFIG['TARGET_TIME'],
    'created_at': datetime.now().isoformat(),
    'total_cameras': len(camera_dirs),
    'videos_found': len(target_videos),
    'videos': []
}

for video_info in target_videos:
    video_path = video_info['video_path']
    time_str = video_path.stem.split('_')[2]
    
    manifest_data['videos'].append({
        'camera': video_info['camera'],
        'filename': video_path.name,
        'full_path': str(video_path),
        'recording_time': f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}",
        'time_diff_minutes': video_info['time_diff_minutes'],
        'file_size_mb': video_info.get('file_size_mb', 0),
        'fps': video_info.get('fps', 0),
        'width': video_info.get('width', 0),
        'height': video_info.get('height', 0),
        'duration_seconds': video_info.get('duration_seconds', 0)
    })

# save to output directory
manifest_file = CONFIG['OUTPUT_DIR'] / f"manifest_{CONFIG['PROCESS_DATE']}.json"
csv_file = CONFIG['OUTPUT_DIR'] / f"manifest_{CONFIG['PROCESS_DATE']}.csv"

with open(manifest_file, 'w') as f:
    json.dump(manifest_data, f, indent=2)

df_manifest = pd.DataFrame(manifest_data['videos'])
df_manifest.to_csv(csv_file, index=False)

print(f"✓ Saved manifest: {manifest_file}")
print(f"✓ Saved CSV: {csv_file}")
print(f"\nSummary:")
print(f"  Videos selected: {len(target_videos)}")
print(f"  Total size: {df_manifest['file_size_mb'].sum():.1f} MB")
print(f"  Average duration: {df_manifest['duration_seconds'].mean()/60:.1f} minutes")
```

## 🎬 Preview Extraction Configuration

Sets parameters for extracting sample frames from each video for quality assessment.

### Parameters
- 🎯 **frames_per_video**: Number of sample frames
- 🎯 **extraction_duration**: Seconds to sample from start
- 📊 **max_videos_to_preview**: Limit for testing
- 📁 **preview_dir**: Output location in batch directory


```python
# preview extraction configuration
PREVIEW_CONFIG = {
    'frames_per_video': 5,          # 🎯 sample frame count
    'extraction_duration': 60,      # 🎯 sample first 60s
    'preview_dir': CONFIG['OUTPUT_DIR'] / 'preview_frames',
    'max_videos_to_preview': None   # 📊 None = all videos
}

# set preview limit
if PREVIEW_CONFIG['max_videos_to_preview'] is None:
    PREVIEW_CONFIG['max_videos_to_preview'] = len(target_videos)

# create preview directory
PREVIEW_CONFIG['preview_dir'].mkdir(exist_ok=True)

print("Preview Configuration:")
print(f"  Frames per video: {PREVIEW_CONFIG['frames_per_video']}")
print(f"  Duration to sample: {PREVIEW_CONFIG['extraction_duration']}s")
print(f"  Output directory: {PREVIEW_CONFIG['preview_dir']}")
print(f"  Videos to preview: {PREVIEW_CONFIG['max_videos_to_preview']} of {len(target_videos)}")
```

## 🎥 Extract Preview Frames

Processes videos to extract sample frames with quality metrics for visual assessment.

### Process
- Extracts evenly-spaced frames from first N seconds
- Calculates brightness and blur scores
- Saves frames to preview directory


```python
# extract preview frames
import cv2

def extract_preview_frames(video_path, output_dir, num_frames=5, duration_seconds=60):
    """extract evenly spaced frames"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_frames = min(int(fps * duration_seconds), total_frames)
    
    # calculate frame indices
    indices = np.linspace(0, duration_frames-1, num_frames, dtype=int)
    
    frames_data = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # calculate metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # save frame
            frame_filename = f"frame_{idx:04d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            frames_data.append({
                'index': idx,
                'brightness': brightness,
                'blur_score': blur_score,
                'path': frame_path
            })
    
    cap.release()
    return frames_data

# process videos
print("Extracting preview frames...")
preview_results = []

for i, video_info in enumerate(target_videos[:PREVIEW_CONFIG['max_videos_to_preview']]):
    camera = video_info['camera']
    video_path = video_info['video_path']
    
    # create camera subdirectory
    output_dir = PREVIEW_CONFIG['preview_dir'] / camera
    output_dir.mkdir(exist_ok=True)
    
    # extract frames
    frames_data = extract_preview_frames(
        video_path, 
        output_dir,
        PREVIEW_CONFIG['frames_per_video'],
        PREVIEW_CONFIG['extraction_duration']
    )
    
    if frames_data:
        avg_brightness = np.mean([f['brightness'] for f in frames_data])
        avg_blur = np.mean([f['blur_score'] for f in frames_data])
        
        preview_results.append({
            'camera': camera,
            'video_path': video_path,
            'frames_extracted': len(frames_data),
            'avg_brightness': avg_brightness,
            'avg_blur_score': avg_blur,
            'frames_data': frames_data
        })
        
        print(f"✓ {camera}: {len(frames_data)} frames, brightness={avg_brightness:.1f}, blur={avg_blur:.1f}")
        
        # update checkpoint
        checkpoint['processed'].append(camera)
        save_checkpoint(checkpoint)
    else:
        print(f"✗ {camera}: Failed to extract frames")
        checkpoint['failed'].append(camera)
        save_checkpoint(checkpoint)

print(f"\nCompleted: {len(preview_results)} of {PREVIEW_CONFIG['max_videos_to_preview']} videos")
```

## 📊 Display Frame Previews

Generates visual grid of extracted frames with quality metrics for review.


```python
# display frame previews
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(preview_results), PREVIEW_CONFIG['frames_per_video'], 
                        figsize=(PREVIEW_CONFIG['frames_per_video'] * 3, len(preview_results) * 2))

if len(preview_results) == 1:
    axes = axes.reshape(1, -1)

for cam_idx, result in enumerate(preview_results):
    camera = result['camera']
    
    # display each frame
    for frame_idx, frame_data in enumerate(result['frames_data']):
        ax = axes[cam_idx, frame_idx]
        
        # read and display
        img = cv2.imread(str(frame_data['path']))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f"F{frame_data['index']}\nB:{frame_data['brightness']:.0f} S:{frame_data['blur_score']:.0f}", 
                    fontsize=8)
        ax.axis('off')
    
    # add camera label
    axes[cam_idx, 0].text(-0.1, 0.5, f"{camera}\nB:{result['avg_brightness']:.0f}\nS:{result['avg_blur_score']:.0f}", 
                          transform=axes[cam_idx, 0].transAxes,
                          ha='right', va='center', fontsize=9, weight='bold')

plt.suptitle(f'Batch Preview - {CONFIG["PROCESS_DATE"]} @ 12:00 ({len(preview_results)} cameras)', 
             fontsize=12, y=0.995)
plt.tight_layout()
plt.show()

# summary table
df_preview = pd.DataFrame([{
    'camera': r['camera'],
    'brightness': r['avg_brightness'],
    'blur_score': r['avg_blur_score']
} for r in preview_results])

print(f"\nQuality Summary ({len(preview_results)} cameras):")
print(f"  Brightness: {df_preview['brightness'].min():.1f} - {df_preview['brightness'].max():.1f}")
print(f"  Blur score: {df_preview['blur_score'].min():.1f} - {df_preview['blur_score'].max():.1f}")
```

## 📈 Quality Analysis & Recommendations

Analyzes frame quality metrics to identify videos needing individual review.


```python
# quality analysis
df_quality = pd.DataFrame([{
    'camera': r['camera'],
    'brightness': r['avg_brightness'],
    'blur_score': r['avg_blur_score']
} for r in preview_results])

# define thresholds
brightness_low = df_quality['brightness'].quantile(0.25)
brightness_high = df_quality['brightness'].quantile(0.75)
blur_low = df_quality['blur_score'].quantile(0.25)
blur_high = df_quality['blur_score'].quantile(0.75)
blur_top = df_quality['blur_score'].quantile(0.90)  # top 10% sharpest

# identify outliers
needs_review = []
high_quality = []

for r in preview_results:
    issues = []
    if r['avg_brightness'] < brightness_low:
        issues.append('low brightness')
    elif r['avg_brightness'] > brightness_high:
        issues.append('high brightness')
    
    if r['avg_blur_score'] < blur_low:
        issues.append('high blur')
    
    if issues:
        needs_review.append({
            'camera': r['camera'],
            'issues': ', '.join(issues),
            'brightness': r['avg_brightness'],
            'blur_score': r['avg_blur_score']
        })
    
    # track high quality
    if r['avg_blur_score'] >= blur_top:
        high_quality.append(r['camera'])

# display results
print("Quality Analysis:")
print(f"  Brightness quartiles: Q1={brightness_low:.1f}, Q3={brightness_high:.1f}")
print(f"  Blur score quartiles: Q1={blur_low:.1f}, Q3={blur_high:.1f}")
print(f"  Top 10% blur threshold: {blur_top:.1f}")

if high_quality:
    print(f"\nHighest quality cameras ({len(high_quality)}):")
    for cam in high_quality:
        print(f"  {cam}")

if needs_review:
    print(f"\nCameras needing review ({len(needs_review)}):")
    for video in needs_review:
        print(f"  {video['camera']}: {video['issues']}")

# scatter plot with colors
plt.figure(figsize=(10, 6))

for idx, row in df_quality.iterrows():
    if row['camera'] in high_quality:
        color = 'green'
        s = 150
    elif any(r['camera'] == row['camera'] for r in needs_review):
        color = 'red'
        s = 100
    else:
        color = 'blue'
        s = 100
    
    plt.scatter(row['brightness'], row['blur_score'], s=s, color=color)
    plt.annotate(row['camera'], (row['brightness'], row['blur_score']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.axvline(brightness_low, color='red', linestyle='--', alpha=0.5, label='Brightness Q1')
plt.axvline(brightness_high, color='red', linestyle='--', alpha=0.5, label='Brightness Q3')
plt.axhline(blur_low, color='blue', linestyle='--', alpha=0.5, label='Blur Q1')
plt.axhline(blur_top, color='green', linestyle='--', alpha=0.5, label='Top 10% Blur')

# legend
plt.scatter([], [], c='green', s=150, label='Highest quality')
plt.scatter([], [], c='red', s=100, label='Needs review')
plt.scatter([], [], c='blue', s=100, label='Normal')

plt.xlabel('Brightness')
plt.ylabel('Blur Score (higher = sharper)')
plt.title(f'Video Quality Distribution - {CONFIG["PROCESS_DATE"]}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```


```python
# quality samples visualization
categories = {
    'highest_quality': sorted([(r['camera'], r['avg_blur_score']) for r in preview_results], 
                             key=lambda x: x[1], reverse=True)[:3],
    'needs_review': [(r['camera'], r['avg_blur_score']) for r in preview_results 
                     if r['camera'] in [n['camera'] for n in needs_review]][:3],
    'highest_blur': sorted([(r['camera'], r['avg_blur_score']) for r in preview_results], 
                          key=lambda x: x[1])[:3],
    'lowest_blur': sorted([(r['camera'], r['avg_blur_score']) for r in preview_results], 
                         key=lambda x: x[1])[:3]
}

# create display
fig, axes = plt.subplots(4, 3, figsize=(12, 13))

row_labels = ['Highest Quality\n(Sharpest)', 'Needs Review', 'Highest Blur\n(Worst)', 'Lowest Blur Score\n(Most Blurry)']

for row_idx, (category, cameras) in enumerate(categories.items()):
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        
        if col_idx < len(cameras):
            camera, blur_score = cameras[col_idx]
            
            # find camera data
            camera_data = next(r for r in preview_results if r['camera'] == camera)
            
            # display middle frame
            if camera_data['frames_data']:
                middle_frame = camera_data['frames_data'][len(camera_data['frames_data'])//2]
                img = cv2.imread(str(middle_frame['path']))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f"{camera}\nB:{camera_data['avg_brightness']:.0f} S:{blur_score:.0f}", 
                            fontsize=10)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
        
        ax.axis('off')
        
        # row label
        if col_idx == 0:
            ax.text(-0.2, 0.5, row_labels[row_idx], 
                   transform=ax.transAxes, rotation=90,
                   ha='center', va='center', fontsize=12, weight='bold')

plt.suptitle('Quality Category Samples', fontsize=14)
plt.tight_layout()
plt.show()
```

## 📚 Quality Metrics Reference

### Brightness (Luminance)
Average pixel intensity across the image, measured on a 0-255 scale for 8-bit images.
- **Calculation**: Mean of grayscale pixel values
- **Reference**: [OpenCV Image Processing](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- **MLOps Context**: [Google Cloud - Image Quality Assessment](https://cloud.google.com/vision/docs/detecting-properties)

### Blur Score (Laplacian Variance)
Measures image sharpness by computing variance of the Laplacian operator output.
- **Higher values** = Sharper image (more edge detail)
- **Lower values** = Blurrier image (less edge detail)
- **Technical Details**: [Laplacian Operator - SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html)
- **Research Paper**: [Diatom autofocusing in brightfield microscopy](https://www.researchgate.net/publication/234073097_Diatom_autofocusing_in_brightfield_microscopy_A_comparative_study)

### Quartile-Based Outlier Detection
Statistical method using Q1 (25th percentile) and Q3 (75th percentile) to identify anomalies.
- **Pandas Documentation**: [DataFrame.quantile](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html)
- **Statistical Background**: [NIST - Quartiles](https://www.itl.nist.gov/div898/handbook/prc/section2/prc252.htm)

### Color Space Conversion (BGR to RGB)
OpenCV uses BGR format by default; conversion needed for matplotlib display.
- **OpenCV Reference**: [Color Space Conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- **Why BGR?**: [Historical reasons from Windows API](https://learnopencv.com/why-does-opencv-use-bgr-color-format/)

## 💾 Export Selection - Option 1: Automatic

Automatically select cameras that were flagged for quality issues.


```python
# automatic selection based on quality
# selected_cameras = []

# for video in needs_review:
#     selected_cameras.append(video['camera'])

# print(f"Auto-selected {len(selected_cameras)} cameras:")
# for cam in selected_cameras:
#     print(f"  {cam}")
```

## 💾 Export Selection - Option 2: Manual

Manually specify which cameras to process. Uncomment and edit the list.


```python
# manual selection (uncomment to use)
selected_cameras = ['ATL-1005', 'ATL-0972', 'ATL-0610', 'ATL-0973']

print(f"Manually selected {len(selected_cameras)} cameras")
```

## 💾 Export Selection - Option 3: All Cameras

Select all previewed cameras for individual processing.


```python
# select all cameras (uncomment to use)
# selected_cameras = [r['camera'] for r in preview_results]

# print(f"Selected all {len(selected_cameras)} cameras")
```

## 💾 Save Selection Queue

Save the selected cameras to a queue file for the individual preprocessing notebook.


```python
# save selection queue
selection_data = {
    'batch_date': CONFIG['PROCESS_DATE'],
    'selected_cameras': selected_cameras,
    'selection_criteria': 'quality_based',  # update based on option used
    'created_at': datetime.now().isoformat()
}

selection_file = CONFIG['OUTPUT_DIR'] / f"individual_queue_{CONFIG['PROCESS_DATE']}.json"
with open(selection_file, 'w') as f:
    json.dump(selection_data, f, indent=2)

print(f"✓ Saved {len(selected_cameras)} cameras to queue")
print(f"  File: {selection_file}")
```

## 📋 Batch Processing Summary

Generate final summary report of the batch preprocessing workflow.


```python
# batch processing summary
print("="*60)
print(f"BATCH PROCESSING SUMMARY - {CONFIG['PROCESS_DATE']}")
print("="*60)

print(f"\n📊 Processing Statistics:")
print(f"  Total cameras: {len(camera_dirs)}")
print(f"  Videos found: {len(target_videos)}")
print(f"  Videos previewed: {len(preview_results)}")
print(f"  Frames extracted: {len(preview_results) * PREVIEW_CONFIG['frames_per_video']}")

print(f"\n📈 Quality Overview:")
print(f"  Avg brightness: {df_quality['brightness'].mean():.1f}")
print(f"  Avg blur score: {df_quality['blur_score'].mean():.0f}")
print(f"  Videos flagged: {len(needs_review)}")
print(f"  High quality: {len(high_quality)}")

print(f"\n📁 Output Location:")
print(f"  {CONFIG['OUTPUT_DIR']}/")

print(f"\n✅ Next Steps:")
print(f"  1. Review quality samples above")
print(f"  2. Run individual preprocessing notebook")
print(f"  3. Load: {selection_file.name}")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

## 🎯 Final Video Selection

Review and finalize which videos to process. This creates the configuration file for the individual preprocessing notebook.


```python
# final video selection
print("Current selection:")
for i, cam in enumerate(selected_cameras):
    print(f"  {i+1}. {cam}")

print(f"\nTotal: {len(selected_cameras)} cameras")

# create final config
preprocessing_config = {
    'batch_date': CONFIG['PROCESS_DATE'],
    'videos_to_process': selected_cameras,
    'source_manifest': str(manifest_file),
    'quality_threshold': {
        'brightness_min': brightness_low,
        'brightness_max': brightness_high,
        'blur_min': blur_low
    },
    'created_at': datetime.now().isoformat()
}

# save config
config_file = CONFIG['OUTPUT_DIR'] / 'preprocessing_config.json'
with open(config_file, 'w') as f:
    json.dump(preprocessing_config, f, indent=2)

print(f"\n✓ Saved preprocessing config: {config_file}")
print(f"  Videos marked for processing: {len(selected_cameras)}")
```
