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

## 📑 Table of Contents (Auto-Generated)

This section will automatically generate a table of contents for your research notebook once you run the **Generate TOC** function. The table of contents will help you navigate through your data collection, analysis, and findings as your citizen science project develops.

➡️ **Do not edit this cell manually. It will be overwritten automatically.**


# 🧪 Experiment Preprocessing - [VERSION]

## 🎯 Purpose
This notebook preprocesses GDOT traffic camera videos specifically for machine learning experiments. It follows the same workflow as general preprocessing but with experiment-specific parameters.

## 📋 Context
- **Data Source**: Raw GDOT traffic camera recordings
- **Output**: Frames optimized for experiment workflows
- **Destination**: `data/preprocessing/experiments/`
- **Next Steps**: Annotation → Training → Evaluation

## 🔄 Workflow Overview
1. Video ingestion from recordings
2. Frame extraction (experiment-specific rate)
3. Quality control (experiment thresholds)
4. Spatial transformations
5. Export to experiments directory

## 📚 Notebook Structure
- **Setup**: Environment and dependencies
- **Processing**: Experiment-specific preprocessing
- **Export**: Organized output for annotation

*Processing completed: [DATE] | Version: [VERSION]*

## 🔧 Experiment Configuration

This cell defines parameters specific to experiment preprocessing. Values are optimized for machine learning workflows rather than general analysis.

#### Problem Statement
- 🎯 **Objective**: Count vehicles in GDOT traffic camera feeds
- 🎯 **Challenge**: Detect and count cars, trucks, buses in various conditions
- 🎯 **Approach**: Preprocess frames specifically for vehicle detection training

#### Target Parameters
- 🎯 **VIDEO_ID**: Specific camera to process
- 🎯 **BATCH_DATE**: Recording date (YYYYMMDD format)
- 🎯 **EXPERIMENT_TYPE**: 'car_counting'

#### Path Configuration  
- 📁 **INPUT_BASE**: Root directory for video recordings
- 📁 **OUTPUT_BASE**: Root for experiment preprocessing (`data/preprocessing/experiments`)

#### Experiment-Specific Settings
- 📊 **FRAMES_TO_EXTRACT**: Higher count for training data (1000+)
- 📊 **SAMPLE_RATE**: Denser sampling for better coverage (every 5 frames)
- 🔍 **INCLUDE_EDGE_CASES**: Keep some lower quality frames for model robustness
- 🔍 **MIN_VEHICLES_PER_FRAME**: Prefer frames with vehicles present

The following cell initializes the experiment configuration with parameters optimized for vehicle counting.




```python
# experiment configuration parameters
from pathlib import Path

CONFIG = {
    # target parameters
    'VIDEO_ID': 'ATL-1005',  # 🎯 camera to process
    'BATCH_DATE': '20250620',  # 🎯 date from batch analysis
    'TARGET_HOUR': 12,  # 🎯 target hour (noon)
    'EXPERIMENT_TYPE': 'car_counting',  # 🎯 experiment type
    
    # path configuration  
    'INPUT_BASE': Path.home() / 'traffic-recordings',  # 📁 video source
    'OUTPUT_BASE': Path('../../data/preprocessing/experiments'),  # 📁 experiment output
    
    # processing settings
    'FRAMES_TO_EXTRACT': 1000,  # 📊 total frames to extract
    'SAMPLE_RATE': 5,  # 📊 extract every Nth frame
    
    # quality thresholds (relaxed for experiments)
    'QUALITY_THRESHOLD': {
        'brightness_min': 90,  # 🔍 minimum brightness
        'brightness_max': 130,  # 🔍 maximum brightness  
        'blur_min': 2500  # 🔍 minimum blur score
    },
    
    # video settings
    'PREFERRED_CODEC': 'mp4v',  # 🎥 primary codec
    'FALLBACK_CODECS': ['h264', 'xvid'],  # 🎥 alternatives
    'MAX_FRAME_WIDTH': 1920,  # 🎥 max width
    'MAX_FRAME_HEIGHT': 1080,  # 🎥 max height
    'JPEG_QUALITY': 95  # 🎥 output quality (0-100)
}

# derived paths
date_formatted = f"{CONFIG['BATCH_DATE'][:4]}-{CONFIG['BATCH_DATE'][4:6]}-{CONFIG['BATCH_DATE'][6:8]}"
CONFIG['OUTPUT_DIR'] = CONFIG['OUTPUT_BASE'] / CONFIG['EXPERIMENT_TYPE'] / date_formatted / CONFIG['VIDEO_ID']
CONFIG['VIDEO_DIR'] = CONFIG['INPUT_BASE'] / CONFIG['VIDEO_ID'] / date_formatted


```

---

*End of Experiment Configuration*

---

## 🔧 Environment Setup

This cell establishes the preprocessing environment with the same core libraries as general preprocessing, plus experiment-specific additions.

1. **Core Libraries**
   - OpenCV for video processing
   - NumPy for array operations
   - Pandas for data organization
   - Logging for process tracking

2. **Same Helper Functions**
   - calculate_brightness()
   - calculate_blur_score()
   - get_video_metadata()

3. **Experiment Additions**
   - Vehicle detection helpers
   - Frame selection priorities
   - Experiment metadata tracking

The following cell imports libraries and initializes the preprocessing environment.

🟢 **IMPLEMENTATION COMPLETE** 🟢


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
print(f"✓ OpenCV version: {cv2.__version__}")
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
        metadata['codec'] = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
    return metadata

# create output directory
CONFIG['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)

print(f"\n✓ Environment setup complete")
print(f"  Output directory created: {CONFIG['OUTPUT_DIR']}")
```

---

*End of Environment Setup*

---

## 🔄 Progress Tracking & Checkpoint System

The following cells implement simple progress tracking and checkpoint functionality to:

1. **Track Processing Progress**
   - Monitor experiment preprocessing status
   - Count frames extracted and processed
   - Display elapsed time

2. **Enable Restart Capability**
   - Save progress after each stage
   - Automatically resume from last checkpoint
   - Maintain experiment metadata

This ensures experiment preprocessing can be resumed if interrupted.


## 💾 Initialize Checkpoint and Progress Tracking Functions

This module establishes checkpoint and progress tracking for experiment preprocessing. Functions track which videos have been processed for experiments and enable recovery from interruptions.

The following cell sets up checkpoint functionality specific to experiment preprocessing.

🟢 **IMPLEMENTATION COMPLETE** 🟢


```python
# checkpoint and progress tracking
import json
import time
from datetime import datetime

CHECKPOINT_FILE = CONFIG['OUTPUT_DIR'] / "experiment_checkpoint.json"
start_time = time.time()

def load_checkpoint():
    """Load previous progress if it exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            print(f"✓ Loaded checkpoint: {checkpoint.get('stage', 'unknown')} stage")
            return checkpoint
    return {
        "stage": "started",
        "processed": {}, 
        "start_time": datetime.now().isoformat()
    }

def save_checkpoint(checkpoint):
    """Save current progress"""
    checkpoint['last_updated'] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def update_progress(stage, details=None):
    """Update and save progress"""
    checkpoint = load_checkpoint()
    checkpoint['stage'] = stage
    if details:
        checkpoint['processed'][stage] = details
    save_checkpoint(checkpoint)
    
    elapsed = time.time() - start_time
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stage: {stage}")
    print(f"Elapsed: {elapsed/60:.1f}min")

# Initialize checkpoint
checkpoint = load_checkpoint()
print(f"Ready to process experiment data. Checkpoint initialized.")
```

## 📹 Video Ingestion & Cataloging

This module loads video files for experiment preprocessing with focus on selecting videos that contain vehicle activity.

1. **Find Video Files**
   - Locate videos for specified camera and date
   - Parse timestamps from filenames
   - Select videos during high-traffic periods

2. **Prioritize for Experiments**
   - Prefer daytime hours (better visibility)
   - Avoid night/dawn/dusk for initial experiments
   - Focus on peak traffic times

3. **Extract Metadata**


```python
# video ingestion and cataloging
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

# find closest to noon
target_minutes = CONFIG['TARGET_HOUR'] * 60  # 720 minutes
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
time_str = closest_video.stem.split('_')[2]
print(f"Selected: {closest_video.name}")
print(f"  Starts at: {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}")

# update checkpoint
update_progress('video_selected', {'video': closest_video.name})
```

## 🎞️ Frame Extraction

This module samples frames from video sequences with parameters optimized for experiment training data.

1. **Experiment-Specific Extraction**
   - Higher frame count for training diversity
   - Denser sampling rate (every 5 frames vs 15)
   - Extract from multiple time periods

2. **Quality Over Compression**
   - Higher JPEG quality for annotation clarity
   - Full resolution preservation
   - No aggressive downsampling

3. **Metadata Tracking**
   - Frame timestamp mapping
   - Source video reference
   - Frame sequence numbering

The following cell extracts frames using experiment-optimized parameters.




```python
# frame extraction
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
        
        frames_extracted += 1
        if frames_extracted % 100 == 0:
            print(f"  Extracted {frames_extracted}/{CONFIG['FRAMES_TO_EXTRACT']} frames")
    
    frame_index += 1

cap.release()

CONFIG['frames_dir'] = frames_dir
CONFIG['frames_extracted'] = frames_extracted

print(f"\n✓ Extracted {frames_extracted} frames to {frames_dir}")

# update checkpoint
update_progress('frames_extracted', {'count': frames_extracted})
```

## 🔍 Image Quality Control

This module filters frames with experiment-specific quality thresholds that balance data quality with training diversity.

1. **Relaxed Thresholds**
   - Slightly lower brightness bounds (include dawn/dusk)
   - Accept moderate blur (real-world conditions)
   - Keep edge cases for model robustness

2. **Vehicle Presence Priority**
   - Prioritize frames with motion
   - Check for object-like shapes
   - Balance empty vs occupied frames

3. **Training Set Diversity**
   - Include various lighting conditions
   - Keep some challenging frames
   - Document quality distribution

The following cell applies quality filtering optimized for ML training diversity.


```python
# image quality control
import matplotlib.pyplot as plt

print("Image Quality Control")

# get all extracted frames
frame_files = sorted(CONFIG['frames_dir'].glob("frame_*.jpg"))
print(f"Checking quality of {len(frame_files)} frames")

quality_results = []
good_frames = []
poor_frames = []

for frame_path in frame_files:
    # read frame
    frame = cv2.imread(str(frame_path))
    if frame is None:
        poor_frames.append(frame_path)
        continue
    
    # calculate metrics
    brightness = calculate_brightness(frame)
    blur_score = calculate_blur_score(frame)
    
    # check thresholds
    passes_quality = (
        brightness >= CONFIG['QUALITY_THRESHOLD']['brightness_min'] and
        brightness <= CONFIG['QUALITY_THRESHOLD']['brightness_max'] and
        blur_score >= CONFIG['QUALITY_THRESHOLD']['blur_min']
    )
    
    result = {
        'frame': frame_path.name,
        'brightness': brightness,
        'blur_score': blur_score,
        'passes': passes_quality
    }
    quality_results.append(result)
    
    if passes_quality:
        good_frames.append(frame_path)
    else:
        poor_frames.append(frame_path)

# save quality report
quality_df = pd.DataFrame(quality_results)
quality_df.to_csv(CONFIG['OUTPUT_DIR'] / 'quality_report.csv', index=False)

CONFIG['good_frames'] = good_frames
CONFIG['quality_results'] = quality_df

print(f"\nResults:")
print(f"  Good frames: {len(good_frames)}")
print(f"  Poor frames: {len(poor_frames)}")
print(f"  Pass rate: {len(good_frames)/len(frame_files)*100:.1f}%")

# update checkpoint
update_progress('quality_control', {'good': len(good_frames), 'poor': len(poor_frames)})
```

## 📐 Spatial Transformations

This module resizes frames if needed. Same as general preprocessing.

The following cell applies standard transformations.




```python
# spatial transformations
print("Spatial Transformations")

frames_to_transform = CONFIG['good_frames']
print(f"Transforming {len(frames_to_transform)} frames")

# create transformed directory
transformed_dir = CONFIG['OUTPUT_DIR'] / 'transformed'
transformed_dir.mkdir(exist_ok=True)

# target dimensions
target_width = CONFIG['MAX_FRAME_WIDTH']
target_height = CONFIG['MAX_FRAME_HEIGHT']

transformed_frames = []

for frame_path in frames_to_transform:
    # read frame
    frame = cv2.imread(str(frame_path))
    height, width = frame.shape[:2]
    
    # resize if needed
    if width > target_width or height > target_height:
        scale = min(target_width/width, target_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # save transformed frame
    output_path = transformed_dir / frame_path.name
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
    transformed_frames.append(output_path)

CONFIG['transformed_frames'] = transformed_frames
print(f"\n✓ Completed spatial transformations")

# update checkpoint
update_progress('spatial_transformations', {'count': len(transformed_frames)})
```

## 🎨 Color Space Normalization

Convert BGR to RGB. Same as general preprocessing.

The following cell normalizes color space.



```python
# color space normalization
print("Color Space Normalization")

frames_to_normalize = CONFIG['transformed_frames']
print(f"Normalizing {len(frames_to_normalize)} frames")

# create normalized directory
normalized_dir = CONFIG['OUTPUT_DIR'] / 'normalized'
normalized_dir.mkdir(exist_ok=True)

normalized_frames = []

for frame_path in frames_to_normalize:
    # read frame (BGR)
    frame_bgr = cv2.imread(str(frame_path))
    
    # convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # save as RGB
    output_path = normalized_dir / frame_path.name
    cv2.imwrite(str(output_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
    normalized_frames.append(output_path)

CONFIG['normalized_frames'] = normalized_frames
print(f"\n✓ Color normalization complete")

# update checkpoint
update_progress('color_normalization', {'count': len(normalized_frames)})
```

## ⏱️ Temporal Downsampling

For experiments, we keep ALL frames with vehicles (no downsampling). This maximizes training data.

The following cell identifies and keeps frames with motion.


```python
# temporal downsampling (keep frames with motion)
print("Temporal Downsampling")

frames_to_analyze = CONFIG['normalized_frames']
print(f"Analyzing {len(frames_to_analyze)} frames for motion")

# motion detection
motion_scores = []

for i in range(len(frames_to_analyze) - 1):
    frame1 = cv2.imread(str(frames_to_analyze[i]), cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(str(frames_to_analyze[i+1]), cv2.IMREAD_GRAYSCALE)
    
    # calculate difference
    diff = cv2.absdiff(frame1, frame2)
    motion_score = np.mean(diff)
    
    motion_scores.append({
        'frame': frames_to_analyze[i].name,
        'motion_score': motion_score,
        'has_motion': motion_score > 5.0
    })

# for experiments, keep ALL frames with motion
motion_df = pd.DataFrame(motion_scores)
frames_with_motion = motion_df[motion_df['has_motion']]['frame'].tolist()

# create downsampled directory
downsampled_dir = CONFIG['OUTPUT_DIR'] / 'downsampled'
downsampled_dir.mkdir(exist_ok=True)

# copy all frames with motion
downsampled_frames = []
for frame_name in frames_with_motion:
    src = normalized_dir / frame_name
    dst = downsampled_dir / frame_name
    frame = cv2.imread(str(src))
    cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']])
    downsampled_frames.append(dst)

CONFIG['downsampled_frames'] = downsampled_frames
```


```python
# data organization
print("Data Organization")

# compile metadata
metadata = {
    'camera_id': CONFIG['VIDEO_ID'],
    'batch_date': CONFIG['BATCH_DATE'],
    'experiment_type': CONFIG['EXPERIMENT_TYPE'],
    'source_video': CONFIG['selected_video'].name,
    'processing_timestamp': datetime.now().isoformat(),
    'parameters': {
        'target_hour': CONFIG['TARGET_HOUR'],
        'frames_extracted': CONFIG['FRAMES_TO_EXTRACT'],
        'sample_rate': CONFIG['SAMPLE_RATE'],
        'quality_thresholds': CONFIG['QUALITY_THRESHOLD'],
        'jpeg_quality': CONFIG['JPEG_QUALITY']
    },
    'processing_summary': {
        'frames_extracted': CONFIG['frames_extracted'],
        'frames_good_quality': len(CONFIG['good_frames']),
        'frames_with_motion': len(CONFIG['downsampled_frames'])
    }
}

# save metadata
metadata_path = CONFIG['OUTPUT_DIR'] / 'metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

# save frame inventory
frame_inventory = []
for frame_path in CONFIG['downsampled_frames']:
    frame_inventory.append({
        'frame_name': frame_path.name,
        'path': str(frame_path)
    })

inventory_df = pd.DataFrame(frame_inventory)
inventory_path = CONFIG['OUTPUT_DIR'] / 'frame_inventory.csv'
inventory_df.to_csv(inventory_path, index=False)

CONFIG['metadata'] = metadata
CONFIG['inventory'] = inventory_df

print(f"\n✓ Data organization complete")
print(f"  Metadata saved: {metadata_path}")
print(f"  Frame inventory: {inventory_path}")

# update checkpoint
update_progress('data_organization', {'frames_cataloged': len(frame_inventory)})
```

## 💾 Export & Storage

Creates final summary and confirms frames are ready for annotation.

The following cell saves experiment preprocessing summary.

🚧 **IMPLEMENTATION PENDING** 🚧


```python
# export and storage
print("Export & Storage Summary")

# create summary report
summary = {
    'experiment_preprocessing_complete': datetime.now().isoformat(),
    'experiment_type': CONFIG['EXPERIMENT_TYPE'],
    'camera': CONFIG['VIDEO_ID'],
    'video_processed': CONFIG['selected_video'].name,
    'frames_ready_for_annotation': len(CONFIG['downsampled_frames']),
    'annotation_directory': str(CONFIG['OUTPUT_DIR'] / 'downsampled'),
    'processing_stages': {
        '1_extracted': CONFIG['frames_extracted'],
        '2_quality_filtered': len(CONFIG['good_frames']),
        '3_transformed': len(CONFIG['transformed_frames']),
        '4_normalized': len(CONFIG['normalized_frames']),
        '5_motion_filtered': len(CONFIG['downsampled_frames'])
    }
}

# save summary
summary_path = CONFIG['OUTPUT_DIR'] / 'experiment_preprocessing_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nExperiment Preprocessing Complete")
print(f"Camera: {CONFIG['VIDEO_ID']}")
print(f"Frames ready: {len(CONFIG['downsampled_frames'])}")
print(f"Location: {CONFIG['OUTPUT_DIR'] / 'downsampled'}")
print(f"Summary: {summary_path}")

# update checkpoint
update_progress('complete', summary)
```

## 🏷️ Prepare Frames for CVAT Annotation

This module packages the preprocessed frames for upload to CVAT (Computer Vision Annotation Tool).

The following cell creates a zip file of frames for CVAT import.

🚧 **IMPLEMENTATION PENDING** 🚧


```python
# prepare frames for CVAT
import zipfile

frames_to_annotate = CONFIG['OUTPUT_DIR'] / 'downsampled'
frame_files = sorted(frames_to_annotate.glob('*.jpg'))

# create annotation directory
annotation_dir = CONFIG['OUTPUT_DIR'].parent.parent.parent / 'annotations' / CONFIG['EXPERIMENT_TYPE'] / date_formatted / CONFIG['VIDEO_ID']
annotation_dir.mkdir(parents=True, exist_ok=True)

# zip frames for CVAT
zip_path = annotation_dir / f"{CONFIG['VIDEO_ID']}_frames_for_cvat.zip"
with zipfile.ZipFile(zip_path, 'w') as zf:
    for frame in frame_files:
        zf.write(frame, frame.name)

print(f"✓ Created CVAT upload file: {zip_path}")
print(f"  Contains {len(frame_files)} frames")
```

## 🐳 CVAT Docker Setup

Run CVAT locally using Docker:

```bash
# Start CVAT
docker run -d --name cvat -p 8080:8080 openvino/cvat

# Access at http://localhost:8080
# Default: username=admin, password=admin
```

1. Create new task
2. Upload the zip file
3. Set labels: car, truck, bus
4. Annotate with bounding boxes
5. Export as CVAT XML

🚧 **MANUAL PROCESS** 🚧

## 📥 Convert CVAT Annotations

This module converts CVAT XML output to training CSV format.

The following cell processes CVAT annotations.




```python
# convert CVAT annotations to CSV
import xml.etree.ElementTree as ET
import pandas as pd

# path to CVAT XML export (update after annotation)
cvat_xml_path = annotation_dir / 'annotations.xml'

if cvat_xml_path.exists():
    # parse CVAT XML
    tree = ET.parse(cvat_xml_path)
    root = tree.getroot()
    
    annotations = []
    
    # extract annotations
    for image in root.findall('.//image'):
        frame_name = image.get('name')
        
        for box in image.findall('.//box'):
            annotations.append({
                'frame': frame_name,
                'label': box.get('label'),
                'xtl': float(box.get('xtl')),
                'ytl': float(box.get('ytl')),
                'xbr': float(box.get('xbr')),
                'ybr': float(box.get('ybr'))
            })
    
    # save as CSV
    annotations_df = pd.DataFrame(annotations)
    csv_path = annotation_dir / 'annotations.csv'
    annotations_df.to_csv(csv_path, index=False)
    
    print(f"✓ Converted {len(annotations)} annotations")
    print(f"  Saved to: {csv_path}")
    
    # summary
    print(f"\nAnnotation summary:")
    print(annotations_df['label'].value_counts())
else:
    print("⚠️ No CVAT XML file found. Complete annotation first.")
```

## ✅ Annotation Workflow Complete

At this point:
1. Frames are preprocessed and packaged
2. CVAT annotation can be performed
3. Annotations are converted to CSV format

**Next Steps:**
- Use annotations.csv for training
- Create v1_03_experiment_training_template.working.ipynb

🟢 **PREPROCESSING & ANNOTATION COMPLETE** 🟢
