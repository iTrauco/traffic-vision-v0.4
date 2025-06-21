#!/usr/bin/env python3
"""
Generate individual preprocessing notebooks from batch analysis config.
Run from project root: python scripts/generate_preprocessing_notebooks.py
"""

import json
import re
from pathlib import Path

def find_latest_config():
    """Find the most recent preprocessing_config.json"""
    batch_dir = Path("data/preprocessing/batch_analysis")
    date_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()], reverse=True)
    
    for dir in date_dirs:
        config_path = dir / "preprocessing_config.json"
        if config_path.exists():
            return config_path
    
    raise FileNotFoundError("No preprocessing_config.json found")

def create_notebook_for_video(video_id, config, template_path, output_dir):
    """Create customized notebook for a specific video"""
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Replace VIDEO_ID
    content = re.sub(r"'VIDEO_ID': '[^']*'", f"'VIDEO_ID': '{video_id}'", content)
    
    # Replace BATCH_DATE
    content = re.sub(r"'BATCH_DATE': '[^']*'", f"'BATCH_DATE': '{config['batch_date']}'", content)
    
    # Replace quality thresholds
    content = re.sub(r"'brightness_min': [\d.]+", f"'brightness_min': {config['quality_threshold']['brightness_min']:.2f}", content)
    content = re.sub(r"'brightness_max': [\d.]+", f"'brightness_max': {config['quality_threshold']['brightness_max']:.2f}", content)
    content = re.sub(r"'blur_min': [\d.]+", f"'blur_min': {config['quality_threshold']['blur_min']:.2f}", content)
    
    # Save notebook with .working.ipynb extension
    output_path = output_dir / f"preprocessing_{video_id}.working.ipynb"
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"  Created: preprocessing_{video_id}.working.ipynb")

def main():
    # Find and load config
    config_path = find_latest_config()
    print(f"Using config: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    date_str = config['batch_date']
    date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    output_dir = Path("notebooks/MLOps/analysis") / date_formatted
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the specified template
    template_path = Path("notebooks/Templates/v2_01_preprocessing_template.working.ipynb")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    print(f"Videos to process: {config['videos_to_process']}")
    print(f"Output directory: {output_dir}\n")
    
    # Generate notebooks
    for video_id in config['videos_to_process']:
        create_notebook_for_video(video_id, config, template_path, output_dir)
    
    print(f"\n✓ Generated {len(config['videos_to_process'])} notebooks")

if __name__ == "__main__":
    main()