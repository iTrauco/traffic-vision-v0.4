# Multi-Object Tracking Computer Vision System Design Proposal

> **Note**: This is a personal repository for developing a systematic approach to data science, AI, and ML system engineering using information systems analysis and design principles. The source of truth for work completed in this repository is located in the notebook within the `notebooks/` directory, not in this README.

> **Branch Scope**: All work for multi-obect tracking will is scoped this branch, `traffic-vision/multi-object-tracking` only. 

> **Clean Rebuild No. 3**: This is a clean rebuild of the traffic-vision systems analysis and design proposal originally developed in [`scenario06/traffic-vision/multi-object-tracking`](https://github.com/iTrauco/data-science-sad/tree/scenario06/traffic-vision/multi-object-tracking), incorporating refined workflow architecture.

## Table of Contents
- [Scenario Overview](#scenario-overview)
- [Proposal Framework](#proposal-framework)
- [System Components](#system-components)
  - [Infrastructure Systems](#infrastructure-systems)
  - [MLOps Workflows](#mlops-workflows-abstracted)
- [Implementation Considerations](#implementation-considerations)
  - [Core System Components](#core-system-components)
- [Reproducibility Framework](#reproducibility-framework)
  - [Environment Setup](#environment-setup)

## Scenario Overview
A transportation research team needs infrastructure to capture and manage traffic camera footage from multiple highway locations.

They require HPC scripts for recording and encoding livestream video feeds, along with a local storage system for archiving recorded videos with proper date/location cataloging. Additionally, they need a data access layer that enables retrieval of stored videos for downstream processing.

They have an HPC workstation with GPU available and need to design the storage architecture, stream capture system, and computational resource allocation to support their requirements.

This proposal outlines the systems infrastructure design for video stream recording, storage management, data access services, and HPC environment configuration to support their video analysis requirements.

## Proposal Framework
This project presents a systems analysis and design approach to the multi-object tracking computer vision challenge. The proposal follows established information systems design principles, addressing both infrastructure systems and MLOps methodologies:

**Infrastructure Architecture:**
- **Video Stream Processing**: HPC scripts and computational requirements for livestream capture and encoding
- **Storage System Design**: Local storage architecture, cataloging, and data access services
- **Computing Environment**: GPU/CPU resource allocation, job scheduling, and system monitoring
- **Infrastructure Performance**: System metrics, capacity planning, and reliability

**MLOps Workflows and Frameworks:**
- **Data Preprocessing**: Frame extraction and transformation processes
- **Annotation Framework**: Labeling tools, formats, and quality control
- **Model Development**: Architecture selection, training, and optimization strategies
- **Model Evaluation**: Performance metrics, validation, and results analysis

## System Components

> **Note**: In this infrastructure-focused design, elements typically considered foundational (like compute environment) are treated as active system components requiring design and management.

### Infrastructure Systems:
1. **Stream Recording System**
   - HPC scripts for livestream capture
   - Video encoding and compression
   - Automated save to local storage

2. **Storage Management System**
   - Local storage architecture
   - Date/location based cataloging
   - Archive organization and retention

3. **Data Access Service**
   - Query system for date/location filtering
   - Abstraction layer for MLOps access

4. **HPC Computing Environment**
   - GPU resource allocation
   - CPU/memory management
   - Job scheduling system

5. **System Monitoring**
   - Storage capacity tracking
   - Stream recording health checks
   - Resource utilization metrics

### MLOps Workflows (Abstracted):
1. **Data Preprocessing** - Frame extraction, transformations
2. **Data Annotation** - Labeling tools and processes
3. **Model Development** - Training and optimization
4. **Model Evaluation** - Performance metrics and validation

Access through notebook interface or REST API, isolated from infrastructure complexity.

## Implementation Considerations

The infrastructure implementation focuses on:
- HPC workstation setup
- Video stream recording and cataloging system
- Data storage and retrieval
- Resource management
- System monitoring

### Core System Components

The infrastructure requires these essential components:
- **Stream Manager**: Video recording and encoding system
- **Storage System**: Local data storage and cataloging
- **Data Access Layer**: File organization and retrieval interface
- **Compute Environment Architecture**: HPC workstation, GPU resources, and environment management
- **Monitoring System**: Performance and health tracking

## Reproducibility Framework
### Environment Setup

This project uses a Conda environment to manage dependencies for reproducible analysis. Follow these steps to set up the environment:

#### Prerequisites
- Anaconda or Miniconda installed on your system
- Git for cloning the repository

#### Setup Instructions

1. Clone the repository and switch to the feature branch:
   ```bash
   git clone https://github.com/iTrauco/traffic-vision.git
   cd traffic-vision
   git checkout traffic-vision/multi-object-tracking
   ```

2. Create the Conda environment:
   ```bash
    conda create -n traffic-vision-multi-object-tracking python=3.11 -y

   ```

3. Activate the environment:
   ```bash
   conda activate traffic-vision-multi-object-tracking
   ```

4. Install baseline packages:
   ```bash
   conda install -c conda-forge jupyter numpy pandas matplotlib seaborn scikit-learn opencv -y
   ```

5. Install deep learning and computer vision packages:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install ultralytics supervision
   ```

6. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

7. Access the notebook in your browser via the URL displayed in the terminal

#### Environment Details

The environment includes essential data science and computer vision packages:
- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- [Jupyter Notebook](https://jupyter.org/documentation)
- [pandas](https://pandas.pydata.org/docs/) & [numpy](https://numpy.org/doc/stable/) for data manipulation
- [matplotlib](https://matplotlib.org/stable/index.html) & [seaborn](https://seaborn.pydata.org/) for visualization
- [scikit-learn](https://scikit-learn.org/stable/documentation.html) for traditional ML algorithms
- [OpenCV](https://docs.opencv.org/4.x/) for image and video processing
- [PyTorch](https://pytorch.org/docs/stable/index.html) for deep learning model development
- [Ultralytics](https://docs.ultralytics.com/) for YOLO object detection
- [Supervision](https://supervision.roboflow.com/) for object tracking utilities

#### Environment Management

For collaborators who enhance the environment with additional packages:

```bash
# Export the updated environment
conda activate traffic-vision-multi-object-tracking
conda env export > environment.yml
```