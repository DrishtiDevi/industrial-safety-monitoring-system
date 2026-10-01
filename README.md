# AI-Based Industrial Safety Monitoring System

An AI-based computer vision system for real-time industrial safety monitoring using PPE detection, worker pose estimation, and zone-specific safety rules.

## Project Overview

This project aims to assist industrial safety monitoring by analyzing video feeds and identifying PPE violations and unsafe worker activities.

The system combines object detection, pose estimation, zone-based safety logic, and automated alert logging.

## Key Features

- Real-time PPE detection using a custom YOLO model
- Detection of multiple PPE categories
- Zone-specific safety monitoring for electrical and conveyor areas
- Electrical-zone PPE compliance checking
- Conveyor movement detection
- Conveyor/fencing safety monitoring
- Worker pose estimation for stair and ladder safety
- 3-point contact checking
- Detection of unsafe activities such as phone usage while climbing
- Automated safety alerts
- Safety violation logging with timestamps in CSV format

## System Workflow

Video Input  
→ Object Detection  
→ PPE and Equipment Detection  
→ Worker Pose Estimation  
→ Zone Identification  
→ Safety Rule Evaluation  
→ Safety Alert  
→ CSV Violation Logging

## Technologies Used

- Python
- OpenCV
- YOLO / Ultralytics
- Pose Estimation
- NumPy
- Computer Vision
- CSV-based logging

## Safety Monitoring Logic

### PPE Monitoring

The system detects PPE and evaluates whether required safety equipment is present.

### Electrical Zone

When an electrical work zone is identified, the system applies electrical-area-specific PPE requirements.

### Conveyor Zone

The system monitors conveyor movement and worker interaction with the conveyor/fencing area to identify potentially unsafe situations.

### Stair/Ladder Safety

Pose estimation is used to analyze worker posture and check 3-point contact during stair or ladder movement.

### Alert Logging

Detected safety violations are recorded with:

- Timestamp
- Alert type
- Alert message

## Project Structure

```text
industrial-safety-monitoring-system/
│
├── industrial_safety_monitor.py
├── test.py
├── README.md
└── .gitignore

### Then commit it

At the bottom of GitHub, use:

**Commit message:**
```text
Update project README
