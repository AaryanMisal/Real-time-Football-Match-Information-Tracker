# Real-time Football Match Information Tracker

This project implements an end to end computer vision pipeline for football analytics using broadcast match video. The system performs player and ball detection, multi object tracking, camera homography estimation, and spatio temporal analysis to generate quantitative match statistics and qualitative visualizations.

The pipeline outputs structured analytics in CSV format, including player level, team level, and ball related statistics, as well as an annotated version of the input video visualizing tracking, possession, speeds, and distances covered.

## Features

- Player and ball detection using YOLOv5
- Multi object tracking using ByteTrack
- Camera homography and pitch projection
- Player distance and speed estimation
- Team possession analysis and spatial breakdown
- Ball trajectory and distance analytics
- Event extraction from spatio temporal interactions
- Annotated output video with visual overlays

## Setup

Create a Python virtual environment:

python -m venv venv
source venv/bin/activate        # Linux or macOS
venv\Scripts\activate           # Windows

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

## Input
Make sure to add Input videos in the directory 'input_videos'. Then add this video to config.yaml so that the project can run on this video to generate the required annotated video and stats csv files.

## Models
In models directory, we have put our designated Yolov5 model. Make sure to attach your required model here. We could not attach the model here due to size constraints.

## Running the Pipeline

To run the full analytics pipeline on a match video, execute:

python run_pipeline.py --config configs/config.yaml

The configuration file controls input paths, model settings, and output directories.

## Outputs

After execution, the pipeline produces CSV files containing player, team, ball, and event analytics, along with an annotated video of the input match with detections, tracking identities, team assignments, possession indicators, and movement statistics. All outputs are saved under the outputs directory.

## Notes

The pipeline operates on standard broadcast soccer videos and does not require a specialized multi camera setup. Performance depends on video resolution and available hardware.

## Acknowledgements

This project builds on open source research and tools including YOLOv5 for object detection and ByteTrack for multi object tracking.

## License

This project is intended for academic and educational use.
