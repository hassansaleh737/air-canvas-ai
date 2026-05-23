# Air Canvas AI

A real-time two-hand air drawing system using Computer Vision.

The project tracks both hands through the webcam and allows users to draw in the air using finger gestures. Each hand can be controlled independently with different colors, eraser mode, brush size control, shape modes, handwriting mode, and clean image export.

## Features

- Real-time hand tracking
- Two-hand drawing support
- Independent left/right hand control
- Gesture-based drawing
- Shape modes:
  - Free Draw
  - Line
  - Rectangle
  - Circle
  - Triangle
- Handwriting mode
- Brush size control
- Eraser mode
- Save output images without overwriting old files
- Professional desktop UI for live demos

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

## Controls

| Key | Function |
|---|---|
| `L` | Select left hand |
| `R` | Select right hand |
| `B` | Select both hands |
| `X` | Toggle selected hand ON/OFF |
| `M` | Switch drawing mode |
| `T` | Toggle handwriting mode |
| `+` / `=` | Increase brush size |
| `-` | Decrease brush size |
| `1` | Red color |
| `2` | Blue color |
| `3` | Green color |
| `4` | Yellow color |
| `E` | Eraser |
| `C` | Clear canvas |
| `S` | Save drawing |
| `Q` | Quit |

## Gestures

| Gesture | Action |
|---|---|
| Index finger only | Draw / place shape |
| Index + middle fingers | Move without drawing |
| Fingers down | Stop drawing / commit shape |


## Install dependencies
pip install -r requirements.txt

## Run the Project
python main.py

## Project Structure

```text
air_drawing_project/
│
├── main.py
├── hand_tracker.py
├── drawing_canvas.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── output/


