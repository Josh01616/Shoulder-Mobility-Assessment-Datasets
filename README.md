# Shoulder Mobility Assessment System

A computer vision system for measuring shoulder range of motion (ROM) using pose estimation.

## Requirements

- Python 3.10 or later
- Webcam **or** a pre-recorded video file (MP4, AVI, MOV, MKV)

## Installation

```powershell
pip install -r requirements.txt
```

## Run the System

```powershell
.\run_app.bat
```

Or directly with Python:

```powershell
python main.py
```

## Core Files

| File / Folder | Purpose |
|---|---|
| `main.py` | Application entry point (GUI + session logic) |
| `src/` | Computer vision modules |
| `config.json` | Runtime thresholds and parameters |
| `assets/audio/` | Audio cue feedback files |
| `assets/guides/` | Exercise guide images |
| `data/videos/` | Place video files here for file-mode analysis |
| `logs/` | Session output (CSV, PDF, TXT reports) |
| `requirements.txt` | Python dependencies |

## Usage

1. Launch the application with `run_app.bat` or `python main.py`.
2. Select **Source**: Camera (live) or File (pre-recorded video).
3. Enter a **Participant ID** for output file naming.
4. Select the **Exercise** type: Abduction or Flexion.
5. Select the **Affected Side**: Right or Left.
6. Position the camera as instructed on screen.
7. Click **Start** to begin the calibration sequence.
8. Perform repetitions. The system tracks ROM, compensation patterns, and performance deterioration in real time.
9. Click **Stop** to end the session. Reports are saved automatically to `logs/`.

## Output Files

Each session generates three files in `logs/<participant_id>/`:

- `<id>_<exercise>_<date>.csv` — per-rep data including ROM, compensation flags, deterioration score
- `<id>_<exercise>_<date>_summary.txt` — session summary report
- `<id>_<exercise>_<date>.pdf` — formatted mobility assessment report

## Configuration

All thresholds are adjustable in `config.json`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `reps_per_set` | 10 | Target repetitions per set |
| `total_sets` | 2 | Number of sets per session |
| `rom_classification` | 150° | Minimum peak angle for correct ROM |
| `cwema_alpha_base` | 0.4 | CW-EMA landmark smoothing factor |
| `calibration_duration_sec` | 10.0 | Calibration window duration |

## Supported Exercises

| Exercise | Camera Position |
|---|---|
| **Abduction** | Frontal view — patient faces camera |
| **Flexion** | Lateral view — affected side toward camera |
