# SafeFall

SafeFall watches CCTV feeds, detects when someone falls, and sends the location to a supervisor's phone and Telegram. It runs on an ordinary laptop CPU.

It uses YOLO11 pose estimation, ByteTrack tracking, a fall rule based on torso angle and a sudden drop in height, and an optional small neural network. A Flask web dashboard shows six demo cameras, and an Android app receives push alerts.

## Run it on Windows

You need **Python 3.10** (64-bit) and **Git**. A GPU is not needed.

1. Clone the repository and open the folder:

```powershell
git clone https://github.com/OWNER/safefall.git
cd safefall
```

Use the address GitHub shows on the repository page. Keep the folder path free of `&`, because it breaks some Windows tools.

2. Create a virtual environment and install the packages. This downloads PyTorch, so it takes a while.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Create your settings file:

```powershell
copy .env.example .env
```

Open `.env` and set `SECRET_KEY` to any long random text. Leave the Telegram lines empty unless you have your own bot. Without one, falls are only logged.

4. Create a login (it asks for a password of at least 8 characters):

```powershell
.venv\Scripts\python.exe create_user.py yourname
```

5. Start SafeFall:

```powershell
.venv\Scripts\python.exe fall_detection.py
```

6. Open http://localhost:5000 in a browser and sign in. The six demo cameras in `cameras.json` start playing, and falls show up with a red DOWN label.

## Run the tests

```powershell
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest tests
```

All 70 tests should pass.

## Reproduce the evaluation

The UR Fall clips, their frame labels and the extracted keypoints are already in `datasets/urfall`.

```powershell
.venv\Scripts\python.exe evaluate_dataset.py
.venv\Scripts\python.exe train_fall_model.py
.venv\Scripts\python.exe evaluate_dataset.py --learned
```

`evaluate_dataset.py` tunes and scores the rules. `train_fall_model.py` trains the neural network on the odd-numbered clips, and `--learned` scores it on the held-out even-numbered clips. To extract keypoints again from the videos, run `extract_keypoints.py` first. That takes a long time on a CPU.

## Settings in `.env`

| Setting | Meaning |
|---|---|
| `DOWN_SECONDS` | How long someone must be down after a fall before alerting |
| `ESCALATE_SECONDS` | When to send a "still down" reminder |
| `FALL_MODEL` | `on` uses the trained network (more falls caught, more false alarms); `off` uses the rules |
| `ALERT_COOLDOWN_SECONDS` | A repeat fall on a camera within this time joins its open alert |
| `ALLOW_REGISTRATION` | Keep at `0` if the server is reachable from the internet |
| `CAMERAS_FILE` | Camera list; each camera can be a video file or an `rtsp://` address |

## Phone app (optional)

The Android app is in `mobile/`. Building it needs your own Expo and Firebase accounts; the steps are in [MOBILE_SETUP.md](MOBILE_SETUP.md). The web dashboard works without it.

## Project files

| Path | What it is |
|---|---|
| `fall_detection.py` | Main app: cameras, detection, dashboard |
| `detector.py` | Fall decision rules |
| `fall_model.py` | Optional temporal network |
| `api.py`, `storage.py` | Phone API and database tables |
| `alerts.py` | Push, Telegram and MQTT alerts |
| `mobile/` | Android app (Expo, React Native) |
| `paper/` | IEEE paper (Word and PDF) and figures |
| `DEMO_GUIDE.md` | Review demo plan |
| `*.pptx` | Review presentations |

## Data

The fall clips come from the UR Fall Detection Dataset (B. Kwolek and M. Kepski, 2014), which is meant for research use. Cite it if you use the results.
