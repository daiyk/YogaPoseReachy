---
title: Reachy Mini Pose Coach
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Reachy Mini Pose Coach

A voice-driven yoga and pose coach for the [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot.
Show Reachy a reference pose, step in front of its camera, and talk to it. Reachy compares your body with the
reference, tells you what to adjust ("raise your left arm about 20°"), and congratulates you when you match.

- **Talk naturally.** Ask "how am I doing?", say "start practice" or "end practice", or "exit the app" to stop.
- **Live scoring.** A 0–100 score, the main mismatches, and whether you are getting better or worse.
- **Your own reference.** Use the built-in Warrior II photo or upload your own, and choose which body parts count.
- **Phone panel.** Score, camera preview with skeleton, reference, and every setting in one mobile-friendly page.
- **Private by design.** Camera images and landmarks never leave the robot/PC. Only a small numeric summary is
  sent to the speech server with each spoken turn.

## How it works

![Pose coach architecture](docs/assets/pose_coach_architecture.svg)

1. The **pose service** reads camera frames at the configured rate (up to 30 FPS) and sends them to a separate
   **MediaPipe worker** process, which runs the Full pose model and compares your landmarks with the reference.
2. The **coach logic** smooths the scores over time, picks the most useful corrections and decides when
   something is worth saying (you matched, got better, got worse, stalled, left the frame...).
3. When you finish speaking, the **coach bridge** attaches a numeric `[pose_context]` snapshot to your audio
   turn, so the reply is grounded in what Reachy actually sees. It can also start a short spoken update on its own.
4. The **conversation handler** streams audio to and from the speech server and plays the reply on Reachy.
5. Optional **automatic framing** gently turns Reachy's head and body to keep you in view. Small antenna
   cues celebrate progress.

The diagram source is [docs/scheme.mmd](docs/scheme.mmd).

## Requirements

| | Reachy Mini Lite | Reachy Mini Wireless |
|---|---|---|
| Where the app runs | Your computer (USB to the robot) | On the robot (Raspberry Pi CM4) |
| Daemon | Reachy Mini Control desktop app | Built into the robot |
| Pose model | MediaPipe 1.1.1, installed with the app | MediaPipe 0.10.18 in `~/reachy-pose-runtime`, created automatically on first start |
| Practical frame rate | Up to the camera's 30 FPS, depending on your CPU | About 3–5 FPS |

You also need a realtime speech server. By default the app uses the hosted Hugging Face server; you can
switch to your own (see [Speech server](#speech-server)).

## Install

The repository is self-contained: the pose model, built-in reference photo, coach profile and web panel are
all bundled in the package.

### Lite (Windows, with the Reachy Mini Control app)

Run in PowerShell:

```powershell
git clone https://github.com/daiyk/YogaPoseReachy.git
cd YogaPoseReachy
& "$env:LOCALAPPDATA\Reachy Mini Control\uv.exe" pip install . --python "$env:LOCALAPPDATA\Reachy Mini Control\apps_venv\Scripts\python.exe"
```

To update later without touching dependencies:

```powershell
& "$env:LOCALAPPDATA\Reachy Mini Control\uv.exe" pip install --no-deps --reinstall-package reachy-mini-pose-coach . --python "$env:LOCALAPPDATA\Reachy Mini Control\apps_venv\Scripts\python.exe"
```

### Wireless

Build the wheel on your computer, copy it to the robot and install it into the robot's app environment:

```powershell
uv build --wheel -o dist .
scp dist\reachy_mini_pose_coach-1.2.0-py3-none-any.whl pollen@<robot-ip>:/home/pollen/
ssh pollen@<robot-ip> "/venvs/apps_venv/bin/pip install /home/pollen/reachy_mini_pose_coach-1.2.0-py3-none-any.whl && /venvs/apps_venv/bin/pip check"
```

The first start creates the separate pose runtime from the bundled lock file. It needs internet access and
takes a few minutes; the panel shows *Installing pose tracker* meanwhile. The log is
`~/reachy-pose-runtime/install.log`.

## Use

1. Start **reachy_mini_pose_coach** from the Reachy Mini Control app list (or
   `POST http://<daemon>:8000/api/apps/start-app/reachy_mini_pose_coach`).
2. Open the panel: `http://127.0.0.1:7860/` on a Lite, `http://<robot-ip>:7860/` on a Wireless. It works on a phone.
3. Pick a reference: keep **Warrior II**, or upload a JPEG/PNG (up to 5 MB, one person, head and hips visible).
   Choose **Same** or **Opposite** side if you want to mirror the reference.
4. Stand so your whole body is visible, hips roughly facing the camera at an angle similar to the reference.
5. Say **"start practice"** (or press *Start practice*), and follow the spoken advice.
6. Say **"end practice"** to stop scoring but keep talking, or **"exit the app"** to put Reachy to sleep and stop.

### Choosing body parts

Under **Parts to compare**, select the torso and each arm or leg you want scored. Head, hands and feet can be
added as *visibility and framing only*: they never affect the score. Selections are saved per reference image
and side. A selected arm needs shoulder, elbow and wrist visible, a leg needs hip, knee and ankle, and both
hips are always required.

## Settings

All settings are in the panel and apply immediately when saved.

| Setting | Default | Range | What it does |
|---|---|---|---|
| Idle / listening / thinking rate | 3 FPS | 0.5–30 FPS | How often a frame is scored in each conversation state |
| Searching rate | 1 FPS | 0.5–30 FPS | Rate while nobody is detected |
| Score bands | 50 / 70 / 85 | 5–95 | Boundaries between *far*, *getting closer*, *close* and *match* |
| Event cooldowns | 5–20 s | 2–120 s | Minimum time between repeated spoken updates of the same kind |
| Speech gap | 5 s | 5–60 s | Quiet time before Reachy may speak up on its own |
| Automatic framing | Off | | Lets Reachy turn its head and body to keep you in view |
| Camera field of view | From the SDK | 20–160° / 15–130° | Only needed if the SDK has no calibration for your camera |

Scoring pauses while Reachy is speaking or moving, and for half a second afterwards. Higher rates make the
coach react faster but use more CPU. The actual rate can be lower than requested if the pose model cannot keep up.

Automatic framing is deliberately gentle: at most 8° per move, one move every 2 seconds, head within 25°
of the body, and body within 60° of centre. Check that Reachy has clear space before enabling it.

Settings are stored in the app's instance folder (`coach-settings.json`, `coach-reference.jpg`,
`coach-reference-parts.json`, `settings.json`). On the robot this is
`/home/pollen/.config/reachy_mini/conversation_app/`.

### Speech server

At the bottom of the panel, **Speech server** lets you choose **Hosted** (default) or **Your own server** with a
URL such as `ws://192.168.1.20:8765/v1/realtime`, then **Save and reconnect**. The choice is remembered in
`settings.json` and wins over any `.env` value. Your server must listen on `0.0.0.0` and be reachable from the
robot. `ws://` is not encrypted, and `HF_TOKEN` (if set) is sent to that server as the bearer token.

### Environment variables

All optional; see [.env.example](.env.example) for the full list.

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token for the hosted speech server |
| `HF_REALTIME_CONNECTION_MODE`, `HF_REALTIME_WS_URL` | Starting speech-server choice (the panel overrides them) |
| `REALTIME_TRANSCRIPTION_LANGUAGE` | Fix the transcription language, e.g. `en` |
| `REACHY_MINI_INSTANCE_PATH` | Move the instance folder |
| `REACHY_POSE_PYTHON` | Run the pose worker with another Python interpreter |
| `REACHY_POSE_RUNTIME` | Move the Wireless pose runtime folder |

### Coach personality

The coach's instructions live in [profiles/coach/profile.md](profiles/coach/profile.md). Edit it and restart the
app to change how Reachy coaches.

## Privacy

- Camera frames are processed locally and never recorded. The preview is only produced while the panel is
  open, and stops within a second of closing it.
- Uploaded reference photos stay in the instance folder.
- The speech server receives your voice and a bounded numeric snapshot (score, band, up to two corrections,
  matched parts). It never receives images or landmarks.

## Known limitations

- **Viewing angle.** Scoring needs your hips within about 30° of facing the camera and within 20° of the
  reference's angle. Otherwise the panel asks you to turn your hips toward the camera. Near the limit this can flicker.
- **Head visibility matters.** MediaPipe uses the face to tell left from right, so the head must stay in the image.
- **Accuracy.** Scores are geometric similarity to one reference photo, not a validated judgement of yoga form.
- **No depth.** Corrections only describe movements visible to the camera (raise/lower, bend, lean, tilt);
  there is no forward/back or hip-turn advice.

## Development

```powershell
uv sync                 # creates .venv with the app and the dev tools
uv run pytest           # tests (no robot needed)
uv run ruff check .
uv run ruff format --check .
uv run mypy             # strict type check of src/
```

Project layout:

| Path | Contents |
|---|---|
| `src/reachy_mini_pose_coach/coach/` | Coaching logic: scoring, corrections, events, framing, snapshots, panel routes |
| `src/reachy_mini_pose_coach/pose/` | Pose service, MediaPipe worker, comparison geometry, bundled model and reference |
| `src/reachy_mini_pose_coach/static/` | Web panel |
| `src/reachy_mini_pose_coach/huggingface_realtime.py` | Conversation handler for the realtime speech server |
| `profiles/coach/profile.md` | Coach instructions |
| `tests/` | Unit and integration tests; `tests/coach/fixtures/` holds synthetic score sequences |

## License

See [LICENSE](LICENSE).
