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



Conversation app with numeric pose coaching, paired audio-turn context, safe framing and a mobile web panel.

Every usable finalized audio turn receives a matching pose-context message before its initial spoken response.

The minimal live service reuses the installed pose-comparison worker on Wireless. If tracking is unavailable,

context says so explicitly; normal conversation and spoken exit remain available.



Read [plan.md](plan.md) for the design and [PHASE0_RESULTS.md](PHASE0_RESULTS.md) for the measured blockers.

The current locked profile is `profiles/coach/profile.md`.

Saying “exit the app” invokes `go_to_sleep`, sleeps the robot, and stops the app.



## Pose runtime (bundled)

The coach needs no other Reachy app. Since 1.2.0 it ships its own MediaPipe worker, comparison geometry,
Full pose model and built-in reference photo (`src/reachy_mini_pose_coach/pose/runtime/` and `pose/assets/`).
`reachy_mini_pose_compare` is no longer imported.

| Robot | Where MediaPipe runs | Setup |
|---|---|---|
| Lite / desktop (Windows, macOS, x86-64 Linux) | MediaPipe 1.1.1 in the app's own environment | Installed with the app's dependencies |
| Wireless (ARM64 Linux, CM4) | MediaPipe 0.10.18 in `~/reachy-pose-runtime` (1.x crashes on the CM4) | Created automatically on first start from `pose/runtime/requirements-arm64.lock.txt`; the panel shows *Installing pose tracker*. An existing compatible runtime is adopted, not reinstalled. Log: `~/reachy-pose-runtime/install.log`. |

`REACHY_POSE_PYTHON` points the worker at another interpreter (never installed into); `REACHY_POSE_RUNTIME`
moves the Wireless runtime directory. The comparison code is a typed copy of the compare app's; the
repository-level `checks/comparison_drift.py` (not part of this app) reports if the two copies diverge.


## Phase 0 diagnostics



Run these from `/home/pollen/reachy_mini_pose_coach` on the robot, using the existing shared app environment.

Measurements require the coach app to be running and, for `--pose` (which measures the compare app's own
Runner), the installed `reachy_mini_pose_compare` package and the

existing isolated pose runtime. They are on-robot Linux diagnostics, not Windows camera tools. Run CPU

measurements separately from installs, backend probes, or other benchmarks. Output directories must exist.



```bash

/venvs/apps_venv/bin/python scripts/phase0/probe_realtime.py --self-test

/venvs/apps_venv/bin/python scripts/phase0/measure_baseline.py --self-test

/venvs/apps_venv/bin/python scripts/phase0/probe_realtime.py --audio scripts/phase0/blue_question.wav --output results/phase0/protocol_probe_strict.json

/venvs/apps_venv/bin/python scripts/phase0/measure_baseline.py --seconds 60 --output results/phase0/conversation.json

/venvs/apps_venv/bin/python scripts/phase0/measure_baseline.py --pose pose --seconds 60 --output results/phase0/conversation_pose.json

/venvs/apps_venv/bin/python scripts/phase0/measure_baseline.py --pose pose --fps-limit 3 --threads 1 --seconds 60 --output results/phase0/pose_3fps_1thread.json

```



The protocol probe uses synthetic speech and separate text snapshots; it does not use the robot microphone

or speaker. It exits nonzero if any gate fails. It saves synthetic output audio and uses the same cancellation cleanup as the app. The corrected strict

probe passed two independent runs; `--no-cancellation-cleanup` reproduces the original backend behavior.

`--pose compare` selects the existing Full-model reference-comparison path; that combined workload has not

yet been validated. No dependency or motor settings are changed by the measurement script.



The original upstream README is retained in `README_OLD.md`. No cloud publication was requested or performed.



## Single-profile runtime



Edit `profiles/coach/profile.md` to author the coach's instructions and tool defaults, then restart the app.

There is no personality selector, editor, migration, or per-personality language setting. Startup voice

and the app-wide language setting remain available; English is the coach default. Saved tool integrations

and their validation are retained, but the coach runtime excludes them from its effective tool registry.



On backend cancellation, the app removes only the input items belonging to the interrupted request.

This prevents the current Hugging Face backend from repeating that request on the next turn. Completed

history and newer speech remain. An interrupted request is abandoned; restate it if you want it resumed.



Validation after cleanup: **514 tests passed**, Ruff and mypy passed. Pose-load readiness is still open.



## Use your own speech server

By default the coach uses the hosted Hugging Face speech backend. To use your own copy of the realtime server
(for example on a Mac at `ws://<mac-ip>:<port>/v1/realtime`), open **Speech server** at the bottom of the coach
page, choose **Your own server**, enter the **Server URL**, then **Save and reconnect**. The app reconnects right away
and remembers the choice in `settings.json` in its instance folder (on the robot:
`/home/pollen/.config/reachy_mini/conversation_app/settings.json`). That file holds the other UI settings too:

```json
{"hf_connection_mode": "local", "hf_ws_url": "ws://192.168.1.20:8765/v1/realtime", "version": 1}
```

`HF_REALTIME_CONNECTION_MODE` / `HF_REALTIME_WS_URL` in an optional `.env` still work as a starting value, but a choice saved
in the UI wins. Switching back to **Hosted** keeps the saved URL. The server must listen on `0.0.0.0` and the Mac firewall must allow
the port, so the robot can reach it. `ws://` is unencrypted, and `HF_TOKEN` (if set) is sent to that server as the bearer
token. Design: [docs/backend_url_plan.md](docs/backend_url_plan.md).



## Offline coach (Phase 2)



`src/reachy_mini_pose_coach/coach/` contains the numeric sample contract, activity/rate policy, scoring,

corrections, event detection/delivery, compact snapshots and pure audio-turn transitions. It adds no

dependencies and performs no camera, motor, microphone or network I/O.



- `Coach.update(now, activity, sample)` accepts completed `PoseSample` values. Call it on activity changes

  even when no frame is available. Capture times and caller-supplied `now` use the same monotonic clock;

  UTC capture time is display metadata. Sample IDs increase within a practice session.

- The one-second EMA and correction window use elapsed time. After a pause, historical EMA/baseline are

  retained, but only a fresh segment can confirm a change (three distinct consistent frames). Holds reset

  across pauses or observation gaps longer than 1.5 seconds. No sample older than one second drives events.

- The six event rules use the plan's holds, priorities and cooldowns. Match achievement is announced once

  per entry into match; stalled and correction-changed advice are suppressed in match. Delivery keeps

  one event per kind, expires it after three seconds, and checks again at dispatch. Its five-second gap

  applies only to proactive speech. The bridge calls `mark_spoken` at actual playback start.

- `Snapshot.to_text(now)` returns bounded `[pose_context]` JSON with at most one correction. It recomputes

  age from capture time and removes numeric advice from stale or unavailable observations. Signed bends

  retain anatomical left/right; unsigned direction errors never invent an adjustment direction.

- `Turns` freezes context at speech stop (or labels a final-input fallback), waits for final transcription

  and the exact context ACK, and guards one initial response by connection epoch and audio item ID.

  New speech, reference/practice invalidation, rejection, timeouts and reconnect make old requests unusable.

  Tool results and legitimate tool-follow-up responses remain outside this initial-response guard.



Run `python -m pytest tests/coach -q` in the existing development environment for the offline checks.

The Phase 2 gate passed **611 tests** (97 coach tests plus 514 retained tests), Ruff and strict mypy.

The existing wheel-build test also passes with the new package included.



The six JSONL fixtures under `tests/coach/fixtures/` are explicitly **synthetic numeric sequences**:

approaching, worsening, stalling, arm motion while matching, stepping outside the frame and leaving it.

They test decision rules, not pose-estimation accuracy. The minimal live service now maps comparison outputs to `PoseSample`; real recorded fixtures and

pose-estimation accuracy validation are still pending. Neither the real robot nor a simulator is

required by the offline tests.



## Audio-turn integration (Phase 3)



`coach/bridge.py` connects the numeric coach to `huggingface_realtime.py`. The client streams the same PCM

audio as before. Server VAD still detects speech and interrupts responses, but automatic response creation

is disabled and the effective settings are checked on every connection.



1. Speech stop freezes the latest coherent cached observation for that audio item. A missing stop event

   uses a labelled fallback at final transcription.

2. Final transcription starts a separate task to insert one user-role `input_text` item, prefixed with

   `[pose_context]`. The transcript is not sent again. The websocket receiver stays free to process ACKs.

3. The exact snapshot item acknowledgement unlocks a tagged, serial `response.create`. Its connection and

   audio-turn identity are checked again immediately before sending, including after explicit busy retries.



The snapshot is a separate conversation item paired with the audio by application IDs, not a native audio

attachment field. It carries bounded numeric status and one correction, never a camera image or raw

landmarks. Native backend vision is not used. Snapshots older than one second lose numeric advice and

become `stale`; absent samples or inactive practice are `unavailable`. Historical snapshots remain history.



The cache entry points on `handler.coach` are `set_reference(reference_id)` and `publish(PoseSample)`.

Reference/practice changes run on the handler's event loop; a worker thread may publish immutable samples.

Samples must use the selected reference, increasing IDs and the same monotonic clock. The bridge never

captures frames or runs inference. The live pose service follows `coach.activity` and the configured rate policy. Starting practice enables

sampling once its reference and isolated runtime are ready.



The only exposed tools are `get_pose_status`, `start_practice`, `end_practice`, `go_to_sleep`, and the

framework's `task_status` / `task_cancel`. Status reads the same snapshot schema. Start/end are idempotent

and use the existing tool-batch acknowledgement path. Required tool outputs survive newer user turns;

old-connection callbacks cannot write into a new session. Ending practice keeps ordinary conversation.



User speech takes priority over unsent advice. Cancelled responses discard late audio and remove only

their own abandoned input items. Reconnect flushes queued playback and invalidates old ACKs and requests.

Rejected/unknown snapshot acceptance closes the affected connection and surfaces an error instead of

generating an unpaired answer. Deadlines are 2 s for snapshot acceptance, 10 s for final transcription,

5 s for effective session settings and 30 s for response completion/acceptance.



Actual local audio pushes and the pending-frame/playback clock drive SPEAKING for paced and unpaced

output. `response.done` alone does not mean the speaker has drained; settling lasts another 0.5 s.

Proactive event context uses the same ACK gate and checks freshness again at dispatch. Head wobble,

head tracking and idle breathing are disabled. Small positive antenna cues preserve the head/body target.



Validation: **641 tests passed**, Ruff/format and strict mypy passed, and the wheel-build test passed.

The synthetic live probe exercised the actual handler on the configured HF backend twice, including a

reconnect: each reply used the attached `close` pose band and produced audio. Run it from this app's

installed development environment with configured HF authentication:



```bash

python scripts/phase3/probe_audio_snapshot.py --audio scripts/phase0/blue_question.wav --output results/phase3/live-probe.json

```



Evidence is in [results/phase3](results/phase3). The probe overrides the coaching instructions for a

deterministic one-word answer and disables tools; it uses synthetic audio/samples and no robot or speaker.

Real pose accuracy and combined inference/audio CPU load remain unvalidated. The initial implementation

was validated without the robot; the subsequent physical test is recorded below. The simulator was not started.



The inherited architecture diagram (`docs/scheme.mmd` and its generated SVG, referenced by `README_OLD.md`)

needs regeneration to depict the bridge, live pose service and coach panel; the SVG was not hand-edited.



### Physical validation — 2026-10-27



Deployed the verified build to the Wireless robot at `10.253.46.210`, with a backup and matching hashes

for all 71 packaged files. No dependencies or saved settings changed. The deployment adds bounded log

records for context ACKs, response requests and playback transitions. Its full regression gate still

passes 641 tests, Ruff/format and strict mypy.



Nine real microphone turns each received one accepted matching snapshot before one initial response

request. The user confirmed clear speech, explicit unavailable pose data, interruption without resuming

the old answer, start/end practice, and spoken exit. Logs verify start/end tool execution and removal of

the interrupted audio/snapshot pair. The first two end-practice attempts were transcribed as “And practice”;

a later attempt executed the tool. This remains an ASR usability observation.



A 30-second conversation-only check measured mean CPU 28.18%, mean reported daemon rate 49.31 Hz,

maximum temperature 46.25 C, zero daemon loop errors and no throttling. It is not a combined pose-load

benchmark. Software playback start/drain transitions and audible interruption were checked; exact

acoustic drain timing was not instrumented.



Exit left no current app and disabled motors. Shutdown logged a stop-request timeout and a daemon

websocket disconnect error, followed by a successful graceful stop; no forced termination was logged.

The stop endpoint waits for the process to exit while the app's own HTTP request times out after two

seconds. This cleanup warning remains recorded, not suppressed. The robot was left asleep.



See [physical-validation.json](results/phase3/physical-validation.json),

[physical-events.log](results/phase3/physical-events.log) and

[conversation-load.json](results/phase3/conversation-load.json). Phase 1 still needs to provide actual

pose samples; this physical run tested unavailable snapshots, not camera-based pose accuracy.



## Framing, live pose service and panel (Phases 4 and 5)



Open the app page at `http://10.253.46.210:7860/` while the coach is running. Choose the built-in Warrior II

reference or upload one JPEG/PNG (maximum 5 MB, one visible person), then start practice from the panel

or by voice. The image and numeric pose processing stay on the robot. Only the bounded numeric snapshot

is attached to each audio turn; no camera image or landmarks are sent to the speech backend.



The panel shows the current score, one anatomical mismatch, tracking availability, recent observations and

conversation connection. Preview starts off; opening it enables one latest JPEG at no more than 5 FPS,

with the skeleton drawn on its matching image. Closing/hiding the page stops its preview lease within one

second. No microphone audio or live camera images are recorded. The uploaded reference is retained locally

as `coach-reference.jpg`; startup selects the built-in reference, and the saved upload remains selectable.



Practice settings save atomically to `coach-settings.json` in the existing app instance directory. Band

boundaries, event cooldowns, speech gap and observation rates apply to the live coach after saving.

Default idle/listening/thinking rates are 3 FPS, searching 1 FPS, using the existing Full model with one

inference thread. Actual throughput can be lower. Inference pauses during playback, motion and the

following half-second settling interval. A stopped coordinator's inference lease expires after one second.



**Automatic framing is off by default.** Enable it in Practice settings after checking clear space around

Reachy. It uses SDK camera intrinsics, or both explicitly measured field-of-view angles. It never guesses

calibration. Present landmarks, including off-screen predictions, determine the one-second average box;

ordinary drift needs a one-second hold outside the dead zone. Edge clipping can request an idle correction.

A person occupying over 90% of the frame needs to step back, rather than causing a turn.



All framing goes through the existing movement owner: at least 0.5 seconds per move, at most an 8-degree

step and one move every two seconds, head yaw relative to body within 25 degrees, pitch within 25 degrees,

and body yaw within 60 degrees. Non-edge corrections wait for actual speaker playback. User speech halts

framing at the last commanded pose. Ending practice clears coach motion and queues a gentle neutral

return after listening ends; it does not terminate the movement worker. Spoken exit retains the existing

sleep/stop lifecycle. Wobble, face tracking and idle breathing remain disabled.



Runtime: the bundled pose runtime (see *Pose runtime (bundled)*); this phase originally used the
compare app's package, which 1.2.0 replaced. The SDK's NumPy environment is not altered. Only the robot's local
GStreamer camera is used; a missing runtime/camera leaves conversation available. The SDK exposes

no sensor exposure timestamp: sample age starts when the latest frame is received from camera IPC,

so upstream camera buffering is not included in the age estimate.



The adapter preserves comparison geometry, visibility and viewpoint gates. Its old three-frames-in-0.5s

gate cannot operate at 3 FPS; per-frame numeric results now feed the coach's existing temporal smoothing

and event holds instead. One worker restart is attempted after an inference error; a repeated error is

reported as unavailable. A reference selection retries preparation without restarting conversation.



Validation: 676 tests pass (35 new), Ruff/format, strict mypy and wheel build pass. The panel was inspected

at desktop and 390-pixel phone width, including start/end and saving settings, using explicitly synthetic

observations. On the robot's real isolated Full worker, 12 repeated public-reference frames produced 33

landmarks and score 100; this verifies the adapter and timing integration, not human-pose accuracy.

Numeric evidence is in `results/phase45`. Live sideways following, exact acoustic timing and the sustained

combined-load gate remain physical validation work; the earlier shutdown timeout/disconnect warning is

not claimed fixed. No simulator was started.



### Live camera smoke check — 2026-10-27



The deployed app loaded the Full-model reference, connected conversation and obtained camera calibration

from the SDK. A 26.27-second check collected 50 status samples and 14 preview packets.

No person was detected, so scores remained unavailable. Playback and settling paused pose observations.

Mean reported daemon rate was 49.04 Hz (minimum 47.96, maximum 49.50); loop errors stayed zero.

This brief searching/speech run with framing off does not pass the sustained >=49Hz framing-load gate.

End practice returned `practice_stopped` while conversation stayed connected; app-manager stop completed.

A final shutdown signal also prevents activity ticks from restarting inference during spoken exit.

The first launch failed because the daemon was stopped; starting it and clearing the failed app record

resolved that environment issue. The user subsequently confirmed feedback appears when the full body is

visible, with intermittent viewing-angle rejection (see below). Score accuracy, sideways following and

natural-motion feedback remain unvalidated. The app was left ready with practice and framing off.



### User feedback — 2026-10-27



The user confirms that coaching feedback appears once the full body is visible, as expected. They also

report intermittent unavailability with “Face the camera like the reference.” This label represents the

comparison engine's `unsupported_view` result: estimated hip yaw over 30 degrees from frontal, or over

20 degrees from the reference yaw. It does not test head/face orientation. Each frame is gated directly,

without angular hysteresis. Near a boundary, changing estimates can toggle availability; no yaw trace

was captured, so the specific failing condition is unconfirmed. Preserve the full-body visibility gate;

clarify the label and collect the rejected numeric angles before deciding whether to tune the limits.

No runtime settings or code changed in response to this report. See `results/phase45/user-feedback.json`.


### Front-facing setup guidance

The panel now gives a setup cue before practice: keep the whole body visible, with the hips roughly
facing Reachy’s camera and a body angle similar to the reference; the head can follow the pose.
The unsupported-view message asks for the same body orientation. The spoken coach gives this setup
cue once when practice starts and repeats it when supplied context reports `unsupported_view`, without
guessing a left/right turn or angle. The 30-degree frontal / 20-degree relative limits and visibility
gates are unchanged. Near-frontal is the supported comparison view; an accuracy advantage has not
been quantified. This wording update does not resolve threshold flicker.

## Body-part selection

In **Your reference**, choose torso, individual arms and legs under **Parts to compare**, then
**Apply body parts**. The score averages only those parts' existing measurements; excluded parts
cannot supply corrections or score-based events. At least one scoring part must stay selected.
A selected arm needs shoulder, elbow and wrist; a selected leg needs hip, knee and ankle. Both hips
always remain required for the existing orientation checks. Scoring pauses when required points are missing.

**Extra visibility and framing** independently includes head (nose and ears), hands (index, pinky,
thumb), and feet (heel and foot index). These add no scores. They default off, so clipping or poor
visibility of these extremities does not block a score or drive camera framing. Selected arms and
legs still require wrists and ankles. All five scoring parts default on.

Selections refer to the person in the reference photo. Opposite-side comparison maps both scoring
and visibility requirements to the opposite live side. The panel and spoken context identify the
scored live parts, and the panel shows missing required landmarks. Skeleton overlays and automatic
framing use the same required points. A partial match establishes similarity only for the scored parts.

Settings are saved locally in `coach-reference-parts.json`, separately for each built-in/uploaded image
and same/opposite side, bound to the image hash. Replacing an upload resets its selections. Startup
still selects the built-in reference. Upload a partial photo, deselect irrelevant parts, and apply;
reference validation then checks the chosen points. Invalid photos show which required points are missing.
Changing the selection resets score history, events, queued advice and pending framing.

`POST /api/coach/reference/select` accepts optional `parts: string[]`; omitting it restores the saved
selection for that image and side. The status exposes `parts`, `part_options`, required landmark indices,
missing names and a revision. Conversation snapshots add `scored_parts`, `visibility_only_parts` and
`missing_landmarks`. The upload endpoint is unchanged. Install the updated shared pose-comparison
package together with this coach: `prepare_reference(..., features=...)` now supports a fixed subset,
while callers omitting `features` retain the original full-body scoring and visibility gates.

Local verification (2026-11-02): 700 coach/project tests and 16 existing comparison tests pass;
Ruff, formatting, strict mypy, JavaScript syntax and wheel build checks pass. The phone-width panel was checked
with synthetic service data, including save/reload and no horizontal overflow. No robot deployment,
physical scoring-accuracy test or sustained combined-load measurement was performed for this change.
The previously noted architecture diagram regeneration remains pending.

## Directional feedback and mirror detection

Design: [docs/directional_feedback_plan.md](docs/directional_feedback_plan.md). Summary:
- **Snapshot schema 2:** `top_corrections` holds up to two items from different body parts, each `{part, measurement, issue, action, action_deg, amount_deg}`. There are also `matched_parts` and `mirrored`. Event turns carry an `event` object, inside the 2 KiB cap.
- **Signed actions:** `action` is a movement with its amount ("raise your left arm about 20°"). It is computed from depth-free signed angles (`coach/geometry.py`) of the raw world landmarks in the camera's plane: limb raise/lower or knee out/in, torso side lean, shoulder tilt, and the existing elbow/knee bends. A segment pointing mostly toward or away from the camera, overhead or across the midline, or changing sign within the 1 s window gets `action: null`. There is no forward/back or hip-turn advice.
- **Mirror detection (report only):** two probe comparisons without the relative-yaw gate (same side and mirrored) run on every usable frame. Two seconds of the mirrored reference fitting clearly better (≥70 and +15) set `mirrored` and suppress joint advice (`coach/mirror.py`).
- **Tracking guards:**
  - The head (nose) must be inside the image, in practice and in references.
  - Framing follows the selected parts plus the nose.
  - Swapped left/right hip or shoulder labels report `unreliable_tracking`.
- **Settings:** older `coach-settings.json` files gain the new `mirrored` cooldown on load, and their saved values are kept.

Local verification (2026-11-03):
- 734 project tests pass, including tests on real MediaPipe 0.10.18 Full landmarks of the bundled reference and its mirror image (`tests/coach/fixtures/reference_landmarks.json`).
- Ruff, formatting, strict mypy and the JavaScript syntax check pass.
- The shared comparison module is unchanged; its 16 benchmark and 20 compare-app tests pass.
