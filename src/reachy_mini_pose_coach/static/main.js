import { describeError, rpcCall } from './js/api.js';

const $ = (id) => document.getElementById(id);
const labels = {
  person_lost: 'Person out of view', framing_bad: 'Framing needs attention',
  pose_achieved: 'Pose matched', got_worse: 'Pose moved away', got_better: 'Pose improved',
  correction_changed: 'New correction', stalled: 'Time for a small adjustment',
  mirrored: 'Mirrored pose',
};
const reasons = {
  no_person: 'Step into view', occluded: 'Keep the selected parts and both hips visible',
  out_of_frame: 'Bring your head and the selected parts into view', too_close: 'Step back to fit in view',
  unsupported_view: 'Turn your hips toward the camera and match the reference body angle', invalid_geometry: 'Pose is unclear',
  stabilizing: 'Hold still for a moment', worker_failure: 'Pose tracking is unavailable',
  no_sample: 'Waiting for a fresh pose', settling: 'Letting the camera settle',
  paused: 'Observation paused while Reachy speaks or moves', practice_stopped: 'Ready when you are',
  unreliable_tracking: 'Face the camera and keep your head visible so left and right can be tracked',
  mirrored: 'You are doing the mirror image of the reference: switch sides',
};
const edges = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[28,30],[30,32]];
let state = null;
let loadedSettings = false;
let dirty = false;
let busy = false;
let partsDirty = false;
let partsRevision = '';
const scoringParts = ['torso','left_arm','right_arm','left_leg','right_leg'];
const partLabels = name => name.replaceAll('_',' ');
let previewOpen = false;
let liveImage = null;
let livePoints = [];
let referenceImage = null;
let referenceKey = '';
let lastPreview = 0;
let micMuted = false;

async function api(path, method = 'GET', body) {
  const headers = {'X-Reachy-Coach': '1'};
  if (body !== undefined && !(body instanceof File)) headers['Content-Type'] = 'application/json';
  const response = await fetch(`/api/coach/${path}`, {
    method, headers, cache: 'no-store', signal: AbortSignal.timeout(8000),
    body: body === undefined ? undefined : body instanceof File ? body : JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = typeof error.detail === 'string' ? error.detail : 'Check the settings and try again.';
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return path === 'reference' ? response.blob() : response.json();
}

async function imageFrom(blob) {
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally { URL.revokeObjectURL(url); }
}

function draw(canvas, image, points, mirror = false) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!image) return;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  ctx.save();
  if (mirror) { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
  ctx.drawImage(image, 0, 0);
  const required = new Set(state?.pose[canvas.id === 'reference-canvas' ? 'required_reference_landmarks' : 'required_live_landmarks'] || []);
  points = points.map((point, index) => required.has(index) ? point : [point[0],point[1],false]);
  ctx.strokeStyle = '#50f4b5';
  ctx.lineWidth = Math.max(2, canvas.width / 200);
  for (const [a, b] of edges) {
    if (!points[a]?.[2] || !points[b]?.[2]) continue;
    ctx.beginPath();
    ctx.moveTo(points[a][0] * canvas.width, points[a][1] * canvas.height);
    ctx.lineTo(points[b][0] * canvas.width, points[b][1] * canvas.height);
    ctx.stroke();
  }
  ctx.fillStyle = '#fff';
  for (const [x,y,visible] of points) {
    if (!visible) continue;
    ctx.beginPath(); ctx.arc(x * canvas.width, y * canvas.height, 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

async function refreshReference() {
  const blob = await api('reference');
  referenceImage = blob ? await imageFrom(blob) : null;
  draw($('reference-canvas'), referenceImage, state?.pose.reference_points || []);
}

function buttons() {
  $('start').disabled = busy || !state?.connected || !state?.pose.reference_ready || state.practice_active;
  $('end').disabled = busy || !state?.practice_active;
  for (const input of $('body-parts').querySelectorAll('input')) input.disabled = busy;
  for (const id of ['reference', 'side', 'upload', 'save', 'apply-parts']) $(id).disabled = busy;
}

async function action(operation) {
  if (busy) return;
  busy = true; buttons(); $('notice').textContent = '';
  try { await operation(); }
  catch (error) { $('notice').textContent = error.message; }
  finally { busy = false; buttons(); }
}

function fillSettings(settings) {
  for (let i = 0; i < 3; i++) $(`band-${i}`).value = settings.band_edges[i];
  for (const mode of ['idle','listening','thinking','searching']) $(`${mode}-fps`).value = settings[`${mode}_fps`];
  $('speech-gap').value = settings.speech_gap_s;
  $('framing-enabled').checked = settings.framing_enabled;
  $('horizontal-fov').value = settings.horizontal_fov_deg ?? '';
  $('vertical-fov').value = settings.vertical_fov_deg ?? '';
  $('cooldowns').replaceChildren();
  for (const [kind, seconds] of Object.entries(settings.cooldowns)) {
    const label = document.createElement('label');
    label.textContent = `${labels[kind]} cooldown (seconds)`;
    const input = document.createElement('input');
    Object.assign(input, {type:'number', min:'2', max:'120', required:true, value:seconds, id:`cooldown-${kind}`});
    label.append(input); $('cooldowns').append(label);
  }
  loadedSettings = true;
}

function show(next) {
  state = next;
  $('connection').textContent = next.connected ? 'Conversation connected' : 'Reconnecting conversation';
  $('connection').classList.toggle('online', next.connected);
  const runtimeUnavailable = ['error','unavailable','invalid_reference'].includes(next.pose.status);
  $('availability').textContent = runtimeUnavailable ?
    (next.pose.status === 'invalid_reference' ? 'Check the photo or selected body parts' : 'Pose tracking is unavailable; conversation can continue') :
    next.status === 'valid' ? 'Pose in view' : reasons[next.reason] || 'Waiting for a fresh pose';
  $('activity').textContent = next.practice_active ? `Practice active · ${next.activity}` : 'Practice is stopped';
  $('reference-status').textContent = ({ready:'Ready',loading:'Checking pose…',starting:'Starting tracker…',installing:'Installing pose tracker (first start, a few minutes)…',invalid_reference:'Photo needs attention'})[next.pose.status] || 'Tracking unavailable';
  if (!busy) {
    $('reference').value = next.pose.reference || 'builtin';
    $('side').value = next.pose.side || 'same';
  }
  if (next.pose.revision !== partsRevision) {
    partsRevision = next.pose.revision; partsDirty = false;
    $('scoring-parts').replaceChildren(); $('visibility-parts').replaceChildren();
    for (const part of next.pose.part_options || []) {
      const label = document.createElement('label'); label.className = 'check';
      const input = document.createElement('input');
      Object.assign(input, {type:'checkbox', value:part, checked:next.pose.parts.includes(part)});
      label.append(input, document.createTextNode(partLabels(part)));
      $(scoringParts.includes(part) ? 'scoring-parts' : 'visibility-parts').append(label);
    }
    $('parts-saved').textContent = '';
    livePoints = []; draw($('live-canvas'), liveImage, []);
  }
  $('score-scope').textContent = `Scoring: ${(next.scored_parts || next.pose.parts?.filter(p => scoringParts.includes(p)) || []).map(partLabels).join(', ') || 'Waiting for reference'}`;
  $('required-parts').textContent = next.pose.error || (next.pose.missing_landmarks?.length ? `Needs visible: ${next.pose.missing_landmarks.join(', ')}` : '');
  $('reference').options[1].disabled = !next.pose.uploaded_available;
  const valid = next.status === 'valid' && Number.isFinite(next.score);
  $('score').textContent = valid ? Math.round(next.score) : '—';
  $('band').textContent = valid ? next.band.replaceAll('_',' ') : 'Pose unavailable';
  $('freshness').textContent = valid ? `Observed ${next.age_s.toFixed(1)} seconds ago` : reasons[next.reason] || 'Waiting for a fresh observation';
  $('meter').firstElementChild.style.width = `${valid ? next.score : 0}%`;
  if (valid) $('meter').setAttribute('aria-valuenow', next.score); else $('meter').removeAttribute('aria-valuenow');
  const cues = (next.top_corrections || []).map(c => c.action
    ? `${c.action[0].toUpperCase()}${c.action.slice(1)}.`
    : `${partLabels(c.measurement)}: ${c.issue} (about ${Math.round(c.amount_deg)}°).`);
  const matched = next.matched_parts?.length ? `Matching: ${next.matched_parts.map(partLabels).join(', ')}.` : '';
  $('correction').textContent = valid
    ? [matched, ...cues].filter(Boolean).join(' ') || 'Follow Reachy’s spoken guidance and stay comfortable.'
    : reasons[next.reason] || 'Pose guidance will appear when tracking is ready.';
  $('correction').classList.toggle('alert', next.reason === 'mirrored');
  $('events').replaceChildren();
  for (const event of next.events.slice(0,6)) {
    const li = document.createElement('li');
    li.textContent = `${labels[event.kind] || event.kind} · ${event.seconds_ago}s ago`;
    $('events').append(li);
  }
  if (!next.events.length) { const li = document.createElement('li'); li.textContent = 'No observations yet'; $('events').append(li); }
  if (!loadedSettings) fillSettings(next.settings);
  $('calibration').textContent = next.framing_calibrated ? 'Camera calibration is available.' : 'Automatic framing requires camera calibration or both measured angles.';
  buttons();
  const key = JSON.stringify([next.pose.revision, next.pose.status]);
  if (key !== referenceKey) {
    referenceKey = key;
    refreshReference().catch(error => { $('notice').textContent = error.message; });
  }
  if (previewOpen && performance.now() - lastPreview > 2000) {
    $('preview-placeholder').textContent = 'Waiting for a fresh camera frame';
    $('preview-placeholder').hidden = false;
    draw($('live-canvas'), null, []);
  }
}

$('start').onclick = () => action(async () => show(await api('practice','POST',{active:true})));
$('end').onclick = () => action(async () => show(await api('practice','POST',{active:false})));
for (const id of ['reference','side']) $(id).onchange = () => action(async () => {
  show(await api('reference/select','POST',{selection:$('reference').value,side:$('side').value}));
  await refreshReference();
});
$('upload').onchange = () => action(async () => {
  const file = $('upload').files[0];
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) throw new Error('Choose a JPEG or PNG of at most 5 MB.');
  show(await api('reference/upload','POST',file));
  $('reference').value = 'uploaded'; $('side').value = 'same';
  await refreshReference(); $('upload').value = '';
});
$('body-parts').oninput = () => { partsDirty = true; $('parts-saved').textContent = 'Unsaved selection'; };
$('body-parts').onsubmit = event => {
  event.preventDefault();
  action(async () => {
    const parts = [...$('body-parts').querySelectorAll('input:checked')].map(input => input.value);
    if (!parts.some(part => scoringParts.includes(part))) throw new Error('Select at least one torso, arm or leg.');
    show(await api('reference/select','POST',{selection:$('reference').value,side:$('side').value,parts}));
    partsDirty = false; $('parts-saved').textContent = 'Saved on Reachy';
  });
};
$('settings').oninput = () => { dirty = true; $('saved').textContent = 'Unsaved changes'; };
$('settings').onsubmit = (event) => {
  event.preventDefault();
  action(async () => {
    const body = {
      band_edges:[0,1,2].map(i => Number($(`band-${i}`).value)),
      cooldowns:Object.fromEntries(Object.keys(labels).map(key => [key,Number($(`cooldown-${key}`).value)])),
      speech_gap_s:Number($('speech-gap').value), framing_enabled:$('framing-enabled').checked,
      horizontal_fov_deg:$('horizontal-fov').value === '' ? null : Number($('horizontal-fov').value),
      vertical_fov_deg:$('vertical-fov').value === '' ? null : Number($('vertical-fov').value),
    };
    for (const mode of ['idle','listening','thinking','searching']) body[`${mode}_fps`] = Number($(`${mode}-fps`).value);
    show(await api('settings','PUT',body)); dirty = false; $('saved').textContent = 'Saved on Reachy';
  });
};
function syncBackend() {
  const local = $('backend-mode').value === 'local';
  $('backend-url-field').hidden = !local;
  $('backend-url').required = local;
}
function fillBackend(status) {
  $('backend-mode').value = status.hf_connection_mode;
  $('backend-url').value = status.hf_ws_url || '';
  $('backend-error').textContent = status.backend_error ? `Last connection error: ${status.backend_error}` : '';
  syncBackend();
}
$('backend-mode').onchange = syncBackend;
$('backend-card').ontoggle = () => {
  if (!$('backend-card').open) return;
  $('backend-saved').textContent = '';
  rpcCall('conversation.status').then(fillBackend).catch(error => { $('backend-saved').textContent = describeError(error); });
};
$('backend').onsubmit = async event => {
  event.preventDefault();
  $('backend-save').disabled = true; $('backend-saved').textContent = 'Saving…';
  try {
    const params = {hf_mode: $('backend-mode').value};
    if (params.hf_mode === 'local') params.hf_ws_url = $('backend-url').value.trim();
    const reply = await rpcCall('backend.config', params);
    fillBackend(reply); $('backend-error').textContent = ''; // the old target's error
    $('backend-saved').textContent = reply.message;
  } catch (error) { $('backend-saved').textContent = describeError(error); }
  finally { $('backend-save').disabled = false; }
};
$('preview').onclick = () => {
  previewOpen = !previewOpen;
  $('preview').textContent = previewOpen ? 'Close camera preview' : 'Open camera preview';
  $('preview').setAttribute('aria-pressed', String(previewOpen));
  $('preview-placeholder').hidden = false;
  $('preview-placeholder').textContent = previewOpen ? 'Waiting for camera…' : 'Camera preview is off';
  if (!previewOpen) { liveImage = null; livePoints = []; draw($('live-canvas'), null, []); }
};
$('mirror').onchange = () => draw($('live-canvas'),liveImage,livePoints,$('mirror').checked);
$('mic').onclick = () => action(async () => {
  const reply = await rpcCall('conversation.mic',{muted:!micMuted});
  micMuted = reply.muted;
  $('mic').textContent = micMuted ? 'Unmute microphone' : 'Mute microphone';
});
window.addEventListener('beforeunload', event => { if (dirty || partsDirty) { event.preventDefault(); event.returnValue = ''; } });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { liveImage = null; livePoints = []; lastPreview = 0; draw($('live-canvas'),null,[]); }
});
async function poll() {
  if (!document.hidden) {
    try { show(await api('status')); }
    catch (error) {
      state = null; buttons(); $('connection').textContent = 'Coach disconnected';
      $('connection').classList.remove('online'); $('availability').textContent = error.message;
      $('score').textContent = '—'; $('band').textContent = 'Pose unavailable';
      $('meter').firstElementChild.style.width = '0%'; $('meter').removeAttribute('aria-valuenow');
      $('correction').textContent = 'Waiting for the coach to reconnect.';
    }
  }
  setTimeout(poll,500);
}
async function previewLoop() {
  if (previewOpen && !document.hidden && state) {
    try {
      const packet = await api('preview');
      if (packet && packet.age_s < 2) {
        const blob = await (await fetch(`data:image/jpeg;base64,${packet.jpeg}`)).blob();
        const image = await imageFrom(blob);
        if (previewOpen && !document.hidden) {
          liveImage = image; livePoints = packet.points; lastPreview = performance.now();
          draw($('live-canvas'),image,livePoints,$('mirror').checked);
          $('preview-placeholder').hidden = true;
        }
      }
    } catch (error) { $('preview-placeholder').textContent = error.message; $('preview-placeholder').hidden = false; }
  }
  setTimeout(previewLoop,33);
}
poll(); previewLoop();

