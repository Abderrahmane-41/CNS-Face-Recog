/* liveness.js ─ MediaPipe Face Mesh blink-based active liveness gate
 * Requires the @mediapipe/face_mesh CDN script loaded before this file.
 * Keep the version in CDN_BASE (below) in sync with the <script> src in the templates.
 * Exposes three globals: initLiveness(onVerified), teardownLiveness(), resetLiveness(onVerified)
 *
 * Flow:
 *   1. initLiveness(cb)  — load FaceMesh, start RAF loop, show "Blink…" badge
 *   2. User blinks       — badge turns green, RAF loop pauses, cb() fires (e.g. enable button)
 *   3. resetLiveness(cb) — reset badge to "Blink…", restart RAF loop (re-arm after each action)
 *   4. teardownLiveness()— stop loop, close FaceMesh, hide badge (call on camera stop)
 */
(function () {
  /* Single source of truth for the MediaPipe version — keep in sync with the
   * <script src="…/face_mesh@0.4/face_mesh.js"> tag in the templates.          */
  const CDN_BASE   = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4';

  /* Eye landmark indices (MediaPipe Face Mesh 468-point model)
   * Layout per eye: [outer, upper1, upper2, inner, lower2, lower1]
   * EAR = (dist(upper1,lower1) + dist(upper2,lower2)) / (2 * dist(outer,inner))
   */
  const RIGHT_EYE  = [33,  160, 158, 133, 153, 144];
  const LEFT_EYE   = [362, 385, 387, 263, 373, 380];
  const EAR_THRESH = 0.22;  // eye aspect ratio below this → closed
  const MIN_CLOSED = 2;     // consecutive closed frames needed to register a blink

  let _mesh         = null;
  let _rafId        = null;
  let _closedFrames = 0;
  let _onVerified   = null;

  function _ear(lm, idx) {
    const d = (a, b) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
    return (d(idx[1], idx[5]) + d(idx[2], idx[4])) / (2 * d(idx[0], idx[3]));
  }

  function _onResults(r) {
    if (!_onVerified || !r.multiFaceLandmarks || !r.multiFaceLandmarks.length) return;
    const lm  = r.multiFaceLandmarks[0];
    const avg = (_ear(lm, RIGHT_EYE) + _ear(lm, LEFT_EYE)) / 2;
    if (avg < EAR_THRESH) {
      _closedFrames++;
    } else if (_closedFrames >= MIN_CLOSED) {
      // Blink confirmed — pause loop, update badge, fire callback
      _closedFrames = 0;
      if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
      _setBadge(true);
      const cb = _onVerified;
      _onVerified = null;
      cb();
    } else {
      _closedFrames = 0;
    }
  }

  function _setBadge(ok) {
    const el  = document.getElementById('livenessStatus');
    const msg = document.getElementById('livMsg');
    if (!el || !msg) return;
    el.className    = 'liveness-badge' + (ok ? ' live-ok' : '');
    msg.textContent = ok ? '✓ Liveness verified — proceed'
                         : 'Blink once to verify liveness…';
  }

  async function _loop() {
    if (!_mesh) return;
    const v = document.getElementById('video');
    // readyState >= 2 (HAVE_CURRENT_DATA) ensures a decoded video frame is available
    if (v && v.readyState >= 2) await _mesh.send({ image: v });
    _rafId = requestAnimationFrame(_loop);
  }

  window.initLiveness = async function (onVerified) {
    window.teardownLiveness();
    _onVerified   = onVerified || null;
    _closedFrames = 0;
    _mesh = new FaceMesh({
      locateFile: f => `${CDN_BASE}/${f}`
    });
    _mesh.setOptions({
      maxNumFaces:            1,
      refineLandmarks:        false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence:  0.5
    });
    _mesh.onResults(_onResults);
    await _mesh.initialize();
    const el = document.getElementById('livenessStatus');
    if (el) el.style.display = 'flex';
    _setBadge(false);
    _loop();
  };

  window.teardownLiveness = function () {
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    if (_mesh)  { _mesh.close();                _mesh  = null; }
    _closedFrames = 0;
    _onVerified   = null;
    const el = document.getElementById('livenessStatus');
    if (el) el.style.display = 'none';
  };

  /* Re-arm after a scan/retake: reset badge + restart RAF loop (reuses existing FaceMesh). */
  window.resetLiveness = function (onVerified) {
    _closedFrames = 0;
    _onVerified   = onVerified || null;
    _setBadge(false);
    if (!_rafId && _mesh) _loop();
  };
}());
