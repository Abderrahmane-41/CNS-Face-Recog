import os, sys, json, base64, sqlite3
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Silent-Face repo path
# Must be cloned as a sibling folder next to this project:
#   git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing ../Silent-Face-Anti-Spoofing
# ---------------------------------------------------------------------------
SILENT_FACE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Silent-Face-Anti-Spoofing')
)
sys.path.insert(0, SILENT_FACE_PATH)

SPOOF_THRESH    = 0.6    # score > threshold -> real face
RECOGNITION_THR = 0.50   # face_distance below this -> match (lower = stricter)
NUM_WEEKS       = 14

# ---------------------------------------------------------------------------
# Lazy-load Silent-Face internals once at startup
# ---------------------------------------------------------------------------
_spoof_predictor = None
_image_cropper   = None

def _load_spoof_models():
    global _spoof_predictor, _image_cropper
    if _spoof_predictor is not None:
        return
    original_dir = os.getcwd()
    try:
        os.chdir(SILENT_FACE_PATH)
        from src.anti_spoof_predict import AntiSpoofPredict
        from src.generate_patches import CropImage
        _spoof_predictor = AntiSpoofPredict(device_id=0)
        _image_cropper   = CropImage()
    finally:
        os.chdir(original_dir)


def is_real_face(bgr_frame: np.ndarray) -> tuple:
    """
    Liveness check — mirrors the test.py loop exactly but accepts a numpy
    array directly instead of a file path.

    parse_model_name() returns 4 values: (h_input, w_input, model_type, scale)
    The model filenames follow the pattern:  scale_HxW_ModelName.pth
    e.g. 2.7_80x80_MiniFASNetV2.pth  ->  h=80, w=80, type=MiniFASNetV2, scale=2.7
    """
    _load_spoof_models()

    original_dir = os.getcwd()
    try:
        os.chdir(SILENT_FACE_PATH)
        from src.utility import parse_model_name

        # Silent-Face requires a 3:4 (w:h) aspect ratio
        h, w = bgr_frame.shape[:2]
        target_h = int(w * 4 / 3)
        if h != target_h:
            bgr_frame = cv2.resize(bgr_frame, (w, target_h))

        # Get face bounding box (same as test.py does)
        image_bbox = _spoof_predictor.get_bbox(bgr_frame)

        model_dir   = os.path.join(SILENT_FACE_PATH, 'resources', 'anti_spoof_models')
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

        prediction = np.zeros((1, 3))
        for model_name in model_files:
            # parse_model_name returns 4 values: h_input, w_input, model_type, scale
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": bgr_frame,
                "bbox":    image_bbox,
                "scale":   scale,
                "out_w":   w_input,
                "out_h":   h_input,
                "crop":    True,
            }
            img_patch   = _image_cropper.crop(**param)
            prediction += _spoof_predictor.predict(
                img_patch, os.path.join(model_dir, model_name)
            )

        # class index 1 = real/live, class index 0 = spoof/fake
        score = float(prediction[0][1] / prediction.sum())
        return score > SPOOF_THRESH, round(score, 3)

    finally:
        os.chdir(original_dir)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'attendance.db')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    week_col_defs = ', '.join(f'week{i} INTEGER DEFAULT 0' for i in range(1, NUM_WEEKS + 1))
    with get_db() as conn:
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS students (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL,
                encoding TEXT NOT NULL,
                {week_col_defs}
            )
        ''')
        conn.commit()


def decode_image(data_url: str) -> np.ndarray:
    _, encoded = data_url.split(',', 1)
    arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image from base64 data.')
    return img


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll')
def enroll_page():
    return render_template('enroll.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')


# ---------------------------------------------------------------------------
# API: enroll
# ---------------------------------------------------------------------------
@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    data  = request.get_json(force=True)
    name  = (data.get('name') or '').strip()
    img64 = data.get('image')

    if not name or not img64:
        return jsonify(success=False, message='Name and image are required.')

    try:
        frame = decode_image(img64)
    except Exception as e:
        return jsonify(success=False, message=f'Image decode error: {e}')

    try:
        real, score = is_real_face(frame)
    except Exception as e:
        return jsonify(success=False, message=f'Anti-spoof error: {e}')

    if not real:
        return jsonify(success=False, message=f'Spoof detected (score {score}). Use a real face.')

    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb, num_jitters=1, model='small')
    if not encs:
        return jsonify(success=False, message='No face detected. Try better lighting or move closer.')

    new_enc = encs[0]
    force   = bool(data.get('force', False))  # bypass duplicate check if user confirmed

    # Duplicate check (skipped if user explicitly confirmed via force=True)
    if not force:
        with get_db() as conn:
            rows = conn.execute('SELECT id, name, encoding FROM students').fetchall()

        if rows:
            existing_encs = [np.array(json.loads(r['encoding'])) for r in rows]
            distances     = face_recognition.face_distance(existing_encs, new_enc)
            best_idx      = int(np.argmin(distances))
            if distances[best_idx] < RECOGNITION_THR:
                matched_name = rows[best_idx]['name']
                return jsonify(success=False,
                               duplicate=True,
                               matched=matched_name,
                               message=f'This face is too similar to already enrolled student "{matched_name}".')

    encoding = json.dumps(new_enc.tolist())
    with get_db() as conn:
        conn.execute('INSERT INTO students (name, encoding) VALUES (?, ?)', (name, encoding))
        conn.commit()

    return jsonify(success=True, message=f'Student "{name}" enrolled successfully!')


# ---------------------------------------------------------------------------
# API: recognize and mark attendance
# ---------------------------------------------------------------------------
@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    data  = request.get_json(force=True)
    img64 = data.get('image')
    week  = data.get('week')

    if not img64:
        return jsonify(success=False, message='No image received.')

    try:
        week = int(week)
        assert 1 <= week <= NUM_WEEKS
    except (TypeError, ValueError, AssertionError):
        return jsonify(success=False, message=f'Please select a valid week (1-{NUM_WEEKS}).')

    try:
        frame = decode_image(img64)
    except Exception as e:
        return jsonify(success=False, message=f'Image decode error: {e}')

    try:
        real, score = is_real_face(frame)
    except Exception as e:
        return jsonify(success=False, message=f'Anti-spoof error: {e}')

    if not real:
        return jsonify(success=False, message=f'Spoof detected (score {score}). Access denied.')

    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb, num_jitters=1, model='small')
    if not encs:
        return jsonify(success=False, message='No face detected in frame.')

    live_enc = encs[0]

    with get_db() as conn:
        rows = conn.execute('SELECT id, name, encoding FROM students').fetchall()

    if not rows:
        return jsonify(success=False, message='No students enrolled yet.')

    students  = [{'id': r['id'], 'name': r['name'],
                  'encoding': np.array(json.loads(r['encoding']))} for r in rows]
    encodings = [s['encoding'] for s in students]

    distances = face_recognition.face_distance(encodings, live_enc)
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist > RECOGNITION_THR:
        return jsonify(success=False, message='Face not recognized. Are you enrolled?')

    matched  = students[best_idx]
    week_col = f'week{week}'

    with get_db() as conn:
        current = conn.execute(
            f'SELECT {week_col} FROM students WHERE id=?', (matched['id'],)
        ).fetchone()

        if current[week_col] == 1:
            return jsonify(success=True, already=True,
                           name=matched['name'],
                           message=f'{matched["name"]} already marked present for Week {week}.')

        conn.execute(f'UPDATE students SET {week_col}=1 WHERE id=?', (matched['id'],))
        conn.commit()

    confidence = round((1 - best_dist) * 100, 1)
    return jsonify(success=True, already=False,
                   name=matched['name'],
                   confidence=confidence,
                   message=f'Welcome, {matched["name"]}! Marked present for Week {week}. ({confidence}% match)')


# ---------------------------------------------------------------------------
# API: full attendance grid
# ---------------------------------------------------------------------------
@app.route('/api/attendance')
def api_attendance():
    week_cols = ', '.join(f'week{i}' for i in range(1, NUM_WEEKS + 1))
    with get_db() as conn:
        rows = conn.execute(
            f'SELECT id, name, {week_cols} FROM students ORDER BY name'
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# API: students list
# ---------------------------------------------------------------------------
@app.route('/api/students')
def api_students():
    with get_db() as conn:
        rows = conn.execute('SELECT id, name FROM students ORDER BY name').fetchall()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# API: delete student
# ---------------------------------------------------------------------------
@app.route('/api/students/<int:sid>', methods=['DELETE'])
def api_delete_student(sid):
    with get_db() as conn:
        conn.execute('DELETE FROM students WHERE id=?', (sid,))
        conn.commit()
    return jsonify(success=True)


# ---------------------------------------------------------------------------
# API: manually mark a student present for a given week (from recognize page)
# ---------------------------------------------------------------------------
@app.route('/api/attendance/mark', methods=['POST'])
def api_mark_attendance():
    data       = request.get_json(force=True)
    student_id = data.get('student_id')
    week       = data.get('week')

    try:
        week = int(week)
        assert 1 <= week <= NUM_WEEKS
    except (TypeError, ValueError, AssertionError):
        return jsonify(success=False, message='Invalid week.')

    week_col = f'week{week}'
    with get_db() as conn:
        current = conn.execute(
            f'SELECT name, {week_col} FROM students WHERE id=?', (student_id,)
        ).fetchone()
        if current is None:
            return jsonify(success=False, message='Student not found.')

        if current[week_col] == 1:
            return jsonify(success=True, already=True,
                           name=current['name'],
                           message=f'{current["name"]} already marked present for Week {week}.')

        conn.execute(f'UPDATE students SET {week_col}=1 WHERE id=?', (student_id,))
        conn.commit()

    return jsonify(success=True, already=False,
                   name=current['name'],
                   message=f'{current["name"]} marked present for Week {week}.')


if __name__ == '__main__':
    init_db()
    _load_spoof_models()    # load once at startup
    app.run(debug=True, port=5000)
