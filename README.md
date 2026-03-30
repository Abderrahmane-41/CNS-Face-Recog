# FaceGuard — Student Face Recognition Attendance System

## Project Structure

```
student-attendance/          ← Your project (this folder)
  app.py
  requirements.txt
  templates/
  static/
  database/

Silent-Face-Anti-Spoofing/   ← Cloned separately (same parent folder)
```

---

## Setup (Step by Step)

### 1. Clone the anti-spoofing repo (next to this project)
```bash
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
```

### 2. Install dependencies
```bash
# Install dlib (required by face_recognition)
# On Ubuntu/Debian:
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev

# On macOS:
brew install cmake

# Then install Python packages:
pip install -r requirements.txt

# Also install the anti-spoofing repo's requirements:
pip install -r ../Silent-Face-Anti-Spoofing/requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## How to Use

### Enroll a student
1. Go to the **Enroll** page
2. Start the camera
3. Enter the student's name
4. Capture their photo (make sure face is clearly visible)
5. Click **Enroll Student**

### Mark attendance
1. Go to the **Recognize** page (home)
2. Start the camera
3. Student stands in front of camera
4. Click **Scan Face**
5. System checks liveness → identifies student → marks present

### View attendance
- Go to the **Attendance** page
- Filter by date
- Export to CSV if needed

---

## Tuning Parameters (in app.py)

| Parameter        | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `SPOOF_THRESH`  | 0.6     | Anti-spoofing sensitivity (higher = stricter)    |
| `RECOGNITION_THR`| 0.50   | Face match threshold (lower = stricter matching) |

---

## Folder Reference

| File/Folder          | Purpose                                      |
|---------------------|----------------------------------------------|
| `app.py`            | Flask backend + all API routes               |
| `templates/`        | HTML pages (base, index, enroll, attendance) |
| `static/css/`       | Stylesheet                                   |
| `database/`         | SQLite DB (auto-created on first run)        |
| `requirements.txt`  | Python dependencies                          |
