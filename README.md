# 👁️ FaceAttend — Facial Recognition Attendance System

A real-time facial recognition attendance system with a role-based web dashboard, built with Python, Flask, and OpenCV.

## ✨ Features

- **Real-time face recognition** via webcam using `face_recognition` + `dlib`
- **Anti-spoofing liveness detection** — blink required (EAR algorithm) to mark attendance
- **Live camera stream in the browser** — no separate terminal needed
- **Role-based dashboard** — Admin / Teacher / Student roles with separate views
- **Dark / Light mode** toggle (persisted in localStorage)
- **Student profiles** — photo, attendance %, donut chart, full log
- **Admin panel** — manage teachers and students, view all attendance
- **Attendance history** — browse by date, export CSV
- **Hot-reload** — adding a new student via web triggers camera re-encoding

## 🗂️ Project Structure

```
facialrecoz/
├── attendence.py          # Standalone camera script (optional)
├── requirements.txt       # Python dependencies
├── webapp/
│   ├── app.py             # Flask web application
│   ├── static/
│   │   └── style.css      # Design system (dark/light mode)
│   └── templates/         # Jinja2 HTML templates
│       ├── login.html
│       ├── index.html           # Teacher dashboard
│       ├── admin_dashboard.html # Admin control panel
│       ├── camera.html          # Live camera stream
│       ├── history.html         # Attendance history
│       ├── students.html        # Student management
│       ├── student_profile.html # Individual student profile
│       ├── student_portal.html  # Student self-service portal
│       └── ...
```

> **Note:** `dataset_images/`, `*.db`, and `*.pkl` files are excluded from this repo  
> (personal photos & local data). You must generate them locally.

## 🚀 Setup & Run

### 1. Clone & install dependencies

```bash
git clone https://github.com/Anuj-Gaud/facialrecoz.git
cd facialrecoz
pip install -r requirements.txt
```

### 2. Start the web application

```bash
python webapp/app.py
```

Open → **http://127.0.0.1:5000**

Default login: `admin` / `admin123`

### 3. Add students

1. Log in as admin → **Students** → **Add Student** (upload a clear front-facing photo)
2. Navigate to **📷 Camera** → click **Start Camera**
3. Face the camera and **blink once** — attendance is marked automatically ✅

### 4. (Optional) Standalone camera script

```bash
python attendence.py
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Web Framework | Flask |
| Face Recognition | `face_recognition` (dlib) |
| Computer Vision | OpenCV |
| Database | SQLite |
| Frontend | Vanilla HTML / CSS / JS |

## 🔐 Default Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |

> Change the admin password after first login for security.

## 📄 License

MIT — free to use and modify.
