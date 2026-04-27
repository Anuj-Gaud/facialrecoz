import os, csv, pickle, sqlite3, cv2, face_recognition
import threading, time
import numpy as np
from io import StringIO
from datetime import datetime, date, timedelta
from functools import wraps
from flask import (Flask, render_template, request, jsonify, Response,
                   redirect, url_for, send_file, session)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH       = os.path.join(BASE_DIR, 'attendance.db')
DATASET_DIR   = os.path.join(BASE_DIR, 'dataset_images')
ENCODING_FILE = os.path.join(BASE_DIR, 'known_face_encodings.pkl')
RELOAD_FLAG   = os.path.join(BASE_DIR, 'reload_encodings.flag')

app = Flask(__name__)
app.secret_key = 'faceattend-super-secret-2026'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── DATABASE ──────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_face_students():
    """Return list of student folder names in the dataset directory."""
    if not os.path.isdir(DATASET_DIR):
        return []
    return [d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')]


def setup_database():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL,
            UNIQUE(name, date)
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','teacher','student')),
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS teacher_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_name TEXT NOT NULL,
            username TEXT NOT NULL,
            date TEXT NOT NULL,
            login_time TEXT NOT NULL,
            UNIQUE(username, date)
        );
    ''')
    # Seed default admin if not exists
    if not conn.execute("SELECT id FROM users WHERE username='admin'").fetchone():
        conn.execute("INSERT INTO users (username,password_hash,role,name) VALUES (?,?,?,?)",
                     ('admin', generate_password_hash('admin123'), 'admin', 'Administrator'))
    conn.commit()
    conn.close()

# ── LIVE CAMERA  (Feature 10) + LIVENESS DETECTION (Feature 9) ─────────────
class FaceCamera:
    """Background thread: webcam → face recognition → EAR blink liveness
    → marks attendance DB → provides annotated MJPEG frames to browser."""
    EAR_BLINK = 0.22   # eye aspect ratio ≤ this  → eye considered closed
    BLINK_N   = 2      # consecutive frames needed to confirm a valid blink

    def __init__(self):
        self.cap, self.running = None, False
        self._frame, self._lock = None, threading.Lock()
        self.liveness           = {}   # {NAME: {consec:int, blinked:bool}}
        self.encodings, self.names = [], []
        self._load_enc()

    def _load_enc(self):
        try:
            if os.path.exists(ENCODING_FILE):
                with open(ENCODING_FILE, 'rb') as f:
                    self.encodings, self.names = pickle.load(f)
        except Exception:
            self.encodings, self.names = [], []

    @staticmethod
    def _ear(pts):      # Soukupova & Cech (2016)
        p = [np.array(pt, dtype=float) for pt in pts]
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) \
               / (2.0 * np.linalg.norm(p[0]-p[3]) + 1e-6)

    @staticmethod
    def _db_mark(name):
        try:
            conn = sqlite3.connect(DB_PATH); now = datetime.now()
            conn.execute(
                "INSERT OR IGNORE INTO attendance (name,date,time) VALUES(?,?,?)",
                (name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')))
            conn.commit(); conn.close()
        except Exception: pass

    def _annotate(self, img):
        if os.path.exists(RELOAD_FLAG):
            self._load_enc()
            try: os.remove(RELOAD_FLAG)
            except: pass
        sm  = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(sm, cv2.COLOR_BGR2RGB)
        locs   = face_recognition.face_locations(rgb)
        encs   = face_recognition.face_encodings(rgb, locs)
        lmarks = face_recognition.face_landmarks(rgb, locs)
        for enc, loc, lm in zip(encs, locs, lmarks):
            name, conf = 'UNKNOWN', 0
            if self.encodings:
                dists = face_recognition.face_distance(self.encodings, enc)
                idx   = int(np.argmin(dists))
                if face_recognition.compare_faces(self.encodings, enc, tolerance=0.50)[idx]:
                    name = self.names[idx].upper(); conf = int((1-dists[idx])*100)
            is_live = False
            if name != 'UNKNOWN':
                le  = lm.get('left_eye',  [])
                re  = lm.get('right_eye', [])
                trk = self.liveness.setdefault(name, {'consec':0,'blinked':False})
                if le and re:
                    ear = (self._ear(le) + self._ear(re)) / 2.0
                    if ear < self.EAR_BLINK:
                        trk['consec'] += 1
                    else:
                        if trk['consec'] >= self.BLINK_N: trk['blinked'] = True
                        trk['consec'] = 0
                is_live = trk['blinked']
                if is_live: self._db_mark(name)
            y1, x2, y2, x1 = [v*4 for v in loc]
            if   name == 'UNKNOWN': col, lbl = (0,0,210),   'UNKNOWN'
            elif not is_live:       col, lbl = (0,165,255), f'{name}  |  BLINK TO VERIFY'
            else:                   col, lbl = (0,210,80),  f'{name}  {conf}%  LIVE'
            cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
            cv2.rectangle(img,(x1,y2-36),(x2,y2),col,cv2.FILLED)
            cv2.putText(img, lbl,(x1+5,y2-10),cv2.FONT_HERSHEY_DUPLEX,.58,(255,255,255),1)
        return img

    def _loop(self):
        n = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok: time.sleep(.05); continue
            n += 1
            if n % 3 == 0: frame = self._annotate(frame)
            with self._lock: self._frame = frame
            time.sleep(.008)
        self.cap.release(); self.cap = None

    def get_jpeg(self):
        with self._lock: f = self._frame
        if f is None: return None
        _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()

    def start(self):
        if self.running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("Cannot open webcam (device 0). Is a camera connected?")
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False; self.liveness = {}

_cam = None
_cam_lock = threading.Lock()

def _the_cam():
    global _cam
    with _cam_lock:
        if _cam is None: _cam = FaceCamera()
    return _cam

def _mjpeg():
    cam = _the_cam()
    while cam.running:
        jpg = cam.get_jpeg()
        if jpg: yield b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        time.sleep(.033)

# ── CONTEXT PROCESSOR (inject user into all templates) ────
@app.context_processor
def inject_user():
    return dict(
        current_role=session.get('role'),
        current_name=session.get('name'),
        current_uid=session.get('user_id')
    )

# ── AUTH DECORATORS ───────────────────────────────────────
def login_required(f):
    @wraps(f)
    def dec(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return dec

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def dec(*args, **kwargs):
            if not session.get('user_id'):
                return redirect(url_for('login'))
            if session.get('role') not in roles:
                return render_template('403.html'), 403
            return f(*args, **kwargs)
        return dec
    return decorator

# ── HELPERS ───────────────────────────────────────────────
def get_face_students():
    if not os.path.exists(DATASET_DIR): return []
    return [d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')]

# ── AUTH ROUTES ───────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('home'))
    error = None
    if request.method == 'POST':
        uname = request.form.get('username','').strip()
        pwd   = request.form.get('password','')
        conn  = get_db()
        user  = conn.execute("SELECT * FROM users WHERE username=?", (uname,)).fetchone()
        if user and check_password_hash(user['password_hash'], pwd):
            session['user_id'] = user['id']
            session['role']    = user['role']
            session['name']    = user['name']
            # Log teacher login as attendance
            if user['role'] == 'teacher':
                now = datetime.now()
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO teacher_attendance (teacher_name, username, date, login_time) VALUES (?,?,?,?)",
                        (user['name'], user['username'], now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'))
                    )
                    conn.commit()
                except: pass
            conn.close()
            return redirect(url_for('home'))
        conn.close()
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    role = session.get('role')
    if role == 'admin':    return redirect(url_for('admin_dashboard'))
    if role == 'teacher':  return redirect(url_for('dashboard'))
    if role == 'student':  return redirect(url_for('student_portal'))
    return redirect(url_for('login'))

# ── ADMIN ROUTES ──────────────────────────────────────────
@app.route('/admin')
@role_required('admin')
def admin_dashboard():
    conn = get_db()
    teachers = conn.execute("SELECT * FROM users WHERE role='teacher' ORDER BY created_at DESC").fetchall()
    students_db = conn.execute("SELECT * FROM users WHERE role='student' ORDER BY created_at DESC").fetchall()
    today = datetime.now().strftime('%Y-%m-%d')
    # Teacher attendance for today
    teacher_att_today = conn.execute(
        "SELECT * FROM teacher_attendance WHERE date=? ORDER BY login_time ASC", (today,)
    ).fetchall()
    teacher_att_present = {row['username'] for row in teacher_att_today}
    # Student attendance for today (from camera / attendance table)
    student_att_today = conn.execute(
        "SELECT name, time FROM attendance WHERE date=? ORDER BY time ASC", (today,)
    ).fetchall()
    conn.close()
    total_face_students = len(get_face_students())
    return render_template('admin_dashboard.html',
                           teachers=teachers,
                           students_db=students_db,
                           today=today,
                           teacher_att_today=teacher_att_today,
                           teacher_att_present=teacher_att_present,
                           present_today=len(teacher_att_today),
                           absent_today=len(teachers)-len(teacher_att_today),
                           student_att_today=student_att_today,
                           student_present=len(student_att_today),
                           student_absent=max(0, total_face_students - len(student_att_today)),
                           total_face_students=total_face_students)

@app.route('/admin/teacher_attendance')
@role_required('admin')
def teacher_attendance_dashboard():
    today      = datetime.now().strftime('%Y-%m-%d')
    this_month = datetime.now().strftime('%Y-%m')
    conn = get_db()
    teachers = conn.execute("SELECT * FROM users WHERE role='teacher'").fetchall()
    today_att = conn.execute(
        "SELECT * FROM teacher_attendance WHERE date=? ORDER BY login_time ASC", (today,)
    ).fetchall()
    teacher_att_present = {row['username'] for row in today_att}
    # All-time records for table
    all_records = conn.execute(
        "SELECT * FROM teacher_attendance ORDER BY date DESC, login_time DESC"
    ).fetchall()
    # Month stats
    month_days_present = conn.execute(
        "SELECT COUNT(DISTINCT date) FROM teacher_attendance WHERE date LIKE ?",
        (f'{this_month}%',)
    ).fetchone()[0]
    total_month_days = (datetime.now().day)
    conn.close()
    total_teachers  = len(teachers)
    present_today   = len(today_att)
    absent_today    = total_teachers - present_today
    return render_template('admin_teacher_attendance.html',
                           teachers=teachers,
                           today_att=today_att,
                           teacher_att_present=teacher_att_present,
                           all_records=all_records,
                           today=today, this_month=this_month,
                           total_teachers=total_teachers,
                           present_today=present_today,
                           absent_today=absent_today,
                           month_days_present=month_days_present,
                           total_month_days=total_month_days)

@app.route('/admin/teacher_analytics')
@role_required('admin')
def teacher_analytics():
    conn       = get_db()
    today_d    = date.today()
    total_teachers = conn.execute("SELECT COUNT(*) FROM users WHERE role='teacher'").fetchone()[0]
    labels, trend_data = [], []
    for i in range(6, -1, -1):
        d  = today_d - timedelta(days=i)
        ds = d.strftime('%Y-%m-%d')
        c  = conn.execute(
            "SELECT COUNT(DISTINCT username) FROM teacher_attendance WHERE date=?", (ds,)
        ).fetchone()[0]
        labels.append(d.strftime('%b %d'))
        trend_data.append(c)
    present = conn.execute(
        "SELECT COUNT(DISTINCT username) FROM teacher_attendance WHERE date=?",
        (today_d.strftime('%Y-%m-%d'),)
    ).fetchone()[0]
    conn.close()
    return jsonify({
        'trend': {'labels': labels, 'data': trend_data},
        'today_donut': {'present': present, 'absent': max(0, total_teachers - present)}
    })

@app.route('/admin/add_teacher', methods=['POST'])
@role_required('admin')
def add_teacher():
    name     = request.form.get('name','').strip()
    username = request.form.get('username','').strip()
    password = request.form.get('password','').strip()
    if not all([name, username, password]):
        return jsonify({'status':'error','message':'All fields are required'}), 400
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username,password_hash,role,name) VALUES (?,?,?,?)",
                     (username, generate_password_hash(password), 'teacher', name))
        conn.commit()
        return jsonify({'status':'success','message':f'Teacher {name} added!'})
    except sqlite3.IntegrityError:
        return jsonify({'status':'error','message':'Username already exists'}), 400
    finally:
        conn.close()

@app.route('/admin/delete_teacher/<int:uid>', methods=['POST'])
@role_required('admin')
def delete_teacher(uid):
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=? AND role='teacher'", (uid,))
    conn.commit()
    conn.close()
    return jsonify({'status':'success','message':'Teacher removed.'})

# ── TEACHER / SHARED ROUTES ───────────────────────────────
@app.route('/dashboard')
@role_required('teacher','admin')
def dashboard():
    today = datetime.now().strftime('%Y-%m-%d')
    conn  = get_db()
    records = conn.execute(
        "SELECT name, time FROM attendance WHERE date=? ORDER BY time ASC", (today,)
    ).fetchall()
    conn.close()
    total   = len(get_face_students())
    present = len(records)
    return render_template('index.html', records=records, today=today,
                           total_students=total, present_count=present,
                           absent_count=total-present)

@app.route('/analytics')
@role_required('teacher','admin')
def analytics():
    conn  = get_db()
    today = date.today()
    labels, trend_data = [], []
    for i in range(6,-1,-1):
        d  = today - timedelta(days=i)
        ds = d.strftime('%Y-%m-%d')
        c  = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (ds,)).fetchone()[0]
        labels.append(d.strftime('%b %d'))
        trend_data.append(c)
    present = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?",
                           (today.strftime('%Y-%m-%d'),)).fetchone()[0]
    conn.close()
    total = len(get_face_students())
    return jsonify({'trend':{'labels':labels,'data':trend_data},
                   'today_donut':{'present':present,'absent':max(0,total-present)}})

@app.route('/history')
@role_required('teacher','admin')
def history():
    selected = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    conn = get_db()
    records = conn.execute(
        "SELECT name, time FROM attendance WHERE date=? ORDER BY time ASC", (selected,)
    ).fetchall()
    conn.close()
    return render_template('history.html', records=records, selected_date=selected)

@app.route('/export_csv')
@role_required('teacher','admin')
def export_csv():
    selected = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    conn = get_db()
    rows = conn.execute(
        "SELECT name, date, time FROM attendance WHERE date=? ORDER BY time ASC", (selected,)
    ).fetchall()
    conn.close()
    output = StringIO()
    w = csv.writer(output)
    w.writerow(['Name','Date','Time'])
    for r in rows: w.writerow([r['name'], r['date'], r['time']])
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition':f'attachment; filename=attendance_{selected}.csv'})

@app.route('/students')
@role_required('teacher', 'admin')
def students():
    student_list = []
    for name in get_face_students():
        person_dir = os.path.join(DATASET_DIR, name)
        files = [f for f in os.listdir(person_dir) if not f.startswith('.')]
        student_list.append({'name': name, 'photo': files[0] if files else None})
    return render_template('students.html', students=student_list)

@app.route('/student_photo/<name>/<filename>')
@login_required
def student_photo(name, filename):
    return send_file(os.path.join(DATASET_DIR, name, filename))

@app.route('/add_student', methods=['POST'])
@role_required('teacher', 'admin')
def add_student():
    name = request.form.get('name','').strip()
    if not name:
        return jsonify({'status':'error','message':'Name is required'}), 400
    if 'photo' not in request.files or not request.files['photo'].filename:
        return jsonify({'status':'error','message':'Photo is required'}), 400

    photo      = request.files['photo']
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    photo_path = os.path.join(person_dir, 'photo.jpg')
    photo.save(photo_path)

    img = cv2.imread(photo_path)
    if img is None:
        return jsonify({'status':'error','message':'Could not read image'}), 400

    encodings = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not encodings:
        os.remove(photo_path); os.rmdir(person_dir)
        return jsonify({'status':'error','message':'No face detected. Use a clear front-facing photo.'}), 400

    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE,'rb') as f: known_enc, known_names = pickle.load(f)
    else:
        known_enc, known_names = [], []
    known_enc.append(encodings[0]); known_names.append(name.upper())
    with open(ENCODING_FILE,'wb') as f: pickle.dump((known_enc, known_names), f)
    with open(RELOAD_FLAG,'w') as f: f.write('reload')

    # Auto-create student login: username=name(lower), password=name(lower)
    conn = get_db()
    try:
        conn.execute("INSERT INTO users (username,password_hash,role,name) VALUES (?,?,?,?)",
                     (name.lower(), generate_password_hash(name.lower()), 'student', name))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()
    return jsonify({'status':'success',
                    'message':f'{name} added! Student login — user: {name.lower()} / pass: {name.lower()}'})

@app.route('/delete_student/<name>', methods=['POST'])
@role_required('teacher', 'admin')
def delete_student(name):
    import shutil
    person_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(person_dir): shutil.rmtree(person_dir)
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE,'rb') as f: enc, names = pickle.load(f)
        pairs = [(e,n) for e,n in zip(enc,names) if n.upper()!=name.upper()]
        enc   = [p[0] for p in pairs]; names = [p[1] for p in pairs]
        with open(ENCODING_FILE,'wb') as f: pickle.dump((enc, names), f)
        with open(RELOAD_FLAG,'w') as f: f.write('reload')
    conn = get_db()
    conn.execute("DELETE FROM users WHERE username=? AND role='student'", (name.lower(),))
    conn.commit(); conn.close()
    return jsonify({'status':'success','message':f'{name} removed.'})

# ── STUDENT PORTAL ────────────────────────────────────────
@app.route('/student')
@role_required('student')
def student_portal():
    sname      = session['name'].upper()
    this_month = datetime.now().strftime('%Y-%m')
    conn       = get_db()
    records    = conn.execute(
        "SELECT date, time FROM attendance WHERE name=? ORDER BY date DESC", (sname,)
    ).fetchall()
    month_present = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE name=? AND date LIKE ?",
        (sname, f'{this_month}%')
    ).fetchone()[0]
    total_days = conn.execute(
        "SELECT COUNT(DISTINCT date) FROM attendance WHERE date LIKE ?",
        (f'{this_month}%',)
    ).fetchone()[0]
    conn.close()
    pct = round((month_present/total_days*100) if total_days > 0 else 0)
    return render_template('student_portal.html', records=records,
                           month_present=month_present, total_days=total_days,
                           percentage=pct, this_month=this_month)

# ── CAMERA ROUTES (Feature 10) ───────────────────────────
@app.route('/camera')
@role_required('teacher','admin')
def camera_page():
    return render_template('camera.html')

@app.route('/video_feed')
@role_required('teacher','admin')
def video_feed():
    cam = _the_cam()
    if not cam.running:
        try: cam.start()
        except RuntimeError: pass
    return Response(_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=f')

@app.route('/camera/start', methods=['POST'])
@role_required('teacher','admin')
def cam_start():
    try:   _the_cam().start(); return jsonify({'ok':True})
    except RuntimeError as e: return jsonify({'ok':False,'msg':str(e)}), 500

@app.route('/camera/stop', methods=['POST'])
@role_required('teacher','admin')
def cam_stop():
    global _cam
    with _cam_lock:
        if _cam: _cam.stop(); _cam = None
    return jsonify({'ok':True})

@app.route('/camera/is_active')
@role_required('teacher','admin')
def cam_is_active():
    with _cam_lock: active = _cam is not None and _cam.running
    return jsonify({'active':active})

# ── STUDENT PROFILE (Feature 12) ──────────────────────────
@app.route('/profile/<name>')
@role_required('teacher','admin')
def student_profile(name):
    uname   = name.upper()
    conn    = get_db()
    records = conn.execute(
        "SELECT date, time FROM attendance WHERE name=? ORDER BY date DESC",
        (uname,)).fetchall()
    this_month    = datetime.now().strftime('%Y-%m')
    month_present = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE name=? AND date LIKE ?",
        (uname, f'{this_month}%')).fetchone()[0]
    total_days = conn.execute(
        "SELECT COUNT(DISTINCT date) FROM attendance WHERE date LIKE ?",
        (f'{this_month}%',)).fetchone()[0]
    conn.close()
    pct        = round(month_present/total_days*100 if total_days else 0)
    person_dir = os.path.join(DATASET_DIR, name)
    photo      = None
    if os.path.isdir(person_dir):
        fs = [f for f in os.listdir(person_dir) if not f.startswith('.')]
        if fs: photo = fs[0]
    return render_template('student_profile.html',
        student_name=name, records=records,
        month_present=month_present, total_days=total_days,
        percentage=pct, this_month=this_month, photo=photo)

@app.route('/attendance_today')
@role_required('teacher','admin')
def attendance_today():
    today = datetime.now().strftime('%Y-%m-%d')
    conn  = get_db()
    rows  = conn.execute(
        "SELECT name, time FROM attendance WHERE date=? ORDER BY time DESC",
        (today,)).fetchall()
    conn.close()
    return jsonify([{'name':r['name'],'time':r['time']} for r in rows])

# ── LEGACY (camera script compat) ─────────────────────────
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    name = data.get('name')
    if not name: return jsonify({'status':'error'}), 400
    conn = get_db(); now = datetime.now()
    try:
        conn.execute("INSERT OR IGNORE INTO attendance (name,date,time) VALUES (?,?,?)",
                     (name, now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')))
        conn.commit()
        return jsonify({'status':'success'})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500
    finally: conn.close()

if __name__ == '__main__':
    setup_database()
    os.makedirs(DATASET_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0')