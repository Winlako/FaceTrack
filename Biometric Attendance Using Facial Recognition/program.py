import cv2        
import time  
from datetime import datetime, timedelta 
from flask_pymongo import PyMongo     
import csv         
import io         
from queue import Queue 
import base64       
import threading
import numpy as np     
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session, send_file
import face_recognition    


app = Flask(__name__, template_folder='templates', static_folder='static') 
app.config['MONGO_URI'] = 'mongodb+srv://josephconduah02:UkNbEzggrk2kbl6Y@cluster0.xd3xxs1.mongodb.net/studentDB?retryWrites=true&w=majority&appName=Cluster0'
mongo = PyMongo(app) 

# Creates the TTL index
mongo.db.activity.create_index("timestamp", expireAfterSeconds=1800)

# Variables
known_faces, known_names, known_images = [], [], {}
latest_recognition = {key: None for key in ['name', 'image', 'student_ref_number', 'index_number', 'course', 'level', 'status', 'recognition_time', 'recognition_date', 'attendance_message', 'assigned_class']}
recognition_cache = {}
face_locations, face_encodings, recent_face_locations = [], [], []
FRAME_COUNT, FRAME_SKIP = 0, 2
SMOOTHING_FACTOR, DEBOUNCE_TIME = 0.6, 0.5
last_recognition_time = 0
frame_queue = Queue(maxsize=10) 
lock = threading.Lock()
video_capture = None
recognition_thread = None
running = False
recognition_threshold = 0.5

# functions
def get_class_assignment(index_number, recognition_date):
    try:
        day = datetime.strptime(recognition_date, '%Y-%m-%d').strftime('%A')
        index_num = int(index_number)
        ranges = {'SF1': (8000000, 8199999), 'SF7': (8200000, 8399999), 'SF8': (8400000, 8599999), 'SF19': (8600000, 8799999), 'SF20': (8800000, 8999999)}
        timetable = mongo.db.timetables.find_one({'exam_day': day})
        for code in timetable.get('class_code', []):
            if code in ranges and ranges[code][0] <= index_num <= ranges[code][1]:
                return code
    except Exception as e:
        print(f"[ERROR] Class assignment: {e}")
    return None

def load_images_from_db():
    known_faces.clear()     
    known_names.clear()
    known_images.clear()
    #fetch student documents in db
    for student in mongo.db.students.find():
        img_bin = student.get('captured_image')
        if img_bin:
            img_array = np.frombuffer(img_bin, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                name = f"{student['first_name']} {student['last_name']}"
                known_faces.append(encodings[0])
                known_names.append(name)
                _, buffer = cv2.imencode('.jpg', image)
                known_images[name] = base64.b64encode(buffer).decode()

def smooth_face_locations(new_locations, prev_locations):
    if not prev_locations:
        return new_locations
    return [tuple(int(SMOOTHING_FACTOR * new + (1 - SMOOTHING_FACTOR) * prev)
                  for new, prev in zip(new_box, prev_box))
            for new_box, prev_box in zip(new_locations, prev_locations)]

def recognize_faces():
    global face_locations, face_encodings, latest_recognition, FRAME_COUNT, recent_face_locations, last_recognition_time
    while running:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                FRAME_COUNT += 1
                if FRAME_COUNT % FRAME_SKIP != 0:
                    with lock:
                        face_locations = recent_face_locations[-1] if recent_face_locations else []
                    continue

                resized = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                temp_locs = face_recognition.face_locations(rgb_frame, model="hog")
                temp_encs = face_recognition.face_encodings(rgb_frame, temp_locs)

                with lock:
                    face_locations = smooth_face_locations(temp_locs, recent_face_locations[-1]) if recent_face_locations else temp_locs
                    face_encodings = temp_encs
                    recent_face_locations.append(face_locations)
                    if len(recent_face_locations) > 5:
                        recent_face_locations.pop(0)

                now = datetime.now()
                for loc, enc in zip(face_locations, face_encodings):
                    name = 'Unknown'
                    if known_faces:
                        distances = face_recognition.face_distance(known_faces, enc)
                        idx = np.argmin(distances)
                        if distances[idx] < recognition_threshold:
                            name = known_names[idx]

                    with lock:  # Ensure cache access is thread-safe
                        # Check cache with a normalized name
                        if name in recognition_cache and (now - recognition_cache[name]['timestamp']).total_seconds() < 1800:
                            latest_recognition.update(recognition_cache[name]['data'])
                            latest_recognition['attendance_message'] = 'Attendance already taken'
                            continue

                    # Debounce check
                    if time.time() - last_recognition_time < DEBOUNCE_TIME:
                        continue
                    last_recognition_time = time.time()

                    # Process new recognition
                    parts = name.split()
                    student = mongo.db.students.find_one({'first_name': parts[0], 'last_name': parts[1]}) if len(parts) == 2 else None
                    if student:
                        assigned_class = get_class_assignment(student.get('index_number'), now.strftime('%Y-%m-%d'))
                        data = {
                            'name': name,
                            'image': known_images.get(name),
                            'student_ref_number': student.get('student_ref_number'),
                            'index_number': student.get('index_number'),
                            'course': student.get('course'),
                            'level': student.get('level'),
                            'status': 'Active',
                            'recognition_time': now.strftime('%H:%M:%S'),
                            'recognition_date': now.strftime('%Y-%m-%d'),
                            'attendance_message': 'Attendance recorded',
                            'assigned_class': assigned_class,
                            'timestamp': now
                        }
                        with lock:  # Ensure cache update is thread-safe
                            # Verify no recent entry exists in MongoDB
                            existing = mongo.db.activity.find_one({
                                'index_number': student.get('index_number'),
                                'timestamp': {'$gte': now - timedelta(minutes=30)}
                            })
                            if existing:
                                latest_recognition.update(data)
                                latest_recognition['attendance_message'] = 'Attendance already taken'
                                recognition_cache[name] = {'timestamp': now, 'data': data}
                                continue

                            mongo.db.activity.insert_one(data)
                            recognition_cache[name] = {'timestamp': now, 'data': data}
                            latest_recognition.update(data)
                    else:
                        with lock:
                            latest_recognition.update({key: None for key in latest_recognition})
        except Exception as e:
            print(f"[ERROR] recognize_faces: {e}")

def generate_frames():
    global video_capture
    while video_capture and video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

        with lock:
            boxes = face_locations.copy()
            name = latest_recognition.get('name') or 'Unknown'

        for (top, right, bottom, left) in boxes:
            top, right, bottom, left = [x * 4 for x in (top, right, bottom, left)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes
@app.route('/')
def index(): return render_template('index.html')

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'GET':
        return render_template('enroll.html')
    if request.is_json:
        data = request.get_json() #gets JSON data
        img_data = base64.b64decode(data['captured_image'].split(',')[1])
        student = {
            'first_name': data['first_name'],
            'last_name': data['last_name'],
            'student_ref_number': data['student_ref_number'],
            'index_number': data['index_number'],
            'email': data['email'],
            'course': data['course'],
            'level': data['level'],
            'captured_image': img_data
        }
        if mongo.db.students.find_one({'index_number': data['index_number']}):
            return jsonify({'message': 'Index number already exists!'}), 400
        mongo.db.students.insert_one(student)
        load_images_from_db()
        return jsonify({'message': 'Student enrolled successfully!'}), 200
    return jsonify({'message': 'Invalid request'}), 400

@app.route('/video_feed')
def video_feed():
    if video_capture is None or not video_capture.isOpened():
        return Response(b'', mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global video_capture, recognition_thread, running
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        running = True
        recognition_thread = threading.Thread(target=recognize_faces, daemon=True)
        recognition_thread.start()
    elif not recognition_thread or not recognition_thread.is_alive():
        recognition_thread = threading.Thread(target=recognize_faces, daemon=True)
        recognition_thread.start()
    return jsonify(status='Camera started')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global video_capture, recognition_thread, running, latest_recognition
    running = False
    if video_capture:
        video_capture.release()
        video_capture = None
    recognition_thread = None
    latest_recognition.update({key: None for key in latest_recognition})
    recognition_cache.clear()
    frame_queue.queue.clear()
    return jsonify(status='Camera stopped')

@app.route('/get_recognition_result')
def get_recognition_result(): return jsonify(latest_recognition)

@app.route('/students')
def students(): return render_template('students.html')

@app.route('/activity')
def activity():
    recent = list(mongo.db.activity.find().sort("timestamp", -1).limit(100))
    return render_template('activity.html', activity=recent)

@app.route('/get_students')
def get_students():
    students = mongo.db.students.find()
    return jsonify([{
        'first_name': s['first_name'],
        'last_name': s['last_name'],
        'student_ref_number': s['student_ref_number'],
        'index_number': s['index_number'],
        'email': s['email'],
        'course': s['course'],
        'level': s['level'],
        'image': base64.b64encode(s['captured_image']).decode() if s.get('captured_image') else None
    } for s in students])

@app.route('/delete_student', methods=['POST'])
def delete_student():
    data = request.get_json()
    ref = data.get('student_ref_number')
    result = mongo.db.students.delete_one({'student_ref_number': ref})
    if result.deleted_count:
        load_images_from_db()
        return jsonify({'message': 'Student deleted successfully!'}), 200
    return jsonify({'message': 'Student not found'}), 404

@app.route('/get_recognition_history')
def get_recognition_history():
    return jsonify(list(mongo.db.activity.find({}, {'_id': 0})))

@app.route('/download_activity_csv')
def download_activity_csv():
    activities = mongo.db.activity.find()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Student Reference Number', 'Index Number', 'Course', 'Level', 'Status', 'Recognition Time', 'Recognition Date', 'Attendance Message', 'Assigned Class'])
    for a in activities:
        writer.writerow([a.get(k, 'N/A') for k in ['name', 'student_ref_number', 'index_number', 'course', 'level', 'status', 'recognition_time', 'recognition_date', 'attendance_message', 'assigned_class']])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='activity_log.csv')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        user = mongo.db.users.find_one({'username': request.form['username']})
        if user and user.get('password') == request.form['password']:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        return render_template('admin_login.html', error='Incorrect username or password')
    return render_template('admin_login.html')

@app.route('/admin_logout')
def admin_logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('index'))
    return render_template('admin_dashboard.html')

@app.route('/overview')
def overview():
    if not session.get('admin_logged_in'):
        return redirect(url_for('index'))
    return render_template('overview.html')

@app.route('/timetables')
def timetables():
    if not session.get('admin_logged_in'):
        return redirect(url_for('index'))
    return render_template('timetables.html', timetables=list(mongo.db.timetables.find()))

@app.route('/save_timetable', methods=['POST'])
def save_timetable():
    if not session.get('admin_logged_in'):
        return jsonify({'status': 'error', 'message': 'Unauthorized access'}), 401
    data = request.get_json()
    if not data.get('course_code'):
        return jsonify({'status': 'error', 'message': 'Course code is required'}), 400
    mongo.db.timetables.update_one(
        {'course_code': data['course_code']},
        {'$set': data},
        upsert=True
    )
    return jsonify({'status': 'success', 'message': 'Timetable updated successfully'})

@app.route('/system_stats')
def system_stats():
    return render_template('system_stats.html')

@app.route('/settings')
def settings(): return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
def save_settings():
    global recognition_threshold
    recognition_threshold = float(request.get_json().get('recognition_threshold', 0.5))
    return jsonify({'message': 'Settings saved successfully!'})


if __name__ == '__main__':
    load_images_from_db()
    app.secret_key = 'your_secret_key'
    app.run(debug=False)