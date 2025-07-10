from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Firebase
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Exercise configurations
exercises = {
    "knee_raises": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 60, "threshold": 15, "optimal_speed_range": (1.0, 2.5)},
    "squats": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (2.0, 4.0)},
    "lunges": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (2.0, 3.5)},
    "push_ups": {"joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (1.5, 3.0)},
    "plank": {"joints": ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"], "target_angle": 180, "threshold": 10, "optimal_speed_range": (0.5, 1.0)},
    "side_lunges": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (2.0, 3.5)},
    "mountain_climbers": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (1.0, 2.0)},
    "burpees": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (2.0, 4.0)},
    "deadlifts": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (2.5, 4.5)},
    "jumping_jacks": {"joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"], "target_angle": 180, "threshold": 15, "optimal_speed_range": (1.0, 2.0)},
    "sit_ups": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 45, "threshold": 10, "optimal_speed_range": (1.5, 3.0)},
    "leg_raises": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 45, "threshold": 10, "optimal_speed_range": (1.5, 3.0)},
    "side_plank": {"joints": ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"], "target_angle": 180, "threshold": 10, "optimal_speed_range": (0.5, 1.0)},
    "triceps_dips": {"joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"], "target_angle": 90, "threshold": 15, "optimal_speed_range": (1.5, 3.0)},
    "bicycle_crunches": {"joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], "target_angle": 45, "threshold": 10, "optimal_speed_range": (1.0, 2.5)}
}

# Global variables
session_data = {
    "count": 0,
    "correct_count": 0,
    "incorrect_count": 0,
    "position": None,
    "exercise_started": False,
    "feedback_message": "Begin Exercise!",
    "start_time": None,
    "last_rep_time": None,
    "exercise": None,
    "target_count": 0
}

# Global variable to store trained models for each user
user_models = {}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def generate_frames():
    global session_data
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks and session_data["exercise_started"] and session_data["exercise"]:
            landmarks = results.pose_landmarks.landmark
            coords = [[landmarks[getattr(mp_pose.PoseLandmark, joint).value].x,
                       landmarks[getattr(mp_pose.PoseLandmark, joint).value].y] for joint in next(iter(session_data["exercise"].values()))["joints"]]
            angle = calculate_angle(*coords)
            if angle > next(iter(session_data["exercise"].values()))["target_angle"] + next(iter(session_data["exercise"].values()))["threshold"]:
                session_data["position"] = "up"
            if session_data["position"] == "up" and angle < next(iter(session_data["exercise"].values()))["target_angle"] - next(iter(session_data["exercise"].values()))["threshold"]:
                session_data["position"] = "down"
                current_time = time.time()
                if session_data["last_rep_time"]:
                    rep_time = current_time - session_data["last_rep_time"]
                    if next(iter(session_data["exercise"].values()))["optimal_speed_range"][0] <= rep_time <= next(iter(session_data["exercise"].values()))["optimal_speed_range"][1]:
                        session_data["correct_count"] += 1
                        session_data["feedback_message"] = "Good form! Keep going."
                    else:
                        session_data["incorrect_count"] += 1
                        session_data["feedback_message"] = "Incorrect form! Adjust speed."
                session_data["count"] += 1
                session_data["last_rep_time"] = current_time
                if session_data["count"] == 1:
                    session_data["start_time"] = current_time
            if angle < next(iter(session_data["exercise"].values()))["target_angle"] - next(iter(session_data["exercise"].values()))["threshold"]:
                session_data["feedback_message"] = "Adjust your angle lower!"
            elif angle > next(iter(session_data["exercise"].values()))["target_angle"] + next(iter(session_data["exercise"].values()))["threshold"]:
                session_data["feedback_message"] = "Adjust your angle higher!"
            cv2.putText(image, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Count: {session_data["count"]}/{session_data["target_count"]}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, f'Correct: {session_data["correct_count"]}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Incorrect: {session_data["incorrect_count"]}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, session_data["feedback_message"], (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if session_data["count"] >= session_data["target_count"]:
                session_data["exercise_started"] = False
                total_time = time.time() - session_data["start_time"] if session_data["start_time"] else 0
                session_data["feedback_message"] = f"Exercise Complete! Time: {total_time:.2f}s"
                if 'email' in session:
                    session_data["post_therapy"] = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                   "total_time": total_time,
                                                   "correct_count": session_data["correct_count"],
                                                   "incorrect_count": session_data["incorrect_count"]}
                    user_ref = db.collection('users').document(session['email'])
                    user_ref.update({"sessions": firestore.ArrayUnion([{key: {"exercise": value,
                                                                              "count": session_data["count"],
                                                                              "correct_count": session_data["correct_count"],
                                                                              "incorrect_count": session_data["incorrect_count"],
                                                                              "total_time": total_time,
                                                                              "post_therapy": session_data["post_therapy"]}
                                                                      for key, value in session_data["exercise"].items()}])})
                    # Update leaderboard score (based on total correct counts)
                    total_correct = sum(session.get("correct_count", 0) for session in user_ref.get().to_dict().get("sessions", []))
                    user_ref.update({"leaderboard_score": total_correct})
                session_data = {**session_data, "count": 0, "correct_count": 0, "incorrect_count": 0, "position": None,
                                "start_time": None, "last_rep_time": None}
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Routes
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('profile'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        users_ref = db.collection('users').where('email', '==', email).limit(1)
        users = list(users_ref.stream())
        if users and users[0].to_dict()['password'] == password:
            session['email'] = email
            session['username'] = users[0].to_dict()['username']
            return redirect(url_for('profile'))
        return render_template('login.html', error="Invalid email or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        height = request.form['height']
        weight = request.form['weight']
        blood_group = request.form['blood_group']
        email_query = db.collection('users').where('email', '==', email).limit(1)
        if list(email_query.stream()):
            return render_template('register.html', error="Email already registered.")
        user_ref = db.collection('users').document(email)
        user_data = {"username": username, "email": email, "password": password, "age": age, "height": height,
                     "weight": weight, "blood_group": blood_group, "sessions": [], "leaderboard_score": 0}
        user_ref.set(user_data)
        session['email'] = email
        session['username'] = username
        return redirect(url_for('profile'))
    return render_template('register.html')

@app.route('/profile')
def profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    if not user.exists:
        return redirect(url_for('login'))
    user_data = user.to_dict()
    sessions = user_data.get('sessions', [])
    session_dates = [session.get('post_therapy', {}).get('date', 'N/A') for session in sessions]
    session_counts = [session.get('count', 0) for session in sessions]
    session_correct_counts = [session.get('correct_count', 0) for session in sessions]
    session_incorrect_counts = [session.get('incorrect_count', 0) for session in sessions]
    session_total_times = [session.get('total_time', 0) for session in sessions]
    session_average_speeds = [session.get('total_time', 0) / session.get('count', 1) if session.get('count', 0) > 0 else 0 for session in sessions]
    recovery_times = [session.get('total_time', 0) * (1 + session.get('average_speed', 0)) for session in sessions]
    predicted_recovery_times = []
    r2 = 0
    if len(sessions) > 1:
        X = np.array([[s.get('count', 0), s.get('average_speed', 0), s.get('total_time', 0)] for s in sessions])
        y = np.array(recovery_times)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        user_models[session['email']] = model
        predicted_recovery_times = model.predict(X).tolist()
        r2 = r2_score(y, predicted_recovery_times)
    else:
        predicted_recovery_times = recovery_times if recovery_times else [0] * len(sessions)
    return render_template('profile.html',
                           user=user_data,
                           session_dates=session_dates,
                           session_counts=session_counts,
                           session_correct_counts=session_correct_counts,
                           session_incorrect_counts=session_incorrect_counts,
                           session_total_times=session_total_times,
                           session_average_speeds=session_average_speeds,
                           recovery_times=recovery_times,
                           predicted_recovery_times=predicted_recovery_times,
                           r2=r2,
                           exercises=exercises.keys())

@app.route('/leaderboard')
def leaderboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    users = list(db.collection('users').stream())
    leaderboard = sorted([user.to_dict() for user in users if 'leaderboard_score' in user.to_dict()],
                         key=lambda x: x['leaderboard_score'], reverse=True)
    return render_template('leaderboard.html', leaderboard=leaderboard)

@app.route('/predict_recovery', methods=['POST'])
def predict_recovery():
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})
    data = request.json
    count = data.get('count')
    average_speed = data.get('average_speed')
    total_time = data.get('total_time')
    if not all([isinstance(count, (int, float)), isinstance(average_speed, (int, float)),
                isinstance(total_time, (int, float))]):
        return jsonify({'success': False, 'error': 'Invalid input values'})
    if count < 0 or average_speed < 0 or total_time < 0:
        return jsonify({'success': False, 'error': 'Input values cannot be negative'})
    try:
        model = user_models.get(session['email'])
        if not model:
            return jsonify({'success': False, 'error': 'No trained model available. Complete at least two sessions.'})
        X_new = np.array([[count, average_speed, total_time]])
        predicted_recovery = model.predict(X_new)[0]
        return jsonify({'success': True, 'predicted_recovery_time': round(predicted_recovery, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin')
def admin():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return redirect(url_for('login'))
    users = list(db.collection('users').stream())
    users = [user.to_dict() for user in users]
    for user in users:
        if 'email' not in user:
            user['email'] = ''
    return render_template('admin.html', users=users, now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/admin/update_session', methods=['POST'])
def update_session():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return jsonify({'success': False, 'error': 'Unauthorized access'})
    data = request.json
    user_email = data.get('email')
    session_index = data.get('session_index')
    date = data.get('date')
    count = data.get('count')
    total_time = data.get('total_time')
    if not all([user_email, session_index is not None, date, count is not None, total_time is not None]):
        return jsonify({'success': False, 'error': 'Missing required fields'})
    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})
        user_data = user_doc.to_dict()
        if 'sessions' not in user_data or len(user_data['sessions']) <= session_index:
            return jsonify({'success': False, 'error': 'Session not found'})
        user_data['sessions'][session_index] = {
            'date': date,
            'count': count,
            'total_time': total_time,
            'average_speed': total_time / count if count > 0 else 0
        }
        user_ref.update({'sessions': user_data['sessions']})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/delete_session', methods=['POST'])
def delete_session():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return jsonify({'success': False, 'error': 'Unauthorized access'})
    data = request.json
    user_email = data.get('email')
    session_index = data.get('session_index')
    if user_email is None or session_index is None:
        return jsonify({'success': False, 'error': 'Missing required fields'})
    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})
        user_data = user_doc.to_dict()
        if 'sessions' not in user_data or len(user_data['sessions']) <= session_index:
            return jsonify({'success': False, 'error': 'Session not found'})
        del user_data['sessions'][session_index]
        user_ref.update({'sessions': user_data['sessions']})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/add_session', methods=['POST'])
def add_session():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return jsonify({'success': False, 'error': 'Unauthorized access'})
    data = request.json
    user_email = data.get('email')
    date = data.get('date')
    count = data.get('count')
    total_time = data.get('total_time')
    if not all([user_email, date, count is not None, total_time is not None]):
        return jsonify({'success': False, 'error': 'Missing required fields'})
    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})
        average_speed = total_time / count if count > 0 else 0
        new_session = {
            'date': date,
            'count': count,
            'total_time': total_time,
            'average_speed': average_speed
        }
        user_ref.update({'sessions': firestore.ArrayUnion([new_session])})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_session', methods=['POST'])
def save_session():
    global session_data
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})
    if session_data["count"] == 0:
        return jsonify({'success': False, 'error': 'No exercise data to save'})
    total_time = time.time() - session_data["start_time"] if session_data["start_time"] else 0
    session_data_to_save = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": session_data["count"],
        "correct_count": session_data["correct_count"],
        "incorrect_count": session_data["incorrect_count"],
        "total_time": total_time,
        "average_speed": total_time / session_data["count"] if session_data["count"] > 0 else 0,
        "post_therapy": session_data["post_therapy"] if "post_therapy" in session_data else {}
    }
    user_ref = db.collection('users').document(session['email'])
    user_ref.update({"sessions": firestore.ArrayUnion([session_data_to_save])})
    return jsonify({'success': True})

@app.template_filter('mean')
def mean_filter(values):
    if not values:
        return 0
    return sum(values) / len(values)

app.jinja_env.filters['mean'] = mean_filter

@app.route('/recommendations')
def recommendations():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    if not user.exists or 'sessions' not in user.to_dict() or len(user.to_dict()['sessions']) == 0:
        return render_template('recommendation.html', error="No session data found.")
    sessions = user.to_dict()['sessions']
    avg_speed = np.mean([session['average_speed'] for session in sessions if session['average_speed'] > 0])
    recommendations = []
    if avg_speed < 1.5:
        recommendations.append("Your speed is too fast. Focus on controlled movements to improve form.")
    if avg_speed > 3.0:
        recommendations.append("Your speed is too slow. Try to increase your pace for better endurance.")
    if not recommendations:
        recommendations.append("Great job! Keep up the good work and aim for consistency.")
    return render_template('recommendation.html', recommendations=recommendations, user=user.to_dict(), avg_speed=avg_speed)

@app.route('/select_exercise', methods=['GET', 'POST'])
def select_exercise():
    if 'email' not in session:
        return redirect(url_for('login'))
    global session_data
    if request.method == 'POST':
        exercise_names = request.form.getlist('exercises[]')
        session_data["exercise"] = {name: exercises[name] for name in exercise_names}
        return redirect(url_for('training'))
    return render_template('select_exercise.html', exercises=exercises.keys())

@app.route('/training')
def training():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('training.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    global session_data
    return jsonify({'count': session_data["count"], 'target': session_data["target_count"],
                    'correct_count': session_data["correct_count"], 'incorrect_count': session_data["incorrect_count"],
                    'feedback': session_data["feedback_message"]})

@app.route('/set_target', methods=['POST'])
def set_target():
    global session_data
    data = request.json
    session_data["target_count"] = int(data.get('target', 0))
    session_data.update({"count": 0, "correct_count": 0, "incorrect_count": 0, "position": None,
                         "exercise_started": True, "start_time": None, "last_rep_time": None,
                         "feedback_message": "Begin Exercise!"})
    return jsonify({'success': True, 'target': session_data["target_count"]})

if __name__ == "__main__":
    app.run(debug=True)