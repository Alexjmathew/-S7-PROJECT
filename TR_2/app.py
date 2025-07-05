import os
import logging
import uuid
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from functools import wraps
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Firebase
try:
    cred = credentials.Certificate('firebase.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {e}")
    raise

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variables
count = 0
target_count = 10
position = None
exercise_started = False
feedback_message = "Ready to begin exercise!"
start_time = None
last_rep_time = None
exercise = None
fatigue_model = None
quality_scorer = None
decision_makers = {}  # Replaces rl_agents
current_user = None
camera_index = 0

# Exercise configurations
EXERCISES = {
    "knee_raises": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 60,
        "threshold": 15,
        "optimal_speed_range": (1.0, 2.5)
    },
    "squats": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 90,
        "threshold": 15,
        "optimal_speed_range": (2.0, 4.0)
    }
}


# Initialize ML models
def initialize_models():
    global fatigue_model, quality_scorer
    try:
        # Fatigue detection model (Random Forest)
        fatigue_model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Train with dummy data to initialize (replace with real data if available)
        X_dummy = np.array([[0.1, 2.0, 10.0, 80.0], [0.5, 1.5, 15.0, 60.0], [0.9, 1.0, 20.0, 40.0]])
        y_dummy = np.array([0, 0, 1])
        fatigue_model.fit(X_dummy, y_dummy)

        # Quality scoring model (unchanged, uses KMeans)
        quality_scorer = KMeans(n_clusters=3, random_state=42)
        logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML models: {e}")
        raise


initialize_models()


# Decision Maker (replaces RL agent)
def build_decision_maker(user_email):
    try:
        # Simple rule-based decision maker
        def make_decision(state):
            completion_ratio, avg_quality, fatigue_prob, rep_rate, session_duration, target = state
            # Rule 1: Increase difficulty if low fatigue and high quality
            if fatigue_prob < 0.3 and avg_quality > 80 and completion_ratio < 0.8:
                return 0  # Increase difficulty
            # Rule 2: Decrease difficulty if high fatigue or low quality
            elif fatigue_prob > 0.7 or avg_quality < 50:
                return 1  # Decrease difficulty
            # Rule 3: Suggest rest if high fatigue and long session
            elif fatigue_prob > 0.6 and session_duration > 300:  # 5 minutes
                return 2  # Suggest rest
            return None  # No action

        decision_makers[user_email] = make_decision
        logger.info(f"Decision maker built for user: {user_email}")
        return make_decision
    except Exception as e:
        logger.error(f"Failed to build decision maker for {user_email}: {e}")
        raise


# Helper functions
def calculate_angle(a, b, c):
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)
    except Exception as e:
        logger.error(f"Error calculating angle: {e}")
        return 0


def save_session_data():
    global count, start_time, current_user, exercise
    if not current_user or count == 0 or not exercise:
        logger.warning("Cannot save session: missing user, count, or exercise")
        return False

    try:
        total_time = time.time() - start_time if start_time else 0
        session_data = {
            "session_id": str(uuid.uuid4()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": count,
            "total_time": total_time,
            "average_speed": total_time / count if count > 0 else 0,
            "exercise": exercise["name"],
            "fatigue_data": []  # Placeholder for fatigue data
        }

        user_ref = db.collection('users').document(current_user['email'])
        user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
        logger.info(f"Session saved for user: {current_user['email']}")
        return True
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return False


# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            logger.warning("Unauthorized access attempt")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'email' not in session:
                logger.warning("Unauthorized access attempt")
                return redirect(url_for('login'))

            try:
                user_ref = db.collection('users').document(session['email'])
                user_data = user_ref.get().to_dict()

                if not user_data or role not in user_data.get('roles', ['user']):
                    logger.warning(f"Role {role} not found for user {session['email']}")
                    return redirect(url_for('unauthorized'))

                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error checking role: {e}")
                return redirect(url_for('unauthorized'))

        return decorated_function

    return decorator


# Routes
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            logger.warning("Login attempt with missing fields")
            return render_template('login.html', error="Please fill all fields")

        try:
            user_ref = db.collection('users').document(email)
            user_data = user_ref.get().to_dict()

            if user_data and user_data.get('password') == password:
                session['email'] = email
                session['username'] = user_data['username']
                global current_user
                current_user = user_data
                logger.info(f"User logged in: {email}")
                return redirect(url_for('dashboard'))
            else:
                logger.warning(f"Invalid login attempt for {email}")
                return render_template('login.html', error="Invalid credentials")
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template('login.html', error="An error occurred")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        username = request.form.get('username', '').strip()

        if not all([email, password, username]):
            logger.warning("Registration attempt with missing fields")
            return render_template('register.html', error="Please fill all fields")

        if len(password) < 8:
            logger.warning("Registration attempt with weak password")
            return render_template('register.html', error="Password must be at least 8 characters")

        try:
            user_ref = db.collection('users').document(email)
            if user_ref.get().exists:
                logger.warning(f"Registration attempt with existing email: {email}")
                return render_template('register.html', error="Email already registered")

            user_data = {
                'email': email,
                'password': password,
                'username': username,
                'roles': ['user'],
                'sessions': [],
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'profile': {
                    'age': None,
                    'height': None,
                    'weight': None
                }
            }

            user_ref.set(user_data)
            session['email'] = email
            session['username'] = username
            global current_user
            current_user = user_data
            logger.info(f"User registered: {email}")
            return redirect(url_for('dashboard'))
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return render_template('register.html', error="An error occurred")

    return render_template('register.html')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    try:
        user_ref = db.collection('users').document(session['email'])
        user_data = user_ref.get().to_dict()

        if request.method == 'POST':
            age = request.form.get('age')
            height = request.form.get('height')
            weight = request.form.get('weight')

            try:
                profile_data = {
                    'profile': {
                        'age': int(age) if age else None,
                        'height': float(height) if height else None,
                        'weight': float(weight) if weight else None
                    }
                }
                user_ref.update(profile_data)
                logger.info(f"Profile updated for user: {session['email']}")
                return redirect(url_for('profile'))
            except Exception as e:
                logger.error(f"Profile update error: {e}")
                return render_template('profile.html', user=user_data, error="Failed to update profile")

        return render_template('profile.html', user=user_data)
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return render_template('error.html', message="An error occurred")


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        user_ref = db.collection('users').document(session['email'])
        user_data = user_ref.get().to_dict()

        sessions = user_data.get('sessions', [])
        session_data = {
            'dates': [s['date'] for s in sessions],
            'counts': [s['count'] for s in sessions],
            'durations': [s['total_time'] for s in sessions],
            'exercises': [s['exercise'] for s in sessions]
        }

        return render_template('dashboard.html',
                               user=user_data,
                               sessions=session_data,
                               exercises=EXERCISES)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('error.html', message="An error occurred")


@app.route('/training', methods=['GET', 'POST'])
@login_required
def training():
    global exercise, exercise_started, count, target_count

    if request.method == 'POST':
        exercise_name = request.form.get('exercise')
        target = request.form.get('target', 10)

        try:
            if exercise_name in EXERCISES:
                exercise = EXERCISES[exercise_name]
                exercise['name'] = exercise_name
                target_count = int(target)
                exercise_started = False
                count = 0
                logger.info(f"Training session started: {exercise_name}, target: {target_count}")
                return jsonify({'success': True})
            logger.warning(f"Invalid exercise selected: {exercise_name}")
            return jsonify({'success': False, 'error': 'Invalid exercise'})
        except Exception as e:
            logger.error(f"Training setup error: {e}")
            return jsonify({'success': False, 'error': 'An error occurred'})

    return render_template('training.html')


@app.route('/video_feed')
@login_required
def video_feed():
    def generate_frames():
        global count, position, exercise_started, feedback_message, start_time, last_rep_time, exercise

        cap = None
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logger.error(f"Failed to open camera at index {camera_index}")
                return

            fatigue_window = []
            rep_quality_scores = []
            current_rep_angles = []

            while exercise_started:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks and exercise_started and exercise:
                    landmarks = results.pose_landmarks.landmark

                    # Get joint coordinates
                    joints = exercise["joints"]
                    coords = []
                    for joint in joints:
                        try:
                            landmark = getattr(mp_pose.PoseLandmark, joint)
                            coords.append([
                                landmarks[landmark.value].x,
                                landmarks[landmark.value].y
                            ])
                        except Exception as e:
                            logger.error(f"Error processing landmark {joint}: {e}")
                            continue

                    if len(coords) == 3:
                        # Calculate angle
                        angle = calculate_angle(*coords)
                        current_time = time.time()
                        current_rep_angles.append(angle)

                        # Rep counting logic
                        if angle > exercise["target_angle"] + exercise["threshold"]:
                            position = "up"
                        if position == "up" and angle < exercise["target_angle"] - exercise["threshold"]:
                            position = "down"
                            count += 1

                            # Calculate rep quality
                            rep_speed = current_time - last_rep_time if last_rep_time else 0
                            rep_rom = max(current_rep_angles) - min(current_rep_angles) if current_rep_angles else 0
                            form_deviation = np.std(
                                [a - exercise["target_angle"] for a in current_rep_angles]) if current_rep_angles else 0

                            # Quality scoring
                            quality_features = [[rep_speed, rep_rom, form_deviation]]
                            quality_cluster = quality_scorer.predict(quality_features)[0]
                            quality_score = 100 - (quality_cluster * 33)
                            rep_quality_scores.append(quality_score)

                            # Fatigue detection
                            fatigue_features = [
                                len(fatigue_window) / 10,
                                np.mean([r['speed'] for r in fatigue_window[-3:]]) if fatigue_window else 0,
                                np.std([r['rom'] for r in fatigue_window[-3:]]) if fatigue_window else 0,
                                quality_score
                            ]

                            fatigue_prob = fatigue_model.predict_proba(np.array([fatigue_features]))[0][1]
                            fatigue_window.append({
                                'time': current_time,
                                'speed': rep_speed,
                                'rom': rep_rom,
                                'quality': quality_score,
                                'fatigue_prob': float(fatigue_prob)
                            })

                            # Adaptive Training (replaces RL)
                            if current_user:
                                user_email = current_user['email']
                                if user_email not in decision_makers:
                                    build_decision_maker(user_email)

                                decision_maker = decision_makers[user_email]
                                state = [
                                    count / target_count,
                                    np.mean(rep_quality_scores[-3:]) if rep_quality_scores else 0,
                                    fatigue_prob,
                                    len(fatigue_window) / 10,
                                    (current_time - start_time) if start_time else 0,
                                    target_count
                                ]

                                action = decision_maker(state)

                                # Apply action
                                if action == 0:  # Increase difficulty
                                    target_count = min(20, target_count + 2)
                                    feedback_message = "Increased difficulty! Keep pushing!"
                                elif action == 1:  # Decrease difficulty
                                    target_count = max(5, target_count - 2)
                                    feedback_message = "Reduced difficulty. Focus on form."
                                elif action == 2:  # Suggest rest
                                    feedback_message = "Consider a short break!"

                            # Reset for next rep
                            last_rep_time = current_time
                            current_rep_angles = []

                            # Fatigue feedback
                            if fatigue_prob > 0.7:
                                feedback_message = "High fatigue detected! Consider resting."

                        # Start timer on first rep
                        if count == 1 and not start_time:
                            start_time = time.time()

                        # Check for completion
                        if count >= target_count:
                            exercise_started = False
                            if save_session_data():
                                feedback_message = f"Exercise complete! Time: {time.time() - start_time:.1f}s"
                            else:
                                feedback_message = "Exercise complete, but failed to save session."
                            start_time = None

                        # Draw feedback on frame
                        cv2.putText(image, f'Reps: {count}/{target_count}', (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, feedback_message, (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    logger.warning("Failed to encode frame")
                    continue
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            logger.error(f"Video feed error: {e}")
        finally:
            if cap:
                cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_exercise', methods=['POST'])
@login_required
def start_exercise():
    global exercise_started, start_time, count, last_rep_time
    try:
        exercise_started = True
        count = 0
        start_time = None
        last_rep_time = None
        logger.info("Exercise started")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Start exercise error: {e}")
        return jsonify({'success': False, 'error': 'An error occurred'})


@app.route('/get_progress')
@login_required
def get_progress():
    try:
        return jsonify({
            'count': count,
            'target': target_count,
            'feedback': feedback_message,
            'exercise': exercise['name'] if exercise else None
        })
    except Exception as e:
        logger.error(f"Get progress error: {e}")
        return jsonify({'success': False, 'error': 'An error occurred'})


@app.route('/switch_camera', methods=['POST'])
@login_required
def switch_camera():
    global camera_index
    try:
        camera_index = int(request.form.get('camera_index', 0))
        logger.info(f"Switched to camera index: {camera_index}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Switch camera error: {e}")
        return jsonify({'success': False, 'error': 'Invalid camera index'})


@app.route('/exercise_history')
@login_required
def exercise_history():
    try:
        user_ref = db.collection('users').document(session['email'])
        user_data = user_ref.get().to_dict()
        sessions = user_data.get('sessions', [])
        return jsonify({'sessions': sessions})
    except Exception as e:
        logger.error(f"Exercise history error: {e}")
        return jsonify({'success': False, 'error': 'An error occurred'})


@app.route('/fatigue_safety')
@login_required
def fatigue_safety():
    try:
        user_ref = db.collection('users').document(session['email'])
        user_data = user_ref.get().to_dict()
        sessions = user_data.get('sessions', [])

        # Aggregate fatigue data
        fatigue_data = []
        for session in sessions:
            if 'fatigue_data' in session and session['fatigue_data']:
                for data_point in session['fatigue_data']:
                    fatigue_data.append({
                        'date': session['date'],
                        'exercise': session['exercise'],
                        'fatigue_prob': data_point['fatigue_prob']
                    })

        safety_tips = [
            "Take breaks when fatigue probability exceeds 0.7.",
            "Ensure proper hydration during workouts.",
            "Maintain a stable surface for balance during exercises.",
            "Stop immediately if you feel dizzy or unwell."
        ]

        return render_template('fatigue_safety.html',
                               user=user_data,
                               fatigue_data=fatigue_data,
                               safety_tips=safety_tips)
    except Exception as e:
        logger.error(f"Fatigue safety error: {e}")
        return render_template('error.html', message="An error occurred")


@app.route('/quality_dashboard')
@login_required
def quality_dashboard():
    try:
        user_ref = db.collection('users').document(session['email'])
        user_data = user_ref.get().to_dict()
        sessions = user_data.get('sessions', [])

        # Aggregate quality data
        quality_data = []
        for session in sessions:
            if 'fatigue_data' in session and session['fatigue_data']:
                for data_point in session['fatigue_data']:
                    quality_data.append({
                        'date': session['date'],
                        'exercise': session['exercise'],
                        'quality_score': data_point['quality']
                    })

        return render_template('quality_dashboard.html',
                               user=user_data,
                               quality_data=quality_data)
    except Exception as e:
        logger.error(f"Quality dashboard error: {e}")
        return render_template('error.html', message="An error occurred")


@app.route('/logout')
def logout():
    try:
        session.clear()
        global current_user
        current_user = None
        logger.info("User logged out")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return render_template('error.html', message="An error occurred")


@app.route('/admin')
@role_required('admin')
def admin():
    try:
        users = list(db.collection('users').stream())
        users = [user.to_dict() for user in users]
        logger.info("Admin panel accessed")
        return render_template('admin.html', users=users)
    except Exception as e:
        logger.error(f"Admin panel error: {e}")
        return render_template('error.html', message="An error occurred")


@app.errorhandler(401)
def unauthorized(e):
    logger.warning("Unauthorized access")
    return render_template('error.html', message="Unauthorized access"), 401


@app.errorhandler(404)
def page_not_found(e):
    logger.warning("Page not found")
    return render_template('error.html', message="Page not found"), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)