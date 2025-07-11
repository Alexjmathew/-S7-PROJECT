import matplotlib
matplotlib.use('Agg')  # Use thread-safe Agg backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template, Response, jsonify, flash, request, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from io import BytesIO
import base64
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('PIL').setLevel(logging.INFO)  # Suppress PIL debug logs

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SESSION_COOKIE_PARTITIONED'] = False  # Disable partitioned cookies to avoid TypeError

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
    "squats": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 90,
        "threshold": 15,
        "optimal_speed_range": (2.0, 4.0),
        "description": "Stand with feet shoulder-width apart, lower your body by bending knees."
    },
    "push_ups": {
        "joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
        "target_angle": 90,
        "threshold": 15,
        "optimal_speed_range": (1.5, 3.0),
        "description": "Keep your body straight, lower until elbows are at 90 degrees."
    },
    "lunges": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 90,
        "threshold": 15,
        "optimal_speed_range": (2.0, 3.5),
        "description": "Step forward and lower your hips until both knees are bent at 90 degrees."
    },
    "plank": {
        "joints": ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
        "target_angle": 180,
        "threshold": 10,
        "optimal_speed_range": (0.5, 1.0),
        "description": "Maintain a straight line from head to heels, engage your core."
    },
    "sit_ups": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 45,
        "threshold": 10,
        "optimal_speed_range": (1.5, 3.0),
        "description": "Lie on your back, bend knees, and lift your torso toward your thighs."
    },
    "shoulder_press": {
        "joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
        "target_angle": 180,
        "threshold": 15,
        "optimal_speed_range": (1.0, 2.0),
        "description": "Press weights overhead until arms are straight, then lower slowly."
    },
    "bicep_curls": {
        "joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
        "target_angle": 60,
        "threshold": 15,
        "optimal_speed_range": (1.5, 3.0),
        "description": "Keep elbows close to torso, curl weights up to shoulder level."
    },
    "leg_raises": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 45,
        "threshold": 10,
        "optimal_speed_range": (1.5, 3.0),
        "description": "Lie on your back, keep legs straight and raise them to 45 degrees."
    },
    "jumping_jacks": {
        "joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
        "target_angle": 180,
        "threshold": 15,
        "optimal_speed_range": (1.0, 2.0),
        "description": "Jump while moving arms overhead and legs outward, then return."
    },
    "triceps_dips": {
        "joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
        "target_angle": 90,
        "threshold": 15,
        "optimal_speed_range": (1.5, 3.0),
        "description": "Lower your body by bending elbows to 90 degrees, then push up."
    }
}

# Session data structure
class SessionData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.correct_count = 0
        self.incorrect_count = 0
        self.position = None
        self.exercise_started = False
        self.feedback_message = "Begin Exercise!"
        self.start_time = None
        self.last_rep_time = None
        self.current_exercise = None
        self.target_count = 0
        self.exercise_history = {ex: {"count": 0, "correct": 0, "incorrect": 0} for ex in exercises}
        self.difficulty_adjusted = False
        self.last_spoken_message = None

session_data = SessionData()

# Helper functions
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

@lru_cache(maxsize=32)
def generate_progress_chart(sessions_tuple):
    try:
        sessions = list(sessions_tuple)
        dates = [datetime.strptime(s['date'], "%Y-%m-%d %H:%M:%S") for s in sessions]
        correct_counts = [s['correct_count'] for s in sessions]
        incorrect_counts = [s['incorrect_count'] for s in sessions]

        plt.figure(figsize=(10, 5))
        plt.plot(dates, correct_counts, label='Correct Reps', marker='o')
        plt.plot(dates, incorrect_counts, label='Incorrect Reps', marker='o')
        plt.xlabel('Session Date')
        plt.ylabel('Count')
        plt.title('Exercise Progress Over Time')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error generating progress chart: {e}")
        return None

@lru_cache(maxsize=32)
def generate_progress_chart_for_exercise(exercise, stats_tuple):
    try:
        stats = list(stats_tuple)
        dates = [datetime.strptime(s['date'], "%Y-%m-%d %H:%M:%S") for s in stats]
        counts = [s['count'] for s in stats]
        correct = [s['correct'] for s in stats]
        incorrect = [s['incorrect'] for s in stats]

        plt.figure(figsize=(8, 4))
        plt.plot(dates, counts, label='Total Reps', marker='o')
        plt.plot(dates, correct, label='Correct Reps', marker='o')
        plt.plot(dates, incorrect, label='Incorrect Reps', marker='o')
        plt.title(f'{exercise.capitalize()} Progress')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error generating chart for {exercise}: {e}")
        return None

def adjust_difficulty():
    if session_data.count > 0:
        accuracy = (session_data.correct_count / session_data.count) * 100
        if accuracy > 80 and not session_data.difficulty_adjusted:
            session_data.target_count += 5
            session_data.feedback_message = f"Great job! Increasing target to {session_data.target_count} reps."
            session_data.difficulty_adjusted = True
            return "Difficulty increased due to high accuracy."
        elif accuracy < 50 and session_data.target_count > 5 and not session_data.difficulty_adjusted:
            session_data.target_count = max(5, session_data.target_count - 5)
            session_data.feedback_message = f"Let's make it easier. Target reduced to {session_data.target_count} reps."
            session_data.difficulty_adjusted = True
            return "Difficulty decreased due to low accuracy."
    return None

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and session_data.exercise_started and session_data.current_exercise:
            landmarks = results.pose_landmarks.landmark
            exercise = exercises[session_data.current_exercise]

            try:
                coords = []
                for joint in exercise["joints"]:
                    landmark = landmarks[getattr(mp_pose.PoseLandmark, joint).value]
                    coords.append([landmark.x, landmark.y])

                angle = calculate_angle(*coords)

                # Position detection
                if angle > exercise["target_angle"] + exercise["threshold"]:
                    session_data.position = "up"

                if session_data.position == "up" and angle < exercise["target_angle"] - exercise["threshold"]:
                    session_data.position = "down"
                    current_time = time.time()

                    if session_data.last_rep_time:
                        rep_time = current_time - session_data.last_rep_time
                        if exercise["optimal_speed_range"][0] <= rep_time <= exercise["optimal_speed_range"][1]:
                            session_data.correct_count += 1
                            session_data.exercise_history[session_data.current_exercise]["correct"] += 1
                            session_data.feedback_message = "Good form! Keep going."
                        else:
                            session_data.incorrect_count += 1
                            session_data.exercise_history[session_data.current_exercise]["incorrect"] += 1
                            session_data.feedback_message = "Adjust your speed."

                    session_data.count += 1
                    session_data.exercise_history[session_data.current_exercise]["count"] += 1
                    session_data.last_rep_time = current_time

                    if session_data.count == 1:
                        session_data.start_time = current_time

                    # Adaptive difficulty adjustment
                    difficulty_message = adjust_difficulty()
                    if difficulty_message:
                        logging.info(difficulty_message)

                # Angle feedback
                if angle < exercise["target_angle"] - exercise["threshold"]:
                    session_data.feedback_message = "Go deeper! Lower your angle."
                elif angle > exercise["target_angle"] + exercise["threshold"]:
                    session_data.feedback_message = "Not deep enough! Increase your angle."

                # Voice feedback
                if session_data.feedback_message != session_data.last_spoken_message:
                    session_data.last_spoken_message = session_data.feedback_message

                # Display information
                cv2.putText(image, f'Exercise: {session_data.current_exercise}', (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f'Angle: {int(angle)}°', (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f'Count: {session_data.count}/{session_data.target_count}', (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(image, f'Correct: {session_data.correct_count}', (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f'Incorrect: {session_data.incorrect_count}', (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, session_data.feedback_message, (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Check if target reached
                if session_data.count >= session_data.target_count:
                    session_data.feedback_message = "Great job! Exercise complete!"
                    session_data.last_spoken_message = session_data.feedback_message
                    complete_exercise()

            except Exception as e:
                logging.error(f"Error processing frame: {e}")

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def complete_exercise():
    session_data.exercise_started = False
    total_time = time.time() - session_data.start_time if session_data.start_time else 0
    session_data.feedback_message = f"Exercise Complete! Time: {total_time:.2f}s"

    if 'email' in session:
        session_record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "exercise": session_data.current_exercise,
            "count": session_data.count,
            "correct_count": session_data.correct_count,
            "incorrect_count": session_data.incorrect_count,
            "total_time": total_time,
            "average_speed": total_time / session_data.count if session_data.count > 0 else 0
        }

        user_ref = db.collection('users').document(session['email'])
        user_ref.update({
            "sessions": firestore.ArrayUnion([session_record]),
            f"exercise_stats.{session_data.current_exercise}": firestore.ArrayUnion([{
                "date": session_record["date"],
                "count": session_record["count"],
                "correct": session_record["correct_count"],
                "incorrect": session_record["incorrect_count"]
            }])
        })

    # Reset counters
    session_data.count = 0
    session_data.correct_count = 0
    session_data.incorrect_count = 0
    session_data.position = None
    session_data.start_time = None
    session_data.last_rep_time = None
    session_data.difficulty_adjusted = False
    session_data.last_spoken_message = None

# Routes
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('profile'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        users_ref = db.collection('users').where('email', '==', email).limit(1)
        users = list(users_ref.stream())

        if users and users[0].to_dict().get('password') == password:
            session['email'] = email
            session['username'] = users[0].to_dict().get('username')
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
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        age = request.form.get('age')
        height = request.form.get('height')
        weight = request.form.get('weight')

        email_query = db.collection('users').where('email', '==', email).limit(1)
        if list(email_query.stream()):
            return render_template('register.html', error="Email already registered.")

        user_ref = db.collection('users').document(email)
        user_data = {
            "username": username,
            "email": email,
            "password": password,
            "age": age,
            "height": height,
            "weight": weight,
            "sessions": [],
            "exercise_stats": {ex: [] for ex in exercises},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

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
    exercise_stats = user_data.get('exercise_stats', {})

    # Prepare exercise statistics
    exercise_data = []
    for ex, stats in exercise_stats.items():
        if stats:
            total_count = sum(s.get('count', 0) for s in stats)
            total_correct = sum(s.get('correct', 0) for s in stats)
            total_incorrect = sum(s.get('incorrect', 0) for s in stats)
            exercise_data.append({
                "name": ex,
                "total_count": total_count,
                "total_correct": total_correct,
                "total_incorrect": total_incorrect,
                "accuracy": (total_correct / total_count * 100) if total_count > 0 else 0
            })

    # Generate progress chart
    chart_img = None
    if len(sessions) > 0:
        chart_img = generate_progress_chart(tuple(sessions))  # Convert to tuple for lru_cache

    return render_template('profile.html',
                           user=user_data,
                           exercises=exercise_data,
                           exercise_descriptions=exercises,
                           chart_img=chart_img)

@app.route('/leaderboard')
def leaderboard():
    if 'email' not in session:
        return redirect(url_for('login'))

    leaderboards = {}
    for exercise in exercises:
        users_ref = db.collection('users')
        users = users_ref.stream()

        exercise_data = []
        for user in users:
            user_data = user.to_dict()
            stats = user_data.get('exercise_stats', {}).get(exercise, [])
            if stats:
                total_correct = sum(s['correct'] for s in stats)
                total_count = sum(s['count'] for s in stats)
                exercise_data.append({
                    "username": user_data.get('username', 'Unknown'),
                    "count": total_count,
                    "correct": total_correct,
                    "accuracy": (total_correct / total_count * 100) if total_count > 0 else 0
                })

        exercise_data.sort(key=lambda x: x['correct'], reverse=True)
        leaderboards[exercise] = exercise_data[:10]

    return render_template('leaderboard.html', leaderboards=leaderboards, exercises=exercises)

@app.route('/progress')
def progress():
    if 'email' not in session:
        return redirect(url_for('login'))

    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()

    if not user.exists:
        return redirect(url_for('login'))

    user_data = user.to_dict()
    exercise_stats = user_data.get('exercise_stats', {})
    chart_data = {}

    try:
        for ex, stats in exercise_stats.items():
            if stats:
                chart_data[ex] = generate_progress_chart_for_exercise(ex, tuple(stats))
    except Exception as e:
        logging.error(f"Error generating progress charts: {e}")
        flash('Error generating progress charts.', 'error')

    return render_template('progress.html', charts=chart_data, exercises=exercises)

@app.route('/recommendation')
def recommendation():
    if 'email' not in session:
        return redirect(url_for('login'))

    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()

    if not user.exists:
        return redirect(url_for('login'))

    user_data = user.to_dict()
    exercise_stats = user_data.get('exercise_stats', {})
    recommendations = []

    for ex, stats in exercise_stats.items():
        if stats:
            total_count = sum(s.get('count', 0) for s in stats)
            total_correct = sum(s.get('correct', 0) for s in stats)
            accuracy = (total_correct / total_count * 100) if total_count > 0 else 0
            if accuracy > 80:
                recommendations.append(f"Great job on {ex}! Try increasing reps to challenge yourself.")
            elif accuracy < 50:
                recommendations.append(f"Consider practicing {ex} with fewer reps to focus on form.")
            else:
                recommendations.append(f"You're doing well with {ex}. Keep it up!")

    if not recommendations:
        recommendations.append("Try starting with easier exercises like squats or push-ups to build confidence.")

    return render_template('recommendation.html', recommendations=recommendations, exercises=exercises)

@app.route('/select_exercise', methods=['GET', 'POST'])
def select_exercise():
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        exercise = request.form.get('exercise')
        target = request.form.get('target', 0)
        logging.debug(f"Received exercise: {exercise}, target: {target}")

        if not exercise or exercise not in exercises:
            logging.error(f"Invalid exercise: {exercise}, available exercises: {list(exercises.keys())}")
            flash('Please select a valid exercise', 'error')
            return redirect(url_for('select_exercise'))

        try:
            target = int(target)
        except (ValueError, TypeError):
            logging.error(f"Invalid target count: {target}")
            flash('Please enter a valid target count', 'error')
            return redirect(url_for('select_exercise'))

        if target <= 0:
            logging.error(f"Target count <= 0: {target}")
            flash('Please enter a valid target count', 'error')
            return redirect(url_for('select_exercise'))

        # Reset session data but preserve exercise and target
        session_data.reset()
        session_data.current_exercise = exercise
        session_data.target_count = target
        logging.debug(f"Redirecting to training with exercise: {exercise}, target: {target}")

        return redirect(url_for('training'))

    return render_template('select_exercise.html', exercises=exercises.keys())

@app.route('/training')
def training():
    if 'email' not in session:
        return redirect(url_for('login'))

    logging.debug(f"Training route: current_exercise={session_data.current_exercise}, target_count={session_data.target_count}")

    if not hasattr(session_data, 'current_exercise'):
        session_data.current_exercise = None
    if not hasattr(session_data, 'target_count'):
        session_data.target_count = 0

    if not session_data.current_exercise or session_data.target_count <= 0:
        logging.error("Validation failed: No exercise selected or invalid target count")
        flash('Please select an exercise and set your target count first', 'warning')
        return redirect(url_for('select_exercise'))

    if session_data.current_exercise not in exercises:
        logging.error(f"Validation failed: Invalid exercise {session_data.current_exercise}")
        flash('Invalid exercise selected', 'error')
        return redirect(url_for('select_exercise'))

    exercise = exercises[session_data.current_exercise]
    return render_template('training.html',
                           exercise=session_data.current_exercise,
                           description=exercise['description'],
                           target=session_data.target_count)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify({
        'count': session_data.count,
        'target': session_data.target_count,
        'correct': session_data.correct_count,
        'incorrect': session_data.incorrect_count,
        'feedback': session_data.feedback_message,
        'exercise': session_data.current_exercise,
        'voice_message': session_data.feedback_message
    })

@app.route('/start_exercise', methods=['POST'])
def start_exercise():
    session_data.exercise_started = True
    session_data.start_time = time.time()
    session_data.feedback_message = f"Starting {session_data.current_exercise}! Let's go!"
    session_data.last_spoken_message = session_data.feedback_message
    return jsonify({'success': True})

@app.route('/end_exercise', methods=['POST'])
def end_exercise():
    complete_exercise()
    return jsonify({'success': True})

if __name__ == "__main__":
    app.run(debug=True)