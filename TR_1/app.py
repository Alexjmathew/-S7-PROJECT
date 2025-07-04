import os
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Firebase
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

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
rl_agents = {}
current_user = None

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
    
    # Fatigue detection model
    fatigue_model = Sequential([
        Dense(32, activation='relu', input_shape=(4,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    fatigue_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Quality scoring model
    quality_scorer = KMeans(n_clusters=3)  # 3 quality levels (poor, average, good)

initialize_models()

# RL Agent builder
def build_rl_agent(user_email):
    model = Sequential([
        Dense(24, activation='relu', input_shape=(6,)),
        Dense(24, activation='relu'),
        Dense(3, activation='linear')  # 3 actions
    ])
    
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    
    agent = DQNAgent(
        model=model,
        policy=policy,
        memory=memory,
        nb_actions=3,
        nb_steps_warmup=10,
        target_model_update=1e-2
    )
    
    agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    rl_agents[user_email] = agent
    return agent

# Helper functions
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def save_session_data():
    global count, start_time, current_user
    
    if not current_user or count == 0:
        return False
    
    total_time = time.time() - start_time if start_time else 0
    session_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": count,
        "total_time": total_time,
        "average_speed": total_time / count if count > 0 else 0,
        "exercise": exercise["name"] if exercise else "unknown"
    }
    
    try:
        user_ref = db.collection('users').document(current_user['email'])
        user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
        return True
    except Exception as e:
        print(f"Error saving session: {e}")
        return False

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'email' not in session:
                return redirect(url_for('login'))
            
            user_ref = db.collection('users').document(session['email'])
            user_data = user_ref.get().to_dict()
            
            if role not in user_data.get('roles', ['user']):
                return redirect(url_for('unauthorized'))
            
            return f(*args, **kwargs)
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
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            return render_template('login.html', error="Please fill all fields")
        
        try:
            user_ref = db.collection('users').document(email)
            user_data = user_ref.get().to_dict()
            
            if user_data and user_data['password'] == password:
                session['email'] = email
                session['username'] = user_data['username']
                global current_user
                current_user = user_data
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error="Invalid credentials")
        except Exception as e:
            return render_template('login.html', error=str(e))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        username = request.form.get('username')
        
        if not all([email, password, username]):
            return render_template('register.html', error="Please fill all fields")
        
        try:
            # Check if user exists
            user_ref = db.collection('users').document(email)
            if user_ref.get().exists:
                return render_template('register.html', error="Email already registered")
            
            # Create new user
            user_data = {
                'email': email,
                'password': password,
                'username': username,
                'roles': ['user'],
                'sessions': [],
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            user_ref.set(user_data)
            session['email'] = email
            session['username'] = username
            global current_user
            current_user = user_data
            
            return redirect(url_for('dashboard'))
        except Exception as e:
            return render_template('register.html', error=str(e))
    
    return render_template('register.html')

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
            'durations': [s['total_time'] for s in sessions]
        }
        
        return render_template('dashboard.html', 
                            user=user_data,
                            sessions=session_data,
                            exercises=EXERCISES)
    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/training', methods=['GET', 'POST'])
@login_required
def training():
    global exercise, exercise_started, count, target_count
    
    if request.method == 'POST':
        exercise_name = request.form.get('exercise')
        target = request.form.get('target', 10)
        
        if exercise_name in EXERCISES:
            exercise = EXERCISES[exercise_name]
            exercise['name'] = exercise_name
            target_count = int(target)
            exercise_started = False
            count = 0
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Invalid exercise'})
    
    return render_template('training.html')

@app.route('/video_feed')
@login_required
def video_feed():
    def generate_frames():
        global count, position, exercise_started, feedback_message, start_time, last_rep_time, exercise
        
        cap = cv2.VideoCapture(0)
        fatigue_window = []
        rep_quality_scores = []
        current_rep_angles = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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
                    landmark = getattr(mp_pose.PoseLandmark, joint)
                    coords.append([
                        landmarks[landmark.value].x,
                        landmarks[landmark.value].y
                    ])
                
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
                    rep_rom = max(current_rep_angles) - min(current_rep_angles)
                    form_deviation = np.std([a - exercise["target_angle"] for a in current_rep_angles])
                    
                    # Quality scoring
                    quality_features = [[rep_speed, rep_rom, form_deviation]]
                    quality_cluster = quality_scorer.fit_predict(quality_features)[0]
                    quality_score = 100 - (quality_cluster * 33)
                    rep_quality_scores.append(quality_score)
                    
                    # Fatigue detection
                    fatigue_features = [
                        len(fatigue_window)/10,
                        np.mean([r['speed'] for r in fatigue_window[-3:]]) if fatigue_window else 0,
                        np.std([r['rom'] for r in fatigue_window[-3:]]) if fatigue_window else 0,
                        quality_score
                    ]
                    
                    fatigue_prob = fatigue_model.predict([fatigue_features])[0][0]
                    fatigue_window.append({
                        'time': current_time,
                        'speed': rep_speed,
                        'rom': rep_rom,
                        'quality': quality_score
                    })
                    
                    # RL Adaptive Training
                    if current_user:
                        user_email = current_user['email']
                        if user_email not in rl_agents:
                            build_rl_agent(user_email)
                        
                        agent = rl_agents[user_email]
                        state = [
                            count/target_count,
                            np.mean(rep_quality_scores[-3:]) if rep_quality_scores else 0,
                            fatigue_prob,
                            len(fatigue_window)/10,
                            (current_time - start_time) if start_time else 0,
                            target_count
                        ]
                        
                        action = agent.forward(state)
                        
                        # Apply action
                        if action == 0:  # Increase difficulty
                            target_count = min(20, target_count + 2)
                            feedback_message = "Increased difficulty! Keep pushing!"
                        elif action == 1:  # Decrease difficulty
                            target_count = max(5, target_count - 2)
                            feedback_message = "Reduced difficulty. Focus on form."
                        else:  # Suggest rest
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
                    save_session_data()
                    feedback_message = f"Exercise complete! Time: {time.time()-start_time:.1f}s"
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
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_exercise', methods=['POST'])
@login_required
def start_exercise():
    global exercise_started, start_time, count
    exercise_started = True
    count = 0
    start_time = None
    return jsonify({'success': True})

@app.route('/get_progress')
@login_required
def get_progress():
    return jsonify({
        'count': count,
        'target': target_count,
        'feedback': feedback_message,
        'exercise': exercise['name'] if exercise else None
    })

@app.route('/logout')
def logout():
    session.clear()
    global current_user
    current_user = None
    return redirect(url_for('index'))

@app.route('/admin')
@role_required('admin')
def admin():
    users = list(db.collection('users').stream())
    users = [user.to_dict() for user in users]
    return render_template('admin.html', users=users)

@app.errorhandler(401)
def unauthorized(e):
    return render_template('error.html', message="Unauthorized access"), 401

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
