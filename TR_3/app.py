from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import bcrypt

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_secret_key_here')

# Initialize Firebase
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Global model variables
fatigue_lstm_model = None
quality_rf_model = None
rep_recommendation_model = None
movement_signature_model = None
signature_scaler = None

# Load machine learning models
def load_models():
    global fatigue_lstm_model, quality_rf_model, rep_recommendation_model, movement_signature_model, signature_scaler
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
        # Initialize dummy models (replace with trained models in production)
        fatigue_lstm_model = Sequential([
            LSTM(64, input_shape=(10, 5)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        fatigue_lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
        fatigue_lstm_model.save(os.path.join(models_dir, 'fatigue_lstm.h5'))
        
        quality_rf_model = RandomForestClassifier(n_estimators=100)
        joblib.dump(quality_rf_model, os.path.join(models_dir, 'quality_rf.pkl'))
        
        rep_recommendation_model = RandomForestRegressor(n_estimators=50)
        joblib.dump(rep_recommendation_model, os.path.join(models_dir, 'rep_recommender.pkl'))
        
        movement_signature_model = Sequential([
            Dense(32, activation='relu', input_shape=(15,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(15, activation='linear')
        ])
        movement_signature_model.compile(optimizer='adam', loss='mse')
        movement_signature_model.save(os.path.join(models_dir, 'movement_signature.h5'))
        
        signature_scaler = StandardScaler()
        joblib.dump(signature_scaler, os.path.join(models_dir, 'signature_scaler.pkl'))
    else:
        try:
            fatigue_lstm_model = load_model(os.path.join(models_dir, 'fatigue_lstm.h5'))
            quality_rf_model = joblib.load(os.path.join(models_dir, 'quality_rf.pkl'))
            rep_recommendation_model = joblib.load(os.path.join(models_dir, 'rep_recommender.pkl'))
            movement_signature_model = load_model(os.path.join(models_dir, 'movement_signature.h5'))
            signature_scaler = joblib.load(os.path.join(models_dir, 'signature_scaler.pkl'))
        except Exception as e:
            print(f"Error loading models: {e}")
            # Initialize dummy models as fallback
            fatigue_lstm_model = Sequential([
                LSTM(64, input_shape=(10, 5)),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            fatigue_lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
            quality_rf_model = RandomForestClassifier(n_estimators=100)
            rep_recommendation_model = RandomForestRegressor(n_estimators=50)
            movement_signature_model = Sequential([
                Dense(32, activation='relu', input_shape=(15,)),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(16, activation='relu'),
                Dense(32, activation='relu'),
                Dense(15, activation='linear')
            ])
            movement_signature_model.compile(optimizer='adam', loss='mse')
            signature_scaler = StandardScaler()

load_models()

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Predict fatigue using LSTM
def predict_fatigue(features_sequence):
    if not fatigue_lstm_model:
        return 0.5
    try:
        seq_array = np.array([
            [f['speed'], f['angle_dev'], f['tremor'], f['time_elapsed'], f['rep_count']]
            for f in features_sequence
        ])
        if len(seq_array) > 10:
            seq_array = seq_array[-10:]
        elif len(seq_array) < 10:
            padding = np.zeros((10 - len(seq_array), 5))
            seq_array = np.vstack([padding, seq_array])
        seq_array = seq_array.reshape(1, 10, 5)
        return float(fatigue_lstm_model.predict(seq_array, verbose=0)[0][0])
    except Exception as e:
        print(f"Error predicting fatigue: {e}")
        return 0.5

# Predict rep quality using Random Forest
def predict_quality(features):
    if not quality_rf_model:
        return 0.7
    try:
        features_array = np.array([
            features['joint_angles'][0],
            features['speed_consistency'],
            features['rom_achieved'],
            features['form_deviation']
        ]).reshape(1, -1)
        return quality_rf_model.predict_proba(features_array)[0][1]
    except Exception as e:
        print(f"Error predicting quality: {e}")
        return 0.7

# Recommend rep count
def recommend_rep_count(user_data, exercise_type):
    if not rep_recommendation_model:
        if user_data.get('age', 0) < 18:
            return 8
        elif user_data.get('age', 0) < 40:
            return 12
        elif user_data.get('age', 0) < 60:
            return 10
        else:
            return 5
    try:
        features = np.array([
            user_data.get('age', 30),
            user_data.get('fitness_level', 3),
            1 if exercise_type == 'beginner' else 2,
            user_data.get('last_session_quality', 0.7),
            user_data.get('bmi', 23)
        ]).reshape(1, -1)
        return int(round(rep_recommendation_model.predict(features)[0]))
    except Exception as e:
        print(f"Error recommending rep count: {e}")
        return 10

# Analyze movement signature
def analyze_movement_signature(landmarks_sequence):
    if not movement_signature_model or not signature_scaler:
        return 0.0
    try:
        features = []
        for frame in landmarks_sequence:
            frame_features = []
            for joint in ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
                         'RIGHT_ELBOW', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 
                         'RIGHT_KNEE']:
                landmark = frame.landmark[getattr(mp_pose.PoseLandmark, joint)]
                frame_features.extend([landmark.x, landmark.y])
            features.append(frame_features[:15])
        features_scaled = signature_scaler.transform(features)
        reconstructed = movement_signature_model.predict(features_scaled, verbose=0)
        error = np.mean(np.square(features_scaled - reconstructed))
        return error
    except Exception as e:
        print(f"Error analyzing movement signature: {e}")
        return 0.0

# Check for notifications
def check_notifications(fatigue_level, quality_scores, movement_error):
    notifications = []
    if fatigue_level > 0.8:
        notifications.append({
            'type': 'fatigue',
            'message': 'High fatigue detected. Consider taking a break.',
            'severity': 'high'
        })
    elif fatigue_level > 0.6:
        notifications.append({
            'type': 'fatigue',
            'message': 'Moderate fatigue detected. Monitor your form.',
            'severity': 'medium'
        })
    if len(quality_scores) > 5 and np.mean(quality_scores[-3:]) < 0.5:
        notifications.append({
            'type': 'quality',
            'message': 'Recent reps show poor form. Consider reviewing technique.',
            'severity': 'medium'
        })
    if movement_error > 0.5:
        notifications.append({
            'type': 'movement',
            'message': 'Unusual movement pattern detected. May indicate injury risk.',
            'severity': 'high'
        })
    return notifications

# Generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Session-specific variables (avoid global variables)
    session_data = {
        'count': 0,
        'target_count': 0,
        'position': None,
        'exercise_started': False,
        'feedback_message': "Begin Exercise!",
        'start_time': None,
        'last_rep_time': None,
        'exercise': None,
        'session_features': {
            'start_time': time.time(),
            'rep_times': [],
            'angle_deviations': [],
            'speeds': []
        },
        'fatigue_window': [],
        'quality_scores': [],
        'landmarks_sequence': [],
        'notification_queue': [],
        'last_notification_check': time.time()
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and session_data['exercise_started'] and session_data['exercise']:
            landmarks = results.pose_landmarks.landmark
            session_data['landmarks_sequence'].append(results.pose_landmarks)
            if len(session_data['landmarks_sequence']) > 30:
                session_data['landmarks_sequence'].pop(0)

            joints = session_data['exercise']["joints"]
            try:
                coords = [
                    [landmarks[getattr(mp_pose.PoseLandmark, joint).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, joint).value].y]
                    for joint in joints
                ]
                angle = calculate_angle(*coords)
            except Exception as e:
                print(f"Error calculating angle: {e}")
                continue

            current_time = time.time()
            time_elapsed = current_time - session_data['session_features']['start_time']

            # Calculate speed and tremor
            speed = 0
            if session_data['last_rep_time'] and session_data['count'] > 0:
                speed = 1 / (current_time - session_data['last_rep_time'])
                session_data['session_features']['speeds'].append(speed)
            
            tremor = 0.0
            if len(session_data['landmarks_sequence']) > 1:
                try:
                    tremor = np.mean([
                        abs(a.x - b.x) + abs(a.y - b.y)
                        for a, b in zip(session_data['landmarks_sequence'][-1].landmark, 
                                      session_data['landmarks_sequence'][-2].landmark)
                    ])
                except Exception as e:
                    print(f"Error calculating tremor: {e}")

            angle_deviation = abs(angle - session_data['exercise']["target_angle"])
            session_data['session_features']['angle_deviations'].append(angle_deviation)

            # Rep counting logic
            if angle > session_data['exercise']["target_angle"] + session_data['exercise']["threshold"]:
                session_data['position'] = "up"
            if session_data['position'] == "up" and angle < session_data['exercise']["target_angle"] - session_data['exercise']["threshold"]:
                session_data['position'] = "down"
                session_data['count'] += 1
                session_data['session_features']['rep_times'].append(current_time)

                # Prepare features for quality and fatigue
                quality_features = {
                    'joint_angles': [angle],
                    'speed_consistency': np.std(session_data['session_features']['speeds'][-3:]) if len(session_data['session_features']['speeds']) >= 3 else 0,
                    'rom_achieved': min(1.0, angle / session_data['exercise']["target_angle"]),
                    'form_deviation': angle_deviation / session_data['exercise']["threshold"]
                }
                quality_score = predict_quality(quality_features)
                session_data['quality_scores'].append(quality_score)

                fatigue_features = {
                    'speed': speed,
                    'angle_dev': angle_deviation,
                    'tremor': tremor,
                    'time_elapsed': time_elapsed,
                    'rep_count': session_data['count']
                }
                session_data['fatigue_window'].append(fatigue_features)
                if len(session_data['fatigue_window']) > 10:
                    session_data['fatigue_window'].pop(0)
                fatigue_score = predict_fatigue(session_data['fatigue_window'])

                # Update feedback based on quality and fatigue
                if quality_score < 0.5:
                    session_data['feedback_message'] = "Poor form! Focus on technique."
                elif quality_score < 0.7:
                    session_data['feedback_message'] = "Good, but can improve form."
                else:
                    session_data['feedback_message'] = "Excellent form! Keep going."
                
                if fatigue_score > 0.7:
                    session_data['feedback_message'] += " High fatigue detected. Consider resting soon."
                elif fatigue_score > 0.5:
                    session_data['feedback_message'] += " Moderate fatigue. Stay mindful of form."

                session_data['last_rep_time'] = current_time

                if session_data['count'] == 1:
                    session_data['start_time'] = current_time

            # Provide angle-based feedback
            if angle < session_data['exercise']["target_angle"] - session_data['exercise']["threshold"]:
                session_data['feedback_message'] = "Lower your knee slightly"
            elif angle > session_data['exercise']["target_angle"] + session_data['exercise']["threshold"]:
                session_data['feedback_message'] = "Raise your knee higher!"

            # Check notifications every 5 seconds
            if current_time - session_data['last_notification_check'] > 5:
                movement_error = analyze_movement_signature(session_data['landmarks_sequence'])
                notifications = check_notifications(fatigue_score, session_data['quality_scores'], movement_error)
                if notifications and 'email' in session:
                    user_ref = db.collection('users').document(session['email'])
                    user_ref.update({'notifications': firestore.ArrayUnion(notifications)})
                    session_data['notification_queue'].extend(notifications)
                session_data['last_notification_check'] = current_time

            # Draw feedback on frame
            cv2.putText(image, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Count: {session_data["count"]}/{session_data["target_count"]}', 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, session_data['feedback_message'], 
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f'Quality: {quality_score:.2f}', 
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Fatigue: {fatigue_score:.2f}', 
                        (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 0, 255) if fatigue_score > 0.6 else (0, 255, 0), 2)

            # Display notifications
            if session_data['notification_queue']:
                notif = session_data['notification_queue'].pop(0)
                if notif['severity'] == 'high':
                    cv2.rectangle(image, (40, 340), (600, 380), (0, 0, 255), -1)
                    cv2.putText(image, f"ALERT: {notif['message']}", 
                                (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(image, f"ALERT: {notif['message']}", 
                                (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Stop exercise if target count is reached
            if session_data['count'] >= session_data['target_count']:
                session_data['exercise_started'] = False
                total_time = time.time() - session_data['start_time'] if session_data['start_time'] else 0
                session_data['feedback_message'] = f"Exercise Complete! Total time: {total_time:.2f}s"
                cv2.putText(image, session_data['feedback_message'], 
                            (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Save session data
                if 'email' in session:
                    movement_error = analyze_movement_signature(session_data['landmarks_sequence'])
                    signature_analysis = {
                        'error_score': movement_error,
                        'consistency': 'good' if movement_error < 0.3 else 'fair' if movement_error < 0.6 else 'poor',
                        'trend': 'stable'
                    }
                    session_data_db = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "count": session_data['count'],
                        "total_time": total_time,
                        "average_speed": total_time / session_data['count'] if session_data['count'] > 0 else 0,
                        "average_quality": np.mean(session_data['quality_scores']) if session_data['quality_scores'] else 0,
                        "max_fatigue": max([f['fatigue'] for f in session_data['fatigue_window']]) if session_data['fatigue_window'] else 0,
                        "exercise_type": session_data['exercise']['name'],
                        "movement_signature_error": movement_error,
                        "movement_signature_analysis": signature_analysis,
                        "details": {
                            "angle_deviations": session_data['session_features']['angle_deviations'],
                            "rep_times": session_data['session_features']['rep_times'],
                            "speeds": session_data['session_features']['speeds'],
                            "quality_scores": session_data['quality_scores'],
                            "fatigue_scores": [f['fatigue'] for f in session_data['fatigue_window']]
                        }
                    }
                    try:
                        user_ref = db.collection('users').document(session['email'])
                        user_ref.update({
                            "sessions": firestore.ArrayUnion([session_data_db]),
                            "last_session": session_data_db,
                            "last_exercise": session_data['exercise']['name']
                        })
                    except Exception as e:
                        print(f"Error saving session to Firebase: {e}")

                # Reset session data
                session_data.update({
                    'count': 0,
                    'position': None,
                    'exercise_started': False,
                    'feedback_message': "Begin Exercise!",
                    'start_time': None,
                    'last_rep_time': None,
                    'fatigue_window': [],
                    'quality_scores': [],
                    'landmarks_sequence': [],
                    'notification_queue': [],
                    'last_notification_check': time.time()
                })

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Routes
@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        try:
            users_ref = db.collection('users').where('email', '==', email).limit(1)
            users = list(users_ref.stream())
            if users and bcrypt.checkpw(password, users[0].to_dict()['password'].encode('utf-8')):
                session['email'] = email
                session['username'] = users[0].to_dict()['username']
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error="Invalid email or password.")
        except Exception as e:
            print(f"Error during login: {e}")
            return render_template('login.html', error="An error occurred. Please try again.")
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
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        blood_group = request.form['blood_group']

        try:
            email_query = db.collection('users').where('email', '==', email).limit(1)
            if list(email_query.stream()):
                return render_template('register.html', error="Email already registered.")
            
            user_ref = db.collection('users').document(email)
            user_data = {
                "username": username,
                "email": email,
                "password": bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                "age": age,
                "height": height,
                "weight": weight,
                "blood_group": blood_group,
                "sessions": [],
                "notifications": []
            }
            user_ref.set(user_data)
            session['email'] = email
            session['username'] = username
            return redirect(url_for('dashboard'))
        except Exception as e:
            print(f"Error during registration: {e}")
            return render_template('register.html', error="An error occurred during registration.")
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        
        user_data = user.to_dict()
        sessions = user_data.get('sessions', [])[-5:]
        stats = {
            'total_sessions': len(user_data.get('sessions', [])),
            'total_reps': sum(session['count'] for session in user_data.get('sessions', [])),
            'avg_quality': np.mean([session.get('average_quality', 0) for session in user_data.get('sessions', [])]) if user_data.get('sessions') else 0,
            'last_session': user_data.get('last_session', {})
        }
        return render_template('dashboard.html', user=user_data, sessions=sessions, stats=stats)
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        return render_template('dashboard.html', error="An error occurred loading the dashboard.")

@app.route('/session/<session_id>')
def session_detail(session_id):
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        
        sessions = user.to_dict().get('sessions', [])
        session_data = sessions[int(session_id)]
        rep_numbers = list(range(1, len(session_data['details']['angle_deviations']) + 1))
        angle_deviations = session_data['details']['angle_deviations']
        speeds = session_data['details']['speeds']
        
        return render_template('session_detail.html', 
                             session_data=session_data,
                             rep_numbers=rep_numbers,
                             angle_deviations=angle_deviations,
                             speeds=speeds)
    except (IndexError, ValueError, KeyError) as e:
        print(f"Error loading session details: {e}")
        return redirect(url_for('dashboard'))

@app.route('/fatigue_analysis')
def fatigue_analysis():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        
        sessions = user.to_dict().get('sessions', [])
        fatigue_data = [
            {'session_num': i + 1, 'fatigue': session.get('max_fatigue', 0), 'date': session['date'].split()[0]}
            for i, session in enumerate(sessions[-10:])
        ]
        return render_template('fatigue_analysis.html', fatigue_data=fatigue_data)
    except Exception as e:
        print(f"Error loading fatigue analysis: {e}")
        return render_template('fatigue_analysis.html', error="An error occurred loading fatigue analysis.")

@app.route('/quality_analysis')
def quality_analysis():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        
        sessions = user.to_dict().get('sessions', [])
        quality_data = [
            {
                'session_num': i + 1,
                'quality': session.get('average_quality', 0),
                'date': session['date'].split()[0],
                'exercise': session.get('exercise_type', 'Unknown')
            }
            for i, session in enumerate(sessions[-10:])
        ]
        return render_template('quality_analysis.html', quality_data=quality_data)
    except Exception as e:
        print(f"Error loading quality analysis: {e}")
        return render_template('quality_analysis.html', error="An error occurred loading quality analysis.")

@app.route('/notifications')
def view_notifications():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        
        notifications = user.to_dict().get('notifications', [])
        user_ref.update({'notifications': []})
        return render_template('notifications.html', notifications=notifications)
    except Exception as e:
        print(f"Error loading notifications: {e}")
        return render_template('notifications.html', error="An error occurred loading notifications.")

@app.route('/movement_analysis')
def movement_analysis():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists or 'sessions' not in user.to_dict():
            return render_template('movement_analysis.html', error="No session data available")
        
        sessions = user.to_dict()['sessions']
        movement_errors = []
        dates = []
        for session in sessions[-10:]:
            if 'movement_signature_error' in session:
                movement_errors.append(session['movement_signature_error'])
                dates.append(session['date'].split()[0])
        
        plt.figure(figsize=(10, 5))
        plt.plot(dates, movement_errors, marker='o')
        plt.title('Movement Signature Consistency Over Time')
        plt.ylabel('Reconstruction Error (lower is better)')
        plt.xlabel('Session Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png')
        img_bytes.seek(0)
        plot_url = base64.b64encode(img_bytes.getvalue()).decode('utf8')
        plt.close()
        
        last_analysis = sessions[-1].get('movement_signature_analysis', {}) if sessions else {}
        return render_template('movement_analysis.html', 
                             plot_url=plot_url,
                             last_analysis=last_analysis,
                             movement_errors=movement_errors)
    except Exception as e:
        print(f"Error loading movement analysis: {e}")
        return render_template('movement_analysis.html', error="An error occurred loading movement analysis.")

@app.route('/post_session_analysis/<session_id>')
def post_session_analysis(session_id):
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        
        session_data = user.to_dict()['sessions'][int(session_id)]
        
        # Quality plot
        quality_plot = None
        if 'details' in session_data and 'quality_scores' in session_data['details']:
            reps = range(1, len(session_data['details']['quality_scores']) + 1)
            plt.figure(figsize=(10, 5))
            plt.plot(reps, session_data['details']['quality_scores'], marker='o')
            plt.title('Quality Score per Repetition')
            plt.xlabel('Repetition Number')
            plt.ylabel('Quality Score (0-1)')
            plt.ylim(0, 1)
            plt.grid(True)
            quality_img = BytesIO()
            plt.savefig(quality_img, format='png')
            quality_img.seek(0)
            quality_plot = base64.b64encode(quality_img.getvalue()).decode('utf8')
            plt.close()
        
        # Fatigue plot
        fatigue_plot = None
        if 'details' in session_data and 'fatigue_scores' in session_data['details']:
            times = np.linspace(0, session_data['total_time'], 
                              len(session_data['details']['fatigue_scores']))
            plt.figure(figsize=(10, 5))
            plt.plot(times, session_data['details']['fatigue_scores'])
            plt.title('Fatigue Trend During Session')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Fatigue Score (0-1)')
            plt.ylim(0, 1)
            plt.grid(True)
            fatigue_img = BytesIO()
            plt.savefig(fatigue_img, format='png')
            fatigue_img.seek(0)
            fatigue_plot = base64.b64encode(fatigue_img.getvalue()).decode('utf8')
            plt.close()
        
        # Cluster plot
        cluster_plot = None
        good_percentage = None
        if ('details' in session_data and 
            'angle_deviations' in session_data['details'] and
            'speeds' in session_data['details'] and
            len(session_data['details']['angle_deviations']) > 5):
            X = np.column_stack([
                session_data['details']['angle_deviations'],
                session_data['details']['speeds']
            ])
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(X)
            plt.figure(figsize=(8, 6))
            plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis')
            plt.title('Movement Quality Clusters')
            plt.xlabel('Angle Deviation')
            plt.ylabel('Movement Speed')
            plt.grid(True)
            cluster_img = BytesIO()
            plt.savefig(cluster_img, format='png')
            cluster_img.seek(0)
            cluster_plot = base64.b64encode(cluster_img.getvalue()).decode('utf8')
            plt.close()
            good_cluster = np.argmin(kmeans.cluster_centers_[:,0])
            good_percentage = np.mean(clusters == good_cluster) * 100
        
        return render_template('post_session_analysis.html',
                             session_data=session_data,
                             quality_plot=quality_plot,
                             fatigue_plot=fatigue_plot,
                             cluster_plot=cluster_plot,
                             good_percentage=good_percentage)
    except (IndexError, ValueError, KeyError) as e:
        print(f"Error loading post-session analysis: {e}")
        return redirect(url_for('dashboard'))

@app.route('/select_exercise', methods=['GET', 'POST'])
def select_exercise():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        exercise_name = request.form['exercise']
        try:
            user_ref = db.collection('users').document(session['email'])
            user_data = user_ref.get().to_dict()
            session_data = {
                'exercise': {
                    "name": exercise_name,
                    "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
                    "target_angle": 60 if exercise_name == "knee_raises" else 90,
                    "threshold": 15,
                    "optimal_speed_range": (1.0, 2.5) if exercise_name == "knee_raises" else (2.0, 4.0),
                    "difficulty": "beginner" if exercise_name == "knee_raises" else "intermediate"
                },
                'target_count': recommend_rep_count(user_data, "beginner" if exercise_name == "knee_raises" else "intermediate")
            }
            session['exercise_data'] = session_data
            return redirect(url_for('training'))
        except Exception as e:
            print(f"Error selecting exercise: {e}")
            return render_template('select_exercise.html', error="An error occurred selecting the exercise.")
    return render_template('select_exercise.html')

@app.route('/training')
def training():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('training.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    session_data = session.get('exercise_data', {})
    return jsonify({
        'count': session_data.get('count', 0),
        'target': session_data.get('target_count', 0),
        'feedback': session_data.get('feedback_message', "Begin Exercise!")
    })

@app.route('/set_target', methods=['POST'])
def set_target():
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})
    
    try:
        data = request.json
        target_count = int(data.get('target', 0))
        session_data = session.get('exercise_data', {})
        session_data.update({
            'target_count': target_count,
            'count': 0,
            'exercise_started': True,
            'feedback_message': "Begin Exercise!",
            'start_time': None,
            'last_rep_time': None
        })
        session['exercise_data'] = session_data
        return jsonify({'success': True, 'target': target_count})
    except Exception as e:
        print(f"Error setting target: {e}")
        return jsonify({'success': False, 'error': 'An error occurred setting the target.'})

@app.route('/profile')
def profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        
        user_data = user.to_dict()
        sessions = user_data.get('sessions', [])
        session_dates = [session['date'] for session in sessions]
        session_counts = [session['count'] for session in sessions]
        session_total_times = [session['total_time'] for session in sessions]
        session_average_speeds = [session['average_speed'] for session in sessions]
        
        return render_template('profile.html',
                             user=user_data,
                             session_dates=session_dates,
                             session_counts=session_counts,
                             session_total_times=session_total_times,
                             session_average_speeds=session_average_speeds)
    except Exception as e:
        print(f"Error loading profile: {e}")
        return render_template('profile.html', error="An error occurred loading the profile.")

@app.route('/admin')
def admin():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return redirect(url_for('login'))
    
    try:
        users = [user.to_dict() for user in db.collection('users').stream()]
        return render_template('admin.html', users=users)
    except Exception as e:
        print(f"Error loading admin page: {e}")
        return render_template('admin.html', error="An error occurred loading the admin page.")

@app.route('/recommendations')
def recommendations():
    if 'email' not in session:
        return redirect(url_for('login'))
    
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists or 'sessions' not in user.to_dict() or len(user.to_dict()['sessions']) == 0:
            return render_template('recommendation.html', error="No session data found.")
        
        sessions = user.to_dict()['sessions']
        avg_speed = np.mean([session['average_speed'] for session in sessions if session['average_speed'] > 0])
        recommendations = []
        if avg_speed < 1.5:
            recommendations.append("Your speed is too fast. Focus on controlled movements to improve form.")
        elif avg_speed > 3.0:
            recommendations.append("Your speed is too slow. Try to increase your pace for better endurance.")
        else:
            recommendations.append("Great job! Keep up the good work and aim for consistency.")
        
        return render_template('recommendation.html', recommendations=recommendations, user=user.to_dict(), avg_speed=avg_speed)
    except Exception as e:
        print(f"Error loading recommendations: {e}")
        return render_template('recommendation.html', error="An error occurred loading recommendations.")

@app.template_filter('mean')
def mean_filter(values):
    if not values:
        return 0
    return sum(values) / len(values)

app.jinja_env.filters['mean'] = mean_filter

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
