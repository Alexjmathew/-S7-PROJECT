from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, send_file
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pandas as pd
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Firebase
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Mediapipe and utility setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Global variables
count = 0
target_count = 0
position = None
exercise_started = False
is_paused = False
feedback_message = "Begin PostureTraining!"
start_time = None
last_rep_time = None
exercise = None
paused_time = 0
pause_start = None

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

# Calorie burn estimation with more exercises
def estimate_calories_burned(exercise_name, duration, weight, count=0):
    # MET (Metabolic Equivalent of Task) values for exercises
    # MET values source: https://sites.google.com/site/compendiumofphysicalactivities/
    met_values = {
        "knee_raises": 3.5,  # Moderate calisthenics
        "squats": 5.0,       # Vigorous calisthenics
        "pushups": 3.8,      # Moderate effort
        "lunges": 4.5,       # Moderate effort
        "plank": 3.0,        # Moderate effort
        "sit-ups": 3.5,      # Moderate effort
        "jumping_jacks": 8.0,# Vigorous effort
        "bicep_curls": 3.0,  # Light effort
        "shoulder_press": 4.0 # Moderate effort
    }
    
    # Some exercises are better calculated by reps
    calories_per_rep = {
        "pushups": 0.3,
        "sit-ups": 0.2,
        "bicep_curls": 0.15,
        "shoulder_press": 0.25
    }
    
    if exercise_name in calories_per_rep and count > 0:
        return round(calories_per_rep[exercise_name] * count, 2)
    
    met = met_values.get(exercise_name, 4.0)  # Default MET for unknown exercises
    # Calorie burn formula: Calories = MET * weight (kg) * duration (hours)
    calories = met * (float(weight) if isinstance(weight, str) else weight) * (duration / 3600)
    return round(calories, 2)

# Generate video frames
def generate_frames():
    global count, position, exercise_started, is_paused, feedback_message, start_time, last_rep_time, exercise, paused_time, pause_start
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks and exercise_started and exercise and not is_paused:
            landmarks = results.pose_landmarks.landmark
            try:
                joints = exercise["joints"]
                coords = [
                    [landmarks[getattr(mp_pose.PoseLandmark, joint).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, joint).value].y]
                    for joint in joints
                ]
                angle = calculate_angle(*coords)
                
                # Exercise-specific logic
                if exercise["name"] == "knee_raises":
                    if angle > exercise["target_angle"] + exercise["threshold"]:
                        position = "up"
                    if position == "up" and angle < exercise["target_angle"] - exercise["threshold"]:
                        position = "down"
                        count += 1
                        update_rep_timing()
                    if angle < exercise["target_angle"] - exercise["threshold"]:
                        feedback_message = "Lower your knee slightly"
                    elif angle > exercise["target_angle"] + exercise["threshold"]:
                        feedback_message = "Raise your knee higher!"
                        
                elif exercise["name"] == "squats":
                    if angle < exercise["target_angle"] - exercise["threshold"]:
                        position = "down"
                    if position == "down" and angle > exercise["target_angle"] + exercise["threshold"]:
                        position = "up"
                        count += 1
                        update_rep_timing()
                    if angle > exercise["target_angle"] + exercise["threshold"]:
                        feedback_message = "Stand up straighter!"
                    elif angle < exercise["target_angle"] - exercise["threshold"]:
                        feedback_message = "Bend your knees more!"
                
                elif exercise["name"] == "pushups":
                    if angle < exercise["target_angle"] - exercise["threshold"]:
                        position = "down"
                    if position == "down" and angle > exercise["target_angle"] + exercise["threshold"]:
                        position = "up"
                        count += 1
                        update_rep_timing()
                    if angle > exercise["target_angle"] + exercise["threshold"]:
                        feedback_message = "Lower your body more!"
                    elif angle < exercise["target_angle"] - exercise["threshold"]:
                        feedback_message = "Push up higher!"
                
                # Display exercise info
                cv2.putText(image, f'Exercise: {exercise["name"]}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Angle: {int(angle)}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Count: {count}/{target_count}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(image, feedback_message, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Check if target count reached
                if count >= target_count and target_count > 0:
                    complete_exercise()
                    
            except Exception as e:
                print(f"Error processing pose: {e}")
                feedback_message = "Adjust your position to be visible"
                
        # Draw landmarks and display pause status
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        if is_paused:
            cv2.putText(image, "PAUSED", (image.shape[1]//2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def update_rep_timing():
    global last_rep_time, feedback_message, start_time
    current_time = time.time()
    if last_rep_time:
        rep_time = current_time - last_rep_time
        if exercise["optimal_speed_range"][0] <= rep_time <= exercise["optimal_speed_range"][1]:
            feedback_message = "Good speed! Keep going."
        elif rep_time < exercise["optimal_speed_range"][0]:
            feedback_message = "Too fast! Slow down."
        else:
魔力
            feedback_message = "Too slow! Speed up."
    last_rep_time = current_time
    if count == 1 and not start_time:
        start_time = current_time

def complete_exercise():
    global exercise_started, feedback_message, start_time, last_rep_time, count, paused_time
    exercise_started = False
    total_time = time.time() - start_time - paused_time if start_time else 0
    
    if 'email' in session:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get().to_dict()
        calories = estimate_calories_burned(exercise["name"], total_time, float(user.get('weight', 70)), count)
        
        session_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": count,
            "total_time": total_time,
            "average_speed": total_time / count if count > 0 else 0,
            "calories": calories,
            "exercise": exercise["name"]
        }
        user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
        update_badges(user_ref, count, total_time)
        
        feedback_message = (f"Exercise Complete! "
                          f"Total time: {total_time:.2f}s, "
                          f"Calories: {calories}, "
                          f"Avg speed: {total_time/count:.2f}s/rep")
    
    # Reset exercise variables
    start_time = None
    last_rep_time = None
    count = 0
    paused_time = 0

# Update badges for gamification
def update_badges(user_ref, count, total_time):
    user = user_ref.get().to_dict()
    badges = user.get('badges', [])
    sessions = user.get('sessions', [])
    total_sessions = len(sessions) + 1  # Including the new session
    total_reps = sum([s['count'] for s in sessions]) + count
    total_calories = sum([s.get('calories', 0) for s in sessions]) + estimate_calories_burned(exercise["name"], total_time, float(user.get('weight', 70)), count)
    
    # Achievement badges
    if total_sessions >= 5 and "Consistent" not in badges:
        badges.append("Consistent")
    if total_reps >= 100 and "Century" not in badges:
        badges.append("Century")
    if total_time < 30 and count >= 10 and "Speedster" not in badges:
        badges.append("Speedster")
    if total_calories >= 500 and "Calorie Burner" not in badges:
        badges.append("Calorie Burner")
    if any(s['count'] >= 50 for s in sessions) and "Marathoner" not in badges:
        badges.append("Marathoner")
    
    user_ref.update({"badges": badges})

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
        try:
            user = auth.get_user_by_email(email)
            user_ref = db.collection('users').document(email)
            user_data = user_ref.get().to_dict()
            if user_data and user_data['password'] == password:
                session['email'] = email
                session['username'] = user_data['username']
                return redirect(url_for('profile'))
            else:
                return render_template('login.html', error="Invalid email or password.")
        except:
            return render_template('login.html', error="Invalid email or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        try:
            auth.generate_password_reset_link(email)
            return render_template('forgot_password.html', message="Password reset link sent to your email.")
        except:
            return render_template('forgot_password.html', error="Email not found.")
    return render_template('forgot_password.html')

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
        try:
            auth.create_user(email=email, password=password)
            user_ref = db.collection('users').document(email)
            user_data = {
                "username": username,
                "email": email,
                "password": password,  # Store for compatibility
                "age": age,
                "height": height,
                "weight": weight,
                "blood_group": blood_group,
                "sessions": [],
                "badges": []
            }
            user_ref.set(user_data)
            session['email'] = email
            session['username'] = username
            return redirect(url_for('profile'))
        except:
            return render_template('register.html', error="Registration failed. Please try again.")
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
    session_dates = [session['date'] for session in sessions]
    session_counts = [session['count'] for session in sessions]
    session_total_times = [session['total_time'] for session in sessions]
    session_average_speeds = [session['average_speed'] for session in sessions]
    session_calories = [session.get('calories', 0) for session in sessions]
    session_exercises = [session.get('exercise', 'Unknown') for session in sessions]
    return render_template('profile.html',
                           user=user_data,
                           sessions=sessions,
                           session_dates=session_dates,
                           session_counts=session_counts,
                           session_total_times=session_total_times,
                           session_average_speeds=session_average_speeds,
                           session_calories=session_calories,
                           session_exercises=session_exercises)

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get().to_dict()
    if request.method == 'POST':
        username = request.form['username']
        age = request.form['age']
        height = request.form['height']
        weight = request.form['weight']
        blood_group = request.form['blood_group']
        user_ref.update({
            "username": username,
            "age": age,
            "height": height,
            "weight": weight,
            "blood_group": blood_group
        })
        session['username'] = username
        return redirect(url_for('profile'))
    return render_template('edit_profile.html', user=user)

@app.route('/manual_log', methods=['GET', 'POST'])
def manual_log():
    if 'email' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        date = request.form['date']
        count = int(request.form['count'])
        total_time = float(request.form['total_time'])
        exercise_name = request.form['exercise']
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get().to_dict()
        calories = estimate_calories_burned(exercise_name, total_time, float(user.get('weight', 70)), count)
        session_data = {
            "date": date,
            "count": count,
            "total_time": total_time,
            "average_speed": total_time / count if count > 0 else 0,
            "calories": calories,
            "exercise": exercise_name
        }
        user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
        update_badges(user_ref, count, total_time)
        return redirect(url_for('profile'))
    return render_template('manual_log.html')

@app.route('/export_sessions')
def export_sessions():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get().to_dict()
    sessions = user.get('sessions', [])
    df = pd.DataFrame(sessions)
    if not df.empty:
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='session_history.csv'
        )
    return redirect(url_for('profile'), error="No sessions to export.")

@app.route('/admin')
def admin():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return redirect(url_for('login'))
    users = [user.to_dict() for user in db.collection('users').stream()]
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
    exercise = data.get('exercise')
    if not all([user_email, session_index is not None, date, count is not None, total_time is not None, exercise]):
        return jsonify({'success': False, 'error': 'Missing required fields'})
    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})
        user_data = user_doc.to_dict()
        if 'sessions' not in user_data or len(user_data['sessions']) <= session_index:
            return jsonify({'success': False, 'error': 'Session not found'})
        calories = estimate_calories_burned(exercise, total_time, float(user_data.get('weight', 70)), count)
        user_data['sessions'][session_index] = {
            'date': date,
            'count': count,
            'total_time': total_time,
            'average_speed': total_time / count if count > 0 else 0,
            'calories': calories,
            'exercise': exercise
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
    exercise = data.get('exercise')
    if not all([user_email, date, count is not None, total_time is not None, exercise]):
        return jsonify({'success': False, 'error': 'Missing required fields'})
    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})
        user_data = user_doc.to_dict()
        calories = estimate_calories_burned(exercise, total_time, float(user_data.get('weight', 70)), count)
        new_session = {
            'date': date,
            'count': count,
            'total_time': total_time,
            'average_speed': total_time / count if count > 0 else 0,
            'calories': calories,
            'exercise': exercise
        }
        user_ref.update({'sessions': firestore.ArrayUnion([new_session])})
        update_badges(user_ref, count, total_time)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_session', methods=['POST'])
def save_session():
    global count, start_time, last_rep_time, paused_time
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})
    if count == 0:
        return jsonify({'success': False, 'error': 'No exercise data to save'})
    total_time = time.time() - start_time - paused_time if start_time else 0
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get().to_dict()
    calories = estimate_calories_burned(exercise["name"], total_time, float(user.get('weight', 70)), count)
    session_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": count,
        "total_time": total_time,
        "average_speed": total_time / count if count > 0 else 0,
        "calories": calories,
        "exercise": exercise["name"]
    }
    user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
    update_badges(user_ref, count, total_time)
    return jsonify({'success': True})

@app.route('/leaderboard')
def leaderboard():
    users = [user.to_dict() for user in db.collection('users').stream()]
    leaderboard_data = []
    for user in users:
        sessions = user.get('sessions', [])
        total_reps = sum([s['count'] for s in sessions])
        total_time = sum([s['total_time'] for s in sessions])
        leaderboard_data.append({
            'username': user.get('username', 'Unknown'),
            'total_reps': total_reps,
            'total_time': round(total_time, 2),
            'total_sessions': len(sessions),
            'badges': len(user.get('badges', []))
        })
    leaderboard_data.sort(key=lambda x: x['total_reps'], reverse=True)
    return render_template('leaderboard.html', leaderboard=leaderboard_data)

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
    global exercise
    
    exercises = {
        "knee_raises": {
            "name": "Knee Raises",
            "description": "Lift your knees to strengthen core and legs",
            "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
            "target_angle": 60,
            "threshold": 15,
            "optimal_speed_range": (1.0, 2.5),
            "image": "knee_raises.png"
        },
        "squats": {
            "name": "Squats",
            "description": "Lower your body to strengthen legs and glutes",
            "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
            "target_angle": 90,
            "threshold": 15,
            "optimal_speed_range": (2.0, 4.0),
            "image": "squats.png"
        },
        "pushups": {
            "name": "Pushups",
            "description": "Upper body strength exercise",
            "joints": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
            "target_angle": 90,
            "threshold": 15,
            "optimal_speed_range": (1.5, 3.0),
            "image": "pushups.png"
        }
    }
    
    if request.method == 'POST':
        exercise_name = request.form['exercise']
        if exercise_name in exercises:
            exercise = exercises[exercise_name]
            session['exercise_name'] = exercise["name"]
            session['exercise_image'] = exercise["image"]
            return redirect(url_for('training'))
        else:
            return render_template('select_exercise.html', error="Invalid exercise selected", exercises=exercises.values())
    
    return render_template('select_exercise.html', exercises=exercises.values())

@app.route('/training')
def training():
    if 'email' not in session:
        return redirect(url_for('login'))
    if not exercise:
        return redirect(url_for('select_exercise'))
    return render_template('training.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    global count, target_count, feedback_message, start_time, paused_time, is_paused
    total_time = time.time() - start_time - paused_time if start_time and count >= target_count else 0
    return jsonify({
        'count': count,
        'target': target_count,
        'feedback': feedback_message,
        'total_time': total_time,
        'exercise_complete': count >= target_count,
        'is_paused': is_paused
    })

@app.route('/set_target', methods=['POST'])
def set_target():
    global target_count, count, exercise_started, feedback_message, start_time, last_rep_time, paused_time, is_paused
    data = request.json
    target_count = int(data.get('target', 0))
    count = 0
    exercise_started = True
    is_paused = False
    feedback_message = "Begin PostureTraining!"
    start_time = None
    last_rep_time = None
    paused_time = 0
    return jsonify({'success': True, 'target': target_count})

@app.route('/pause_resume', methods=['POST'])
def pause_resume():
    global is_paused, pause_start, paused_time
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})
    if not exercise_started:
        return jsonify({'success': False, 'error': 'No active exercise'})
    is_paused = not is_paused
    if is_paused:
        pause_start = time.time()
    else:
        if pause_start:
            paused_time += time.time() - pause_start
        pause_start = None
    return jsonify({'success': True, 'is_paused': is_paused})

if __name__ == "__main__":
    app.run(debug=True)
