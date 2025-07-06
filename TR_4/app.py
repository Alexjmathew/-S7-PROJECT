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
from flask import session


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
feedback_message = "Begin PostureTraining!"
start_time = None
last_rep_time = None
exercise = None


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


# Generate video frames
def generate_frames():
    global count, position, exercise_started, feedback_message, start_time, last_rep_time, exercise
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and exercise_started and exercise:
            landmarks = results.pose_landmarks.landmark

            joints = exercise["joints"]
            coords = [
                [landmarks[getattr(mp_pose.PoseLandmark, joint).value].x,
                 landmarks[getattr(mp_pose.PoseLandmark, joint).value].y]
                for joint in joints
            ]

            angle = calculate_angle(*coords)

            # PostureTraining counting and feedback logic
            if angle > exercise["target_angle"] + exercise["threshold"]:
                position = "up"
            if position == "up" and angle < exercise["target_angle"] - exercise["threshold"]:
                position = "down"
                count += 1

                # Calculate time for the repetition
                current_time = time.time()
                if last_rep_time:
                    rep_time = current_time - last_rep_time
                    if exercise["optimal_speed_range"][0] <= rep_time <= exercise["optimal_speed_range"][1]:
                        feedback_message = "Good speed! Keep going."
                    elif rep_time < exercise["optimal_speed_range"][0]:
                        feedback_message = "Too fast! Slow down."
                    else:
                        feedback_message = "Too slow! Speed up."
                last_rep_time = current_time

                # Start timer for the first rep
                if count == 1:
                    start_time = current_time

            # Provide feedback based on angle
            if angle < exercise["target_angle"] - exercise["threshold"]:
                feedback_message = "Lower your knee slightly"
            elif angle > exercise["target_angle"] + exercise["threshold"]:
                feedback_message = "Raise your knee higher!"

            # Draw feedback on the frame
            cv2.putText(image, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Count: {count}/{target_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)
            cv2.putText(image, feedback_message, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Stop exercise if target count is reached
            if count >= target_count:
                exercise_started = False
                total_time = time.time() - start_time if start_time else 0
                feedback_message = f"PostureTraining Complete! Total time: {total_time:.2f}s"
                cv2.putText(image, feedback_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Save session data
                if 'email' in session:
                    session_data = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "count": count,
                        "total_time": total_time,
                        "average_speed": total_time / count if count > 0 else 0
                    }
                    user_ref = db.collection('users').document(session['email'])
                    user_ref.update({"sessions": firestore.ArrayUnion([session_data])})

                # Reset global variables for the next session
                start_time = None
                last_rep_time = None
                count = 0

        # Encode the frame and send to the frontend
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

        # Query users by email
        users_ref = db.collection('users').where('email', '==', email).limit(1)
        users = list(users_ref.stream())

        if users and users[0].to_dict()['password'] == password:
            session['email'] = email
            session['username'] = users[0].to_dict()['username']
            return redirect(url_for('profile'))
        else:
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

        # Check if email already exists
        email_query = db.collection('users').where('email', '==', email).limit(1)
        if list(email_query.stream()):
            return render_template('register.html', error="Email already registered.")

        # Create document with email as the document ID
        user_ref = db.collection('users').document(email)

        user_data = {
            "username": username,
            "email": email,
            "password": password,
            "age": age,
            "height": height,
            "weight": weight,
            "blood_group": blood_group,
            "sessions": []
        }
        user_ref.set(user_data)
        session['email'] = email
        session['username'] = username
        return redirect(url_for('profile'))
    return render_template('register.html')



# ... (existing imports: flask, cv2, mediapipe, firebase_admin, etc.)

# Global variable to store trained models for each user
user_models = {}  # Dictionary to store models by user email


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

    # Prepare data for charts and regression
    session_dates = [session['date'] for session in sessions]
    session_counts = [session['count'] for session in sessions]
    session_total_times = [session['total_time'] for session in sessions]
    session_average_speeds = [session['average_speed'] for session in sessions]

    # Simulate recovery time (replace with actual data if available)
    recovery_times = [session['total_time'] * (1 + session['average_speed']) for session in sessions]

    # Train RandomForestRegressor model and store it
    predicted_recovery_times = []
    r2 = 0
    if len(sessions) > 1:  # Need at least 2 data points for regression
        X = np.array([[s['count'], s['average_speed'], s['total_time']] for s in sessions])
        y = np.array(recovery_times)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        user_models[session['email']] = model  # Store model for the user
        predicted_recovery_times = model.predict(X).tolist()
        r2 = r2_score(y, predicted_recovery_times)  # Compute R² score
    else:
        predicted_recovery_times = recovery_times if recovery_times else [0] * len(sessions)

    return render_template('profile.html',
                           user=user_data,
                           session_dates=session_dates,
                           session_counts=session_counts,
                           session_total_times=session_total_times,
                           session_average_speeds=session_average_speeds,
                           recovery_times=recovery_times,
                           predicted_recovery_times=predicted_recovery_times,
                           r2=r2)


@app.route('/predict_recovery', methods=['POST'])
def predict_recovery():
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})

    data = request.json
    count = data.get('count')
    average_speed = data.get('average_speed')
    total_time = data.get('total_time')

    # Validate input
    if not all([isinstance(count, (int, float)), isinstance(average_speed, (int, float)),
                isinstance(total_time, (int, float))]):
        return jsonify({'success': False, 'error': 'Invalid input values'})

    if count < 0 or average_speed < 0 or total_time < 0:
        return jsonify({'success': False, 'error': 'Input values cannot be negative'})

    try:
        model = user_models.get(session['email'])
        if not model:
            return jsonify({'success': False, 'error': 'No trained model available. Complete at least two sessions.'})

        # Predict recovery time for the input
        X_new = np.array([[count, average_speed, total_time]])
        predicted_recovery = model.predict(X_new)[0]

        return jsonify({'success': True, 'predicted_recovery_time': round(predicted_recovery, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ... (rest of your existing Flask code)
# Update the existing admin route to pass the current date/time
@app.route('/admin')
def admin():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return redirect(url_for('login'))
    users = list(db.collection('users').stream())
    users = [user.to_dict() for user in users]
    for user in users:
        if 'email' not in user:
            user['email'] = ''  # Fallback in case email is missing
    return render_template('admin.html', users=users, now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# Add a new route to update session
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

    # Validate input
    if not all([user_email, session_index is not None, date, count is not None, total_time is not None]):
        return jsonify({'success': False, 'error': 'Missing required fields'})

    try:
        # Get the user document
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})

        user_data = user_doc.to_dict()

        # Make sure sessions exist and the index is valid
        if 'sessions' not in user_data or len(user_data['sessions']) <= session_index:
            return jsonify({'success': False, 'error': 'Session not found'})

        # Update the session
        user_data['sessions'][session_index] = {
            'date': date,
            'count': count,
            'total_time': total_time,
            'average_speed': total_time / count if count > 0 else 0
        }

        # Save changes
        user_ref.update({'sessions': user_data['sessions']})

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Add a new route to delete session
@app.route('/admin/delete_session', methods=['POST'])
def delete_session():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return jsonify({'success': False, 'error': 'Unauthorized access'})

    data = request.json
    user_email = data.get('email')
    session_index = data.get('session_index')

    # Validate input
    if user_email is None or session_index is None:
        return jsonify({'success': False, 'error': 'Missing required fields'})

    try:
        # Get the user document
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})

        user_data = user_doc.to_dict()

        # Make sure sessions exist and the index is valid
        if 'sessions' not in user_data or len(user_data['sessions']) <= session_index:
            return jsonify({'success': False, 'error': 'Session not found'})

        # Remove the session
        del user_data['sessions'][session_index]

        # Save changes
        user_ref.update({'sessions': user_data['sessions']})

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Add a new route to add session
@app.route('/admin/add_session', methods=['POST'])
def add_session():
    if 'email' not in session or session['username'] != 'ALEX J MATHEW':
        return jsonify({'success': False, 'error': 'Unauthorized access'})

    data = request.json
    user_email = data.get('email')
    date = data.get('date')
    count = data.get('count')
    total_time = data.get('total_time')

    # Validate input
    if not all([user_email, date, count is not None, total_time is not None]):
        return jsonify({'success': False, 'error': 'Missing required fields'})

    try:
        # Get the user document
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})

        # Calculate average speed
        average_speed = total_time / count if count > 0 else 0

        # Create the new session
        new_session = {
            'date': date,
            'count': count,
            'total_time': total_time,
            'average_speed': average_speed
        }

        # Add the session to the user document
        user_ref.update({
            'sessions': firestore.ArrayUnion([new_session])
        })

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/save_session', methods=['POST'])
def save_session():
    global count, start_time, last_rep_time
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})

    if count == 0:
        return jsonify({'success': False, 'error': 'No exercise data to save'})

    total_time = time.time() - start_time if start_time else 0
    session_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": count,
        "total_time": total_time,
        "average_speed": total_time / count if count > 0 else 0
    }

    user_ref = db.collection('users').document(session['email'])
    user_ref.update({"sessions": firestore.ArrayUnion([session_data])})

    return jsonify({'success': True})


@app.template_filter('mean')
def mean_filter(values):
    if not values:
        return 0
    return sum(values) / len(values)


# Register the filter
app.jinja_env.filters['mean'] = mean_filter


@app.route('/recommendations')
def recommendations():
    if 'email' not in session:
        return redirect(url_for('login'))

    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()

    if not user.exists or 'sessions' not in user.to_dict() or len(user.to_dict()['sessions']) == 0:
        return render_template('recommendation.html', error="No session data found.")

    # Calculate average speed
    sessions = user.to_dict()['sessions']
    avg_speed = np.mean([session['average_speed'] for session in sessions if session['average_speed'] > 0])

    # Rule-based recommendations
    recommendations = []
    if avg_speed < 1.5:
        recommendations.append("Your speed is too fast. Focus on controlled movements to improve form.")
    if avg_speed > 3.0:
        recommendations.append("Your speed is too slow. Try to increase your pace for better endurance.")
    if not recommendations:
        recommendations.append("Great job! Keep up the good work and aim for consistency.")

    return render_template('recommendation.html', recommendations=recommendations, user=user.to_dict(),
                           avg_speed=avg_speed)


@app.route('/select_exercise', methods=['GET', 'POST'])
def select_exercise():
    if 'email' not in session:
        return redirect(url_for('login'))
    global exercise
    if request.method == 'POST':
        exercise_name = request.form['exercise']
        if exercise_name == "knee_raises":
            exercise = {
                "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
                "target_angle": 60,  # Ideal angle for knee raised to hip level
                "threshold": 15,
                "optimal_speed_range": (1.0, 2.5)  # Optimal time in seconds for one rep
            }
        elif exercise_name == "squats":
            exercise = {
                "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
                "target_angle": 90,  # Ideal angle for squat
                "threshold": 15,
                "optimal_speed_range": (2.0, 4.0)  # Optimal time in seconds for one rep
            }
        return redirect(url_for('training'))
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
    global count, target_count, feedback_message
    return jsonify({'count': count, 'target': target_count, 'feedback': feedback_message})


@app.route('/set_target', methods=['POST'])
def set_target():
    global target_count, count, exercise_started, feedback_message, start_time, last_rep_time
    data = request.json
    target_count = int(data.get('target', 0))
    count = 0
    exercise_started = True
    feedback_message = "Begin PostureTraining!"
    start_time = None
    last_rep_time = None
    return jsonify({'success': True, 'target': target_count})


if __name__ == "__main__":
    app.run(debug=True)