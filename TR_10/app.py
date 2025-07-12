from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import uuid
import pickle
import base64
from ucimlrepo import fetch_ucirepo

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize Firebase Admin
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Pyrebase for Firebase Authentication
firebase_config = {
    "apiKey": "your_api_key",
    "authDomain": "your_project_id.firebaseapp.com",
    "databaseURL": "https://your_project_id.firebaseio.com",
    "projectId": "your_project_id",
    "storageBucket": "your_project_id.appspot.com",
    "messagingSenderId": "your_messaging_sender_id",
    "appId": "your_app_id"
}
firebase = pyrebase.initialize_app(firebase_config)
pyrebase_auth = firebase.auth()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Global variables for exercise tracking
count = 0
target_count = 0
position = None
exercise_started = False
feedback_message = "Begin Exercise!"
start_time = None
last_rep_time = None
exercise = None
user_models = {}  # Store trained models by user email
nutrition_models = {}  # Store nutrition ML models by user email

# Exercise configurations
EXERCISES = {
    "knee_raises": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 60,
        "threshold": 15,
        "optimal_speed_range": (1.0, 2.5),
        "calories_per_rep": 0.5
    },
    "squats": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 90,
        "threshold": 15,
        "optimal_speed_range": (2.0, 4.0),
        "calories_per_rep": 0.8
    }
}

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

# Generate video frames for exercise tracking
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
            if angle > exercise["target_angle"] + exercise["threshold"]:
                position = "up"
            if position == "up" and angle < exercise["target_angle"] - exercise["threshold"]:
                position = "down"
                count += 1
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
                if count == 1:
                    start_time = current_time
            if angle < exercise["target_angle"] - exercise["threshold"]:
                feedback_message = "Lower your knee slightly"
            elif angle > exercise["target_angle"] + exercise["threshold"]:
                feedback_message = "Raise your knee higher!"
            cv2.putText(image, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Count: {count}/{target_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, feedback_message, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if count >= target_count:
                exercise_started = False
                total_time = time.time() - start_time if start_time else 0
                calories_burned = count * exercise["calories_per_rep"]
                feedback_message = f"Exercise Complete! Total time: {total_time:.2f}s, Calories: {calories_burned:.2f}"
                cv2.putText(image, feedback_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if 'email' in session:
                    session_data = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "exercise": exercise["name"],
                        "count": count,
                        "total_time": total_time,
                        "average_speed": total_time / count if count > 0 else 0,
                        "calories_burned": calories_burned
                    }
                    user_ref = db.collection('users').document(session['email'])
                    user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
                start_time = None
                last_rep_time = None
                count = 0
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Nutrition AI Model Training
def train_nutrition_model(user_data, sessions):
    dataset = fetch_ucirepo(id=544)
    X = dataset.data.features
    y = dataset.data.targets['NObeyesdad']
    
    scaler = StandardScaler()
    le = LabelEncoder()
    
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC']
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])
    
    X['BMI'] = X['Weight'] / (X['Height'] ** 2)
    X['Healthy_Diet'] = (X['FCVC'] >= 2) & (X['FAVC'] == 0)
    X['Healthy_Diet'] = X['Healthy_Diet'].astype(int)
    
    numerical_cols = ['Age', 'Height', 'Weight', 'BMI', 'FCVC', 'NCP', 'FAF', 'TUE']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    diet_mapping = {
        'Insufficient_Weight': 0,
        'Normal_Weight': 0,
        'Overweight_Level_I': 2,
        'Overweight_Level_II': 2,
        'Obesity_Type_I': 2,
        'Obesity_Type_II': 1,
        'Obesity_Type_III': 1
    }
    y = y.map(diet_mapping)
    
    total_calories_burned = sum(session.get('calories_burned', 0) for session in sessions)
    avg_speed = np.mean([session['average_speed'] for session in sessions if session['average_speed'] > 0]) if sessions else 0
    bmi = user_data['weight'] / (user_data['height'] / 100) ** 2
    gender = 1 if user_data.get('gender', 'M') == 'M' else 0
    healthy_diet = 1 if user_data.get('vegetable_intake', 2) >= 2 and user_data.get('high_calorie_food', 'no') == 'no' else 0
    
    user_features = np.array([[
        user_data['age'],
        user_data['height'] / 100,
        user_data['weight'],
        bmi,
        gender,
        healthy_diet,
        total_calories_burned / 1000,
        avg_speed
    ]])
    
    user_features[:, [0, 1, 2, 3, 6, 7]] = scaler.transform(user_features[:, [0, 1, 2, 3, 6, 7]])
    
    X = np.vstack([X[['Age', 'Height', 'Weight', 'BMI', 'Gender', 'Healthy_Diet', 'FAF', 'TUE']].values, user_features])
    y = np.append(y, 0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    model_bytes = pickle.dumps(model)
    model_base64 = base64.b64encode(model_bytes).decode('utf-8')
    user_ref = db.collection('users').document(user_data['email'])
    user_ref.update({'nutrition_model': model_base64})
    
    return model

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
        password = request.form['password']
        try:
            user = pyrebase_auth.sign_in_with_email_and_password(email, password)
            session['email'] = email
            user_ref = db.collection('users').document(email)
            user_data = user_ref.get().to_dict()
            session['username'] = user_data['username']
            return redirect(url_for('dashboard'))
        except Exception as e:
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
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        blood_group = request.form['blood_group']
        gender = request.form.get('gender', 'M')
        vegetable_intake = float(request.form.get('vegetable_intake', 2))
        high_calorie_food = request.form.get('high_calorie_food', 'no')
        try:
            # Create user with Firebase Authentication
            user = pyrebase_auth.create_user_with_email_and_password(email, password)
            # Store additional user data in Firestore
            user_ref = db.collection('users').document(email)
            user_data = {
                "username": username,
                "email": email,
                "age": age,
                "height": height,
                "weight": weight,
                "blood_group": blood_group,
                "gender": gender,
                "vegetable_intake": vegetable_intake,
                "high_calorie_food": high_calorie_food,
                "sessions": [],
                "weight_history": [{"date": datetime.now().strftime("%Y-%m-%d"), "weight": weight}],
                "daily_calories": [],
                "nutrition_model": None
            }
            user_ref.set(user_data)
            session['email'] = email
            session['username'] = username
            return redirect(url_for('dashboard'))
        except Exception as e:
            return render_template('register.html', error="Email already registered or invalid input.")
    return render_template('register.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        try:
            pyrebase_auth.send_password_reset_email(email)
            return render_template('forgot_password.html', success="Password reset email sent. Check your inbox.")
        except Exception as e:
            return render_template('forgot_password.html', error="Error sending reset email. Ensure the email is registered.")
    return render_template('forgot_password.html')

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    if not user.exists:
        return redirect(url_for('login'))
    user_data = user.to_dict()
    sessions = user_data.get('sessions', [])
    total_calories_burned = sum(session.get('calories_burned', 0) for session in sessions)
    recent_sessions = sessions[-5:] if len(sessions) > 5 else sessions
    return render_template('dashboard.html', user=user_data, sessions=recent_sessions, total_calories=total_calories_burned)

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
    session_calories = [session.get('calories_burned', 0) for session in sessions]
    recovery_times = [session['total_time'] * (1 + session['average_speed']) for session in sessions]
    predicted_recovery_times = []
    r2 = 0
    if len(sessions) > 1:
        X = np.array([[s['count'], s['average_speed'], s['total_time']] for s in sessions])
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
                          session_total_times=session_total_times,
                          session_average_speeds=session_average_speeds,
                          session_calories=session_calories,
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

@app.route('/leaderboard')
def leaderboard():
    users = list(db.collection('users').stream())
    leaderboard_data = []
    for user in users:
        user_data = user.to_dict()
        sessions = user_data.get('sessions', [])
        total_calories = sum(session.get('calories_burned', 0) for session in sessions)
        total_reps = sum(session.get('count', 0) for session in sessions)
        leaderboard_data.append({
            'username': user_data['username'],
            'total_calories': total_calories,
            'total_reps': total_reps,
            'session_count': len(sessions)
        })
    leaderboard_data = sorted(leaderboard_data, key=lambda x: x['total_calories'], reverse=True)
    return render_template('leaderboard.html', leaderboard=leaderboard_data)

@app.route('/nutrition')
def nutrition():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    if not user.exists:
        return redirect(url_for('login'))
    user_data = user.to_dict()
    sessions = user_data.get('sessions', [])
    total_calories_burned = sum(session.get('calories_burned', 0) for session in sessions)
    avg_speed = np.mean([session['average_speed'] for session in sessions if session['average_speed'] > 0]) if sessions else 0
    
    model = nutrition_models.get(session['email'])
    if not model and user_data.get('nutrition_model'):
        try:
            model_bytes = base64.b64decode(user_data['nutrition_model'])
            model = pickle.loads(model_bytes)
            nutrition_models[session['email']] = model
        except:
            model = None
    if not model:
        model = train_nutrition_model(user_data, sessions)
        nutrition_models[session['email']] = model
    
    scaler = StandardScaler()
    bmi = user_data['weight'] / (user_data['height'] / 100) ** 2
    gender = 1 if user_data.get('gender', 'M') == 'M' else 0
    healthy_diet = 1 if user_data.get('vegetable_intake', 2) >= 2 and user_data.get('high_calorie_food', 'no') == 'no' else 0
    features = np.array([[
        user_data['age'],
        user_data['height'] / 100,
        user_data['weight'],
        bmi,
        gender,
        healthy_diet,
        total_calories_burned / 1000,
        avg_speed
    ]])
    features[:, [0, 1, 2, 3, 6, 7]] = scaler.fit_transform(features)
    
    diet_categories = {
        0: "Balanced Diet: Maintain a mix of proteins, carbs, and fats.",
        1: "High Protein Diet: Increase protein intake (e.g., lean meats, eggs, legumes) for muscle recovery.",
        2: "Low Carb Diet: Reduce refined sugars and carbs to support weight management.",
        3: "High Carb Diet: Increase complex carbs (e.g., whole grains, fruits) for energy."
    }
    diet_category = model.predict(features)[0]
    recommendation = diet_categories.get(diet_category, "Balanced Diet: Maintain a mix of proteins, carbs, and fats.")
    
    return render_template('nutrition.html', user=user_data, recommendations=[recommendation], total_calories=total_calories_burned)

@app.route('/calorie_tracker', methods=['GET', 'POST'])
def calorie_tracker():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    if not user.exists:
        return redirect(url_for('login'))
    user_data = user.to_dict()
    if request.method == 'POST':
        weight = float(request.form.get('weight'))
        calories_consumed = float(request.form.get('calories_consumed', 0))
        vegetable_intake = float(request.form.get('vegetable_intake', user_data.get('vegetable_intake', 2)))
        high_calorie_food = request.form.get('high_calorie_food', user_data.get('high_calorie_food', 'no'))
        user_data['weight_history'].append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'weight': weight
        })
        user_data['daily_calories'].append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'calories_consumed': calories_consumed
        })
        user_ref.update({
            'weight_history': user_data['weight_history'],
            'daily_calories': user_data['daily_calories'],
            'weight': weight,
            'vegetable_intake': vegetable_intake,
            'high_calorie_food': high_calorie_food
        })
        sessions = user_data.get('sessions', [])
        train_nutrition_model(user_data, sessions)
        return redirect(url_for('calorie_tracker'))
    sessions = user_data.get('sessions', [])
    total_calories_burned = sum(session.get('calories_burned', 0) for session in sessions)
    calorie_balance = sum(entry.get('calories_consumed', 0) for entry in user_data.get('daily_calories', [])) - total_calories_burned
    return render_template('calorie_tracker.html', user=user_data, total_calories_burned=total_calories_burned, calorie_balance=calorie_balance)

@app.route('/diet_plan')
def diet_plan():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_ref = db.collection('users').document(session['email'])
    user = user_ref.get()
    if not user.exists:
        return redirect(url_for('login'))
    user_data = user.to_dict()
    age = user_data['age']
    weight = user_data['weight']
    height = user_data['height']
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if user_data.get('gender', 'M') == 'M' else -161)
    activity_factor = 1.5
    daily_calories = bmr * activity_factor
    diet_plan = {
        'breakfast': 'Oatmeal with berries and nuts (400 kcal)',
        'lunch': 'Grilled chicken salad with quinoa (600 kcal)',
        'dinner': 'Baked salmon with steamed vegetables (500 kcal)',
        'snacks': 'Greek yogurt and fruit (200 kcal)'
    }
    return render_template('diet_plan.html', user=user_data, daily_calories=daily_calories, diet_plan=diet_plan)

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
    exercise_name = data.get('exercise')
    if not all([user_email, session_index is not None, date, count is not None, total_time is not None, exercise_name]):
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
            'exercise': exercise_name,
            'count': count,
            'total_time': total_time,
            'average_speed': total_time / count if count > 0 else 0,
            'calories_burned': count * EXERCISES[exercise_name]["calories_per_rep"]
        }
        user_ref.update({'sessions': user_data['sessions']})
        train_nutrition_model(user_data, user_data['sessions'])
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
        train_nutrition_model(user_data, user_data['sessions'])
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
    exercise_name = data.get('exercise')
    if not all([user_email, date, count is not None, total_time is not None, exercise_name]):
        return jsonify({'success': False, 'error': 'Missing required fields'})
    try:
        user_ref = db.collection('users').document(user_email)
        user_doc = user_ref.get()
        if not user_doc.exists:
            return jsonify({'success': False, 'error': 'User not found'})
        average_speed = total_time / count if count > 0 else 0
        new_session = {
            'date': date,
            'exercise': exercise_name,
            'count': count,
            'total_time': total_time,
            'average_speed': average_speed,
            'calories_burned': count * EXERCISES[exercise_name]["calories_per_rep"]
        }
        user_ref.update({'sessions': firestore.ArrayUnion([new_session])})
        user_data = user_doc.to_dict()
        train_nutrition_model(user_data, user_data['sessions'] + [new_session])
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_session', methods=['POST'])
def save_session():
    global count, start_time, last_rep_time, exercise
    if 'email' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'})
    if count == 0:
        return jsonify({'success': False, 'error': 'No exercise data to save'})
    total_time = time.time() - start_time if start_time else 0
    session_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exercise": exercise["name"],
        "count": count,
        "total_time": total_time,
        "average_speed": total_time / count if count > 0 else 0,
        "calories_burned": count * exercise["calories_per_rep"]
    }
    user_ref = db.collection('users').document(session['email'])
    user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
    user_data = user_ref.get().to_dict()
    train_nutrition_model(user_data, user_data['sessions'] + [session_data])
    return jsonify({'success': True})

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
    if request.method == 'POST':
        exercise_name = request.form['exercise']
        if exercise_name in EXERCISES:
            exercise = EXERCISES[exercise_name]
            exercise["name"] = exercise_name
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
    feedback_message = "Begin Exercise!"
    start_time = None
    last_rep_time = None
    return jsonify({'success': True, 'target': target_count})

@app.template_filter('mean')
def mean_filter(values):
    if not values:
        return 0
    return sum(values) / len(values)

if __name__ == "__main__":
    app.run(debug=True)
