# 💪 FitnessVision - AI-Powered Workout Tracker

![Demo](demo.gif)

**Real-time exercise tracking with form feedback using computer vision**

## ✨ Key Features

- 🎯 **10+ Exercises**: Squats, pushups, lunges + manual logging
- 👁️ **Real-Time AI Coach**: MediaPipe pose detection with form feedback
- 🔥 **Calorie Calculator**: MET-based & rep-based estimates
- 📊 **Progress Tracking**: Workout history & performance analytics
- 🏆 **Gamification**: Badges & leaderboards
- 🔒 **Secure Auth**: Firebase authentication
- 📱 **Admin Dashboard**: Manage user data

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/yourusername/FitnessVision.git
cd FitnessVision

# 2. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Firebase
# - Add firebase.json to project root
# - Enable Email/Password auth

# 5. Run!
python app.py
