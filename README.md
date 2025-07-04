# AI-Powered Physiotherapy Assistance System for Rehabilitation

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Storage](#data-storage)
7. [Contact](#contact)

## Introduction <a name="introduction"></a>
The AI-Powered Physiotherapy Assistance System is an innovative solution designed to enhance rehabilitation through artificial intelligence and computer vision. This system provides real-time feedback on exercise form, tracks progress, and personalizes rehabilitation programs to optimize recovery outcomes while reducing injury risks.

## Key Features <a name="key-features"></a>

### 1. AI-Based Real-Time Movement Tracking
- Utilizes MediaPipe Pose Detection for precise joint tracking
- Monitors key body joints (knees, shoulders, spine)
- Measures angles between joints to assess posture accuracy

### 2. Automated Repetition Counting & Exercise Validation
- AI-powered rep counting eliminates manual tracking
- Validates range of motion (ROM) compliance
- Alerts for movement deviations

### 3. Personalized Rehabilitation with RL
- Reinforcement Learning adapts exercises based on:
  - Performance progress
  - Recovery stage
  - Individual capabilities
- Dynamically adjusts:
  - Exercise difficulty
  - Repetition counts
  - Rest periods

### 4. Fatigue Detection & Safety
- Neural network monitors:
  - Movement speed degradation
  - Motion tremors
  - Form breakdowns
- Provides safety recommendations

### 5. Quality Scoring System
- K-Means Clustering evaluates each rep on:
  - ROM accuracy
  - Movement smoothness
  - Form consistency
- Real-time feedback and corrections

### 6. Cloud-Based Progress Tracking
- Firebase integration for data storage
- Remote monitoring capabilities
- Automated report generation
- Historical performance comparison



## Installation <a name="installation"></a>

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/ai-physio-system.git
   cd ai-physio-system
   ```

2. Create and activate virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure Firebase:
   - Create Firebase project
   - Add configuration details in `config/firebase_config.json`

5. Run the system:
   ```
   python main.py
   ```

## Usage <a name="usage"></a>

### Patient Interface
1. Select your prescribed exercise program
2. Position yourself within camera view
3. Follow on-screen instructions
4. Receive real-time form feedback
5. View post-session performance summary

### Therapist Interface
1. Log in to dashboard
2. View patient progress reports
3. Adjust treatment plans remotely
4. Set exercise parameters
5. Monitor compliance statistics

## Data Storage <a name="data-storage"></a>
All session data is securely stored in Firebase with:
- Exercise performance metrics
- Form analysis data
- Progress trends
- Therapist notes

## Contact <a name="contact"></a>
For questions or support, please contact:  
alexjmt1@gmail.com  
+91 9074723677
