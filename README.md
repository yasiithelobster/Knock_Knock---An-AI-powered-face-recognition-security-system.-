# 🔐 Knock Knock!

**AI-Based Face Recognition Security System**

Knock Knock! is an AI-powered security system that uses facial recognition to control access. It allows administrators to register authorized individuals, build a facial embedding database, and perform real-time identity verification using a webcam.

---

## 🌟 Features

- 🔑 Admin-Protected Dashboard
    - Secure login required to access system controls
- 📸 Face Enrollment
    - Capture multiple images per person using webcam
    - Ensures only front-facing single-face images are stored
- 🧠 Embedding-Based Recognition
    - Converts faces into numerical embeddings (feature vectors)
    - More flexible and accurate than traditional classification
- 🗄️ Database Rebuild System
    - Processes stored images into embeddings
    - Updates the system’s recognition memory
- 🎥 Real-Time Access Gate
    - Detects faces via webcam
    - Matches against stored embeddings
    - Grants or denies access
- 📊 Access Logging
    - Records:
    - Name
    - Access status
    - Timestamp
    - Matching distance
- 🌐 Web Dashboard (Flask)
    - Central control interface
    - No terminal interaction required

---

## 🧠 How It Works

The system follows this pipeline:

```
Enrollment → Dataset → Embedding Database → Live Recognition → Access Decision → Logging
```

1. Enrollment
- Admin registers a person
- System captures ~30 face images
- Images stored in:
```
dataset/<person_name>/
```

2. Database Rebuild
- All images are processed
- Faces are converted into embeddings
- Stored in:

  ```
  models/embedding_database.pkl
  ```

👉 This file acts as the system’s AI memory


--- 

## 3. Face Recognition (Access Gate)
- Webcam captures live video
- Face is detected and converted into embedding
- Compared against stored embeddings

Decision Logic:
- If distance < threshold → ✅ Access Granted
- Else → ❌ Access Denied

---

## 4. Logging

Every verified attempt is recorded:
```
logs/access_log.txt
```

Example:
```
2026-04-01 10:30:22 - yasiru - Access Granted - Distance: 0.42
```

---

## 🧰 Technologies & Libraries Used

**🐍 Core Language**
- Python 3.13

**🧠 AI & Computer Vision**
- face_recognition – face embeddings & detection
- OpenCV (cv2) – webcam handling & image processing
- NumPy – numerical operations

**🌐 Web Framework**
- Flask – dashboard & routing

**📦 Other**
- pickle – storing embedding database
- os, sys, subprocess – system operations
- datetime – logging timestamps


---

## 📁 Project Structure

```
Knock_Knock/
├── dataset/                # Enrolled face images
├── models/
│   └── embedding_database.pkl
├── logs/
│   └── access_log.txt
├── outputs/                # Training visuals (optional)
├── templates/              # HTML pages
├── static/                 # CSS styling
├── app.py                  # Face recognition (access gate)
├── build_database.py       # Embedding generator
├── enroll.py               # Face capture
├── dashboard.py            # Flask web app
├── settings.py             # Config (password, secret key)
├── requirements.txt
```

---

## 🚀 How to Run

1. Install dependencies
```
pip install -r requirements.txt
```

2. Start the dashboard
```
python dashboard.py
```
Open in browser:

```
http://127.0.0.1:5000
```

3. Workflow

Step 1: Login
- Enter admin password

Step 2: Register Person
- Enter name
- Capture face images

Step 3: Rebuild Database
- Converts images → embeddings

Step 4: Start Access Gate
- Opens webcam
- Performs real-time recognition

Step 5: View Logs
- See system activity

---

## ⚙️ Configuration

In settings.py:
```
ADMIN_PASSWORD = "knock123"
SECRET_KEY = "knock_knock_secret_2026"
```

---

## ⚠️ Limitations
- Works best with front-facing faces
- Lighting and camera quality affect accuracy
- Not optimized for multi-face authentication
- Uses local storage (no cloud/database)

---

## 🎓 Project Type
-	AI / Machine Learning
-	Computer Vision
-	Real-Time System
-	Security Application

---


## 👨‍💻 Author

Developed by **Yasiru Athauda**
📧 yasirudilmith34@gmail.com














