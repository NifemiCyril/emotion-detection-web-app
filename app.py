from flask import Flask, render_template, request, redirect, url_for
import sqlite3, os, cv2, requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# --- Flask app setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model setup ---
MODEL_PATH = "emotion_model.h5"

def download_model():
    """Download model from Google Drive if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1NGYYh0DWY8krPe96k7PTHoQi58O9zo9U"
        try:
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            print("✅ Model download complete.")
        except Exception as e:
            print("❌ Failed to download model:", e)

# Download model if needed
download_model()

# Load model
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)

# --- Database setup ---
DB_NAME = "emotion_users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT,
                     image_path TEXT,
                     emotion TEXT)''')
    conn.close()

init_db()

# --- Emotion labels ---
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    username = request.form.get('username')
    file = request.files.get('image')

    if not username or not file:
        return render_template('index.html', error="Please provide a name and an image.")

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # --- Predict emotion ---
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        emotion = labels[np.argmax(prediction)]
    except Exception as e:
        return render_template('index.html', error=f"Prediction failed: {e}")

    # --- Save to database ---
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute("INSERT INTO users (username, image_path, emotion) VALUES (?, ?, ?)",
                     (username, path, emotion))
        conn.commit()
        conn.close()
    except Exception as e:
        return render_template('index.html', error=f"Database error: {e}")

    # --- Display result ---
    return render_template('index.html', emotion=emotion, image_path=path, username=username)

# --- View all records ---
@app.route('/records')
def records():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT username, image_path, emotion FROM users ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return render_template('records.html', data=data)

# --- Run app ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
