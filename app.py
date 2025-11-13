import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, g
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'emotion_data.db'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your pre-trained model
model = load_model("emotion_model.h5")

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# ---------- DATABASE SETUP ----------
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    emotion TEXT NOT NULL
                )''')
    db.commit()


# Initialize database on startup
with app.app_context():
    init_db()
# -------------------------------------


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        username = request.form['username']
        file = request.files['image']
        if file and username.strip():
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (48, 48))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Predict emotion
            preds = model.predict(image)[0]
            emotion = EMOTIONS[np.argmax(preds)]

            # Store in database
            db = get_db()
            db.execute("INSERT INTO users (username, image_path, emotion) VALUES (?, ?, ?)",
                       (username, file_path, emotion))
            db.commit()

            return render_template('index.html', emotion=emotion, username=username, image_path=file_path)

    # Show all results (optional)
    db = get_db()
    records = db.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    return render_template('index.html', records=records)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5005)), debug=True)
