# from flask import Flask, render_template, request, redirect, url_for
# import sqlite3, os, cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # Load model
# model = load_model("emotion_model.h5")


# # Database setup
# def init_db():
#     conn = sqlite3.connect("emotion_users.db")
#     conn.execute('''CREATE TABLE IF NOT EXISTS users
#                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                      username TEXT,
#                      image_path TEXT,
#                      emotion TEXT)''')
#     conn.close()

# init_db()

# # Emotion labels (example)
# labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     username = request.form['username']
#     file = request.files['image']
#     if not file:
#         return redirect(url_for('home'))

#     filename = secure_filename(file.filename)
#     path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(path)

#     # Predict emotion
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (48,48))
#     img = img_to_array(img)/255.0
#     img = np.expand_dims(img, axis=0)

#     prediction = model.predict(img)
#     emotion = labels[np.argmax(prediction)]

#     # Store in DB
#     conn = sqlite3.connect("emotion_users.db")
#     conn.execute("INSERT INTO users (username, image_path, emotion) VALUES (?, ?, ?)",
#                  (username, path, emotion))
#     conn.commit()
#     conn.close()

#     return render_template('index.html', emotion=emotion, image_path=path, username=username)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, redirect, url_for
import sqlite3, os, cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# --- Flask app setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load trained model ---
# Make sure emotion_model.h5 exists in the same folder
try:
    model = load_model("emotion_model.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)

# --- Database setup ---
def init_db():
    conn = sqlite3.connect("emotion_users.db")
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
    conn = sqlite3.connect("emotion_users.db")
    conn.execute("INSERT INTO users (username, image_path, emotion) VALUES (?, ?, ?)",
                 (username, path, emotion))
    conn.commit()
    conn.close()

    # --- Display result ---
    return render_template('index.html', emotion=emotion, image_path=path, username=username)

# --- Run app ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
