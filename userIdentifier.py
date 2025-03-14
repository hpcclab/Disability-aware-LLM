import os
import numpy as np
import librosa
import soundfile as sf
import hashlib
import sqlite3
from flask import Flask, request, jsonify
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# Connect to database
conn = sqlite3.connect('voice_users.db', check_same_thread=False)
cursor = conn.cursor()
# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT UNIQUE,
        features BLOB
    )
''')
conn.commit()
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)
def generate_id(features):
    return hashlib.sha256(features.tobytes()).hexdigest()[:10]
def register_user(audio_path):
    features = extract_features(audio_path)
    user_id = generate_id(features)
    # Store in database
    cursor.execute("INSERT INTO users (user_id, features) VALUES (?, ?)", (user_id, features.tobytes()))
    conn.commit()
    return user_id
def verify_user(audio_path):
    features = extract_features(audio_path)
    # Retrieve stored users
    cursor.execute("SELECT user_id, features FROM users")
    users = cursor.fetchall()
    for user_id, stored_features in users:
        stored_features = np.frombuffer(stored_features, dtype=np.float32)
        # Compute similarity (Euclidean distance)
        if np.linalg.norm(features - stored_features) < 10:  # Threshold tuning required
            return user_id
    return None
app = Flask(__name__)
@app.route('/register', methods=['POST'])
def register():
    audio = request.files['audio']
    file_path = "temp.wav"
    audio.save(file_path)
    user_id = register_user(file_path)
    os.remove(file_path)
    return jsonify({'user_id': user_id})
@app.route('/verify', methods=['POST'])
def verify():
    audio = request.files['audio']
    file_path = "temp.wav"
    audio.save(file_path)
    user_id = verify_user(file_path)
    os.remove(file_path)
    if user_id:
        return jsonify({'user_id': user_id})
    else:
        return jsonify({'error': 'User not recognized'}), 401
if __name__ == '__main__':
    app.run(debug=True)