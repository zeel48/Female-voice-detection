import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
model = load_model('svm_model.h5')  # Replace with the path to your trained model file


# Function to predict emotion
def predict_emotion(audio_file):
    # Load and preprocess the audio file
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    input_data = np.expand_dims(mfccs_scaled, axis=0)
    
    # Predict emotion
    predictions = model.predict(input_data)
    emotion_label = np.argmax(predictions)
    emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    predicted_emotion = emotions[emotion_label]
    return predicted_emotion

# Function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        emotion_label = predict_emotion(file_path)
        label.config(text="Predicted Emotion: " + emotion_label)

# Create GUI
root = tk.Tk()
root.title("Emotion Recognition")

label = tk.Label(root, text="No file selected")
label.pack(pady=10)

button_select = tk.Button(root, text="Select Audio File", command=select_file)
button_select.pack(pady=5)

root.mainloop()
