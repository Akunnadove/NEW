import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tempfile
import os

st.set_page_config(page_title="🎼 Music Describer", layout="centered")
st.title("🎼 Music Describer")
st.markdown("Upload a music/audio file to get tempo, key, duration, and a generated description.")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    try:
        key = librosa.key.estimate_key(y)
    except:
        key = "Unknown"
    duration = librosa.get_duration(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    energy = "high" if rms > 0.04 else "medium" if rms > 0.02 else "low"

    return {
        "tempo": round(tempo, 2),
        "key": key,
        "duration": round(duration, 2),
        "energy": energy
    }

def generate_description(features):
    mood = "energetic" if features["energy"] == "high" else "relaxed" if features["energy"] == "low" else "balanced"
    genre = "electronic" if features["tempo"] > 120 else "lo-fi" if features["tempo"] < 80 else "pop"
    use_case = "ideal for upbeat videos" if features["energy"] == "high" else "great for background or study sessions"

    return f"""🎵 **Track Overview**
- Tempo: {features['tempo']} BPM
- Key: {features['key']}
- Duration: {features['duration']} seconds
- Energy Level: {features['energy']}

📝 **Mood & Style**
This track has a {mood} mood and fits within the {genre} genre range. It's {use_case}.

📌 **Suggested Tags**: #{genre} #{mood} #musicdescription
"""

audio_file = st.file_uploader("Upload an audio file (.mp3, .wav, .ogg)", type=["mp3", "wav", "ogg"])

if audio_file:
    st.audio(audio_file, format="audio/wav")

    if st.button("🧠 Analyze"):
        with st.spinner("Extracting audio features..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            try:
                features = extract_features(tmp_path)
                description = generate_description(features)

                st.subheader("🔍 Features")
                st.json(features)

                st.subheader("📝 Description")
                st.markdown(description)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                os.remove(tmp_path)
