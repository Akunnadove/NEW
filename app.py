import streamlit as st
import torch
import torchaudio
import librosa
import tempfile
import os
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.set_page_config(page_title="🎼 Music Describer", layout="centered")
st.title("🎼 Music Describer")
st.markdown("Upload a music/audio file to get genre, mood, tempo, key, and a formatted description.")

@st.cache_resource
def load_whisper():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        device=device,
        generate_kwargs={"language": "English", "task": "translate"}
    )

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == 'cuda' else -1)

@st.cache_resource
def load_tag_model():
    extractor = AutoFeatureExtractor.from_pretrained("mtg/jamendo-tagging")
    model = AutoModelForAudioClassification.from_pretrained("mtg/jamendo-tagging")
    return extractor, model

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    try:
        key = librosa.key.estimate_key(y)
    except:
        key = "Unknown"
    duration = librosa.get_duration(y=y, sr=sr)
    return {
        "tempo": round(tempo, 2),
        "key": key,
        "duration_sec": round(duration, 2)
    }

def classify_tags(audio_path, extractor, model):
    waveform, sample_rate = torchaudio.load(audio_path)
    inputs = extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.sigmoid(logits).squeeze().numpy()
    labels = model.config.id2label
    top_tags = [(labels[i], round(float(scores[i]), 3)) for i in scores.argsort()[-5:][::-1]]
    return top_tags

def generate_description(features, tags):
    genre_tags = [t for t in tags if "genre" in t[0]]
    mood_tags = [t for t in tags if "mood" in t[0]]
    genre = genre_tags[0][0] if genre_tags else "Unknown"
    mood = mood_tags[0][0] if mood_tags else "Neutral"

    return f"""🎵 **Track Overview**
- Genre: {genre}
- Mood: {mood}
- Tempo: {features['tempo']} BPM
- Key: {features['key']}
- Duration: {features['duration_sec']} seconds

📝 **Use Cases**
Perfect for:
- YouTube background music
- Podcast intros/outros
- Study/work playlists

📌 **Tags**: #{genre.replace(" ", "")} #{mood.replace(" ", "")} #instrumental
"""

audio_file = st.file_uploader("Upload an audio file (.mp3, .ogg, .wav)", type=["mp3", "ogg", "wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")
    if st.button("🔍 Analyze Audio"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            whisper_pipe = load_whisper()
            extractor, model = load_tag_model()
            transcription = whisper_pipe(tmp_path)
            full_text = " ".join(chunk["text"] for chunk in transcription["chunks"])
            summary = load_summarizer()(full_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
            features = extract_audio_features(tmp_path)
            tags = classify_tags(tmp_path, extractor, model)
            os.remove(tmp_path)

        st.subheader("📄 Transcription")
        st.text_area("Transcription", full_text, height=250)

        st.subheader("🧠 Summary")
        st.success(summary)

        st.subheader("🎼 Tags")
        st.json(tags)

        st.subheader("📝 Generated Description")
        st.markdown(generate_description(features, tags))
