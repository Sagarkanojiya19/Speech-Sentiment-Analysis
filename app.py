import streamlit as st
import numpy as np
import sounddevice as sd
import tempfile
import os
import tensorflow as tf
import speech_recognition as sr
from scipy.io.wavfile import write as wav_write
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.express as px
import html

# ======================
# Load Model + Resources
# ======================

def load_model_robust():
    """Load Keras model handling legacy InputLayer(batch_shape) and provide fallbacks.
    Order:
    1) Try direct load of Model.h5 with custom_objects
    2) Fallback: load from model_architecture.json + model_weights.weights.h5
    """
    # Custom InputLayer that converts batch_shape to input_shape for deserialization
    class CustomInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, **kwargs):
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape and len(batch_shape) > 1 and batch_shape[0] is None:
                    kwargs['input_shape'] = tuple(batch_shape[1:])
            super().__init__(**kwargs)

    # DTypePolicy shim for some older saved models
    class DTypePolicy:
        def __init__(self, name):
            self.name = name
            self.compute_dtype = 'float32'
            self.variable_dtype = 'float32'
        def __eq__(self, other):
            if isinstance(other, str):
                return self.name == other
            return getattr(other, 'name', None) == self.name

    custom_objects = {
        'InputLayer': CustomInputLayer,
        'DTypePolicy': DTypePolicy,
    }

    # 1) Try Model.h5 directly
    try:
        return tf.keras.models.load_model('Model.h5', compile=False, custom_objects=custom_objects)
    except Exception as e1:
        st.warning(f"Direct model load failed: {e1}")

    # 2) Fallback to JSON architecture + weights
    arch_path = 'model_architecture.json'
    weights_path = 'model_weights.weights.h5'
    if os.path.exists(arch_path) and os.path.exists(weights_path):
        try:
            with open(arch_path, 'r', encoding='utf-8') as f:
                arch_json = f.read()
            model = tf.keras.models.model_from_json(arch_json, custom_objects=custom_objects)
            model.load_weights(weights_path)
            return model
        except Exception as e2:
            st.error(f"Failed to load model from JSON+weights: {e2}")

    st.error("Could not load model. Ensure 'Model.h5' or JSON+weights are present and compatible.")
    return None

@st.cache_resource(show_spinner=False)
def load_glove_embeddings(filepath: str):
    """Load GloVe embeddings (100d) into a dictionary."""
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

@st.cache_resource(show_spinner=False)
def load_resources():
    model_local = load_model_robust()
    embeddings_index_local = load_glove_embeddings('glove.6B.100d.txt')
    le = LabelEncoder()
    le.classes_ = np.array(['negative', 'neutral', 'positive'])
    return model_local, embeddings_index_local, le

model, embeddings_index, label_encoder = load_resources()
if model is None or embeddings_index is None:
    st.error("Failed to load model or GloVe embeddings. Ensure files are present.")
    st.stop()

MAX_LEN = 100

# ======================
# Helper Functions
# ======================
def record_audio(duration=5, fs=16000):
    """Record mono audio and save as PCM 16-bit WAV compatible with SpeechRecognition."""
    # Record directly as int16 for PCM WAV compatibility
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    # Create a temp file path and ensure no handle conflicts on Windows
    fd, file_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wav_write(file_path, fs, recording.squeeze())  # int16 array -> PCM WAV
    return file_path

def transcribe_audio(file_path):
    r = sr.Recognizer()
    # Quick sanity checks
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return "Could not transcribe audio."
    try:
        with sr.AudioFile(file_path) as source:
            audio = r.record(source)
        try:
            return r.recognize_google(audio)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return "Could not transcribe audio."
    except Exception as e:
        st.error(f"Audio file read error: {e}")
        return "Could not transcribe audio."

def preprocess_text(text: str) -> str:
    """Preprocess text exactly like the notebook: remove mentions, non-letters, lowercase."""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def vectorize_text_with_glove(text: str, embeddings_index_local: dict) -> np.ndarray:
    """Average GloVe vectors for tokens present."""
    words = text.split()
    # Use dimension from first vector (100d expected)
    embedding_dim = len(next(iter(embeddings_index_local.values())))
    text_vector = np.zeros(embedding_dim, dtype=np.float32)
    count = 0
    for word in words:
        vec = embeddings_index_local.get(word)
        if vec is not None:
            text_vector += vec
            count += 1
    if count:
        text_vector /= count
    return text_vector

def predict_sentiment(text: str):
    """Notebook-style prediction: preprocess -> GloVe avg -> reshape (1,1,100) -> model.predict."""
    if model is None:
        return "neutral", np.array([0.33, 0.34, 0.33], dtype=np.float32)
    pre = preprocess_text(text)
    vec = vectorize_text_with_glove(pre, embeddings_index)
    x = vec.reshape((1, 1, vec.shape[0]))  # (batch=1, timesteps=1, features=100)
    pred_probs = model.predict(x, verbose=0)[0]
    label = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]
    return label.capitalize(), pred_probs

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="🎤 Speech Sentiment App", page_icon="🎤", layout="wide")

# Modern dark theme styling
st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #0b0b0b, #141414); }
    #MainMenu, header, footer { visibility: hidden; }
    .main .block-container { padding-top: 0.6rem; }

    .title-container { text-align:center; padding: 1.5rem 0 0.5rem; }
    .title { font-size: 2.2rem; color: #ffffff; margin: 0; font-weight: 700; }
    .subtitle { color:#cfcfcf; font-size:1rem; margin-top: 0.25rem; }

    .card {
        background: linear-gradient(135deg, rgba(30,30,30,0.85), rgba(18,18,18,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.6rem 1.4rem;
        margin: 0.9rem auto;
        color: #fff;
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
        text-align: center;
        max-width: 980px;
    }

    .hero-card {
        padding: 2.2rem 1.6rem;
    }

    .hero-title { font-size: 3.4rem; font-weight: 800; line-height: 1.05; margin: 0 0 .4rem; }
    .hero-subtitle { color:#cfcfcf; font-size: 1.15rem; margin: 0; }

    .analysis-title { font-size: 1.6rem; font-weight: 750; margin: 0 0 .35rem; }
    .analysis-subtitle { color:#bdbdbd; font-size: 1rem; margin: 0 0 .8rem; }

    .mic-area { width:100%; display:flex; justify-content:center; align-items:center; padding:.4rem 0 .1rem; margin: 0 auto; }
    .mic-area .stButton>button {
        width: clamp(72px, 8vw, 96px) !important; height: clamp(72px, 8vw, 96px) !important;
        min-width: unset !important; 
        border-radius: 999px !important; border: none !important; outline: none !important;
        background: #47d764 !important; color:#0b0b0b !important; font-size: 26px !important; font-weight: 800 !important;
        box-shadow: 0 6px 16px rgba(71,215,100,0.30) !important;
        transition: transform .18s ease, box-shadow .18s ease !important;
        position: relative; display: inline-flex; align-items:center; justify-content:center;
        padding: 0 !important; 
        animation: ringPulse 2.6s ease-out infinite;
    }
    .mic-area .stButton>button:hover { transform: translateY(-1px) scale(1.03); box-shadow: 0 14px 28px rgba(71,215,100,0.42) !important; }

    @keyframes ringPulse {
        0% { box-shadow: 0 0 0 0 rgba(71,215,100,0.40); }
        60% { box-shadow: 0 0 0 12px rgba(71,215,100,0.00); }
        100% { box-shadow: 0 0 0 0 rgba(71,215,100,0.00); }
    }

    .badge {
        display:inline-block; padding: 6px 14px; border-radius: 270px;
        background: rgba(255,255,255,0.08); font-weight: 700;
    }
    .muted { color: #cfcfcf; font-size: .95rem; }

    /* Info cards grid */
    .info-grid { 
        display: grid; 
        grid-template-columns: 1fr; 
        gap: 1rem; 
        max-width: 980px; 
        margin: 1rem auto; 
    }
    @media (min-width: 760px) {
        .info-grid { grid-template-columns: 1fr 1fr; }
    }
    .info-card {
        background: linear-gradient(135deg, rgba(30,30,30,0.85), rgba(18,18,18,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.1rem 1rem;
        color: #fff;
        box-shadow: 0 10px 22px rgba(0,0,0,0.32);
        height: 100%;
    }

    /* Global button fallback styling (ensures mic button looks correct even outside .mic-area) */
    .stButton>button {
        border: none !important;
        border-radius: 999px !important;
        width: clamp(64px, 8vw, 92px) !important;
        height: clamp(64px, 8vw, 92px) !important;
        background: #47d764 !important;
        color: #0b0b0b !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        box-shadow: 0 6px 16px rgba(71,215,100,0.30) !important;
        padding: 0 !important;
    }
    .stButton>button:hover { transform: translateY(-1px) scale(1.02); box-shadow: 0 10px 22px rgba(71,215,100,0.42) !important; }

    /* Transcription text styling */
    .transcription-text { font-size: 1.25rem; color: #ececec; margin: .5rem 0 0; }
    .transcription-card { text-align: center; }
    .centered { text-align: center; }
    /* Ensure Streamlit buttons are horizontally centered */
    div.stButton { display: flex; justify-content: center; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card hero-card">
        <div class="hero-title">Speech Sentiment<br/>Analysis</div>
        <div class="hero-subtitle">Speak your mind, AI understands your emotions in real-time</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card analysis-card">
      <div class="analysis-title">⚙️ AI Voice Analysis</div>
      <div class="analysis-subtitle">Click the microphone to start recording for 5 seconds</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='mic-area'>", unsafe_allow_html=True)
left, center, right = st.columns([1,1,1])
with center:
    mic_clicked = st.button("🎤", key="mic_record")
    st.markdown("<p class='centered muted'>Click to start recording…</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if mic_clicked:
    st.markdown("<p class='centered muted'>🎙 Recording for 5 seconds...</p>", unsafe_allow_html=True)
    with st.spinner("Recording 5 seconds..."):
        file_path = record_audio()

    # Listen card (centered)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4 class='centered'>▶️ Listen to your recording</h4>", unsafe_allow_html=True)
    try:
        audio_bytes = open(file_path, 'rb').read()
        st.audio(audio_bytes, format='audio/wav')
    except Exception as e:
        st.error(f"Audio playback failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Transcription card (centered)
    st.markdown("<p class='centered muted'>📝 Transcribing speech...</p>", unsafe_allow_html=True)
    with st.spinner("Transcribing speech..."):
        text = transcribe_audio(file_path)
    st.markdown("<div class='card transcription-card'> <h4>📝 Transcription</h4>", unsafe_allow_html=True)
    safe_text = html.escape(text if text else "(no speech detected)")
    st.markdown(f"<p class='transcription-text'>{safe_text}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction card (centered)
    if text and text != "Could not transcribe audio.":
        st.markdown("<p class='centered muted'>📊 Analyzing sentiment...</p>", unsafe_allow_html=True)
        with st.spinner("Analyzing sentiment..."):
            sentiment, probs = predict_sentiment(text)

        emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😔"}
        color_map = {"Positive": "#70e000", "Neutral": "#ffd166", "Negative": "#ff6b6b"}
        badge_color = color_map.get(sentiment, "#fff")
        badge_emoji = emoji_map.get(sentiment, "🤖")

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 class='centered'>📊 Sentiment Prediction</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='centered'><span class='badge' style='color:{badge_color}'>{badge_emoji} {sentiment}</span></div>",
            unsafe_allow_html=True
        )
        prob_df = pd.DataFrame({
            'Sentiment': ['Negative 😔', 'Neutral 😐', 'Positive 😊'],
            'Probability': probs
        })
        fig = px.bar(prob_df, x='Sentiment', y='Probability',
                     color='Sentiment',
                     color_discrete_map={
                         'Negative 😔': '#ff6b6b',
                         'Neutral 😐': '#ffd166',
                         'Positive 😊': '#70e000'
                     })
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#ffffff'), yaxis=dict(range=[0,1], tickformat='.0%', gridcolor='rgba(255,255,255,0.1)'),
                          xaxis=dict(gridcolor='rgba(255,255,255,0.05)'), margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

       

# Subtle footer

# Additional informative sections
st.markdown("""
<div class='info-grid'>
    <div class='info-card'>
        <h4>🧭 How it works</h4>
        <ol style='margin-top:.4rem;'>
                <li>Click the green mic to record for 5 seconds (mono, 16 kHz).</li>
                <li>We transcribe speech to text using an online speech recognizer.</li>
                <li>Text is cleaned and converted to a 100‑dimensional vector with GloVe.</li>
                <li>A trained neural network predicts Negative / Neutral / Positive.</li>
        </ol>
    </div>
    <div class='info-card'>
        <h4>💡 Tips for best results</h4>
        <ul style='margin-top:.4rem;'>
                <li>Speak clearly in a quiet place; keep the device close.</li>
                <li>Short sentences (5–10 words) tend to transcribe more accurately.</li>
                <li>Avoid filler words; speak at a consistent pace.</li>
        </ul>
    </div>
    <div class='info-card'>
        <h4>🔒 Privacy</h4>
        <p class='muted' style='margin:.4rem 0 0;'>Your audio is recorded locally, saved to a temporary WAV for transcription, and not stored permanently by this app.</p>
    </div>
    <div class='info-card'>
        <h4>🧰 Tech stack</h4>
        <p class='muted' style='margin:.4rem 0 0;'>Streamlit • TensorFlow/Keras • NumPy • scikit‑learn • GloVe (100d) • Plotly</p>
    </div>
</div>
""", unsafe_allow_html=True)



st.markdown('<p class="muted" style="text-align:center;margin-top:1rem;">🤖 Powered by TensorFlow • 🚀 Built with Streamlit</p>', unsafe_allow_html=True)