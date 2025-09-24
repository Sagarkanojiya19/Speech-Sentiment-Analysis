import gradio as gr
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write
import re
import os
import tempfile
import time

# Global variables for model and embeddings
model = None
label_encoder = None
embeddings_index = None

def load_glove_embeddings(filepath):
    """Load GloVe embeddings from file"""
    embeddings_index = {}
    try:
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index
    except FileNotFoundError:
        return None

def initialize_model():
    """Initialize the model and embeddings"""
    global model, label_encoder, embeddings_index
    
    try:
        # Load the pre-trained model
        model = tf.keras.models.load_model('Model.h5', compile=False)
        
        # Initialize label encoder with class labels
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(['negative', 'neutral', 'positive'])
        
        # Load GloVe embeddings
        embeddings_index = load_glove_embeddings('glove.6B.100d.txt')
        
        if embeddings_index is None:
            return False, "❌ GloVe embeddings file not found. Please ensure 'glove.6B.100d.txt' is in the project directory."
        
        return True, "✅ Model and embeddings loaded successfully!"
        
    except Exception as e:
        return False, f"❌ Error loading model: {str(e)}"

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Remove non-alphanumeric characters
    text = text.lower()                        # Convert to lowercase
    return text.strip()

def vectorize_text_with_glove(text, embeddings_index):
    """Vectorize text using GloVe embeddings"""
    if not embeddings_index:
        return np.zeros(100)  # Return zero vector if no embeddings
    
    words = text.split()
    embedding_dim = len(next(iter(embeddings_index.values())))
    text_vector = np.zeros(embedding_dim)
    count = 0
    
    for word in words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            text_vector += embedding_vector
            count += 1
    
    if count != 0:
        text_vector /= count
    
    return text_vector

def predict_sentiment_from_text(text):
    """Predict sentiment from text input"""
    global model, label_encoder, embeddings_index
    
    if model is None or embeddings_index is None:
        return "❌ Model not initialized. Please check if all required files are present.", "", {"Negative": 0, "Neutral": 0, "Positive": 0}
    
    if not text.strip():
        return "❌ Please enter some text to analyze.", "", {"Negative": 0, "Neutral": 0, "Positive": 0}
    
    try:
        # Preprocess and vectorize text
        preprocessed_text = preprocess_text(text)
        if not preprocessed_text:
            return "❌ No valid text found after preprocessing.", "", {"Negative": 0, "Neutral": 0, "Positive": 0}
        
        text_vector = vectorize_text_with_glove(preprocessed_text, embeddings_index)
        
        # Reshape for model input (timesteps, features)
        text_vector_reshaped = text_vector.reshape((1, 1, len(text_vector)))
        
        # Predict sentiment
        pred_probs = model.predict(text_vector_reshaped, verbose=0)
        pred_label_index = np.argmax(pred_probs, axis=1)
        pred_label = label_encoder.inverse_transform(pred_label_index)[0]
        
        # Get confidence scores
        confidence_scores = pred_probs[0]
        labels = ['Negative', 'Neutral', 'Positive']
        confidence_dict = {labels[i]: float(confidence_scores[i]) for i in range(len(labels))}
        
        # Format result
        emoji_map = {'negative': '😞', 'neutral': '😐', 'positive': '😊'}
        result = f"{emoji_map.get(pred_label, '')} **{pred_label.upper()}**"
        
        return result, f"Preprocessed text: '{preprocessed_text}'", confidence_dict
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}", "", {"Negative": 0, "Neutral": 0, "Positive": 0}

def record_audio_and_predict(duration):
    """Record audio and predict sentiment"""
    global model, label_encoder, embeddings_index
    
    if model is None or embeddings_index is None:
        return "❌ Model not initialized. Please check if all required files are present.", "", "", {"Negative": 0, "Neutral": 0, "Positive": 0}
    
    try:
        # Create temporary file for recording
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            temp_filename = tmp_file.name
        
        # Record audio
        samplerate = 16000
        print(f"🎙️ Recording for {duration} seconds...")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is complete
        
        # Save recording
        write(temp_filename, samplerate, recording)
        
        # Convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_filename) as source:
            audio = recognizer.record(source)
        
        recognized_text = recognizer.recognize_google(audio)
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        if not recognized_text.strip():
            return "❌ No speech detected. Please try again.", "", "", {"Negative": 0, "Neutral": 0, "Positive": 0}
        
        # Predict sentiment
        sentiment_result, processed_text, confidence_scores = predict_sentiment_from_text(recognized_text)
        
        return sentiment_result, f"Recognized text: '{recognized_text}'", processed_text, confidence_scores
        
    except sr.UnknownValueError:
        return "❌ Could not understand the audio. Please speak clearly and try again.", "", "", {"Negative": 0, "Neutral": 0, "Positive": 0}
    except sr.RequestError as e:
        return f"❌ Speech recognition service error: {str(e)}", "", "", {"Negative": 0, "Neutral": 0, "Positive": 0}
    except Exception as e:
        return f"❌ Error during audio recording/processing: {str(e)}", "", "", {"Negative": 0, "Neutral": 0, "Positive": 0}

# Initialize the model at startup
init_success, init_message = initialize_model()

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    title="Speech Sentiment Analysis",
    css="""
    .gradio-container {
        max-width: 1000px !important;
    }
    .status-box {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    """
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🎤 Speech Sentiment Analysis</h1>
        <p>Analyze sentiment from text or speech using deep learning and GloVe embeddings</p>
    </div>
    """)
    
    # Status display
    status_display = gr.HTML(
        value=f"<div class='status-box' style='background-color: {'#d4edda' if init_success else '#f8d7da'}; color: {'#155724' if init_success else '#721c24'}'>{init_message}</div>"
    )
    
    with gr.Tabs() as tabs:
        
        # Text Analysis Tab
        with gr.Tab("📝 Text Analysis", id="text_tab"):
            gr.Markdown("### Enter text to analyze its sentiment")
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter the text you want to analyze for sentiment...",
                        lines=4,
                        max_lines=8
                    )
                    
                    text_analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    text_result = gr.HTML(label="Sentiment Result")
                    text_processed = gr.Textbox(label="Processing Info", interactive=False)
                    text_confidence = gr.Label(label="Confidence Scores", num_top_classes=3)
        
        # Speech Analysis Tab
        with gr.Tab("🎙️ Speech Analysis", id="speech_tab"):
            gr.Markdown("### Record your speech and analyze its sentiment")
            
            with gr.Row():
                with gr.Column(scale=1):
                    duration_slider = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Recording Duration (seconds)"
                    )
                    
                    speech_record_btn = gr.Button("🎙️ Start Recording", variant="primary", size="lg")
                    
                    gr.Markdown("**Instructions:**\n- Click the button above to start recording\n- Speak clearly into your microphone\n- Wait for the analysis to complete")
                
                with gr.Column(scale=2):
                    speech_result = gr.HTML(label="Sentiment Result")
                    speech_recognized = gr.Textbox(label="Recognized Text", interactive=False)
                    speech_processed = gr.Textbox(label="Processing Info", interactive=False)
                    speech_confidence = gr.Label(label="Confidence Scores", num_top_classes=3)
    
    # About section
    with gr.Accordion("ℹ️ About This Application", open=False):
        gr.Markdown("""
        This application uses:
        - **Deep Learning Model**: LSTM-based neural network for sentiment classification
        - **GloVe Embeddings**: Pre-trained word vectors for text representation
        - **Speech Recognition**: Google's speech-to-text API
        - **Audio Processing**: Real-time audio recording and processing
        
        **Sentiment Classes:**
        - 😞 **Negative**: Sad, angry, disappointed emotions
        - 😐 **Neutral**: Objective, factual statements
        - 😊 **Positive**: Happy, satisfied, excited emotions
        
        **Requirements:**
        - Model.h5 (trained model file)
        - glove.6B.100d.txt (GloVe embeddings)
        - Working microphone for speech analysis
        """)
    
    # Event handlers
    text_analyze_btn.click(
        fn=predict_sentiment_from_text,
        inputs=[text_input],
        outputs=[text_result, text_processed, text_confidence]
    )
    
    speech_record_btn.click(
        fn=record_audio_and_predict,
        inputs=[duration_slider],
        outputs=[speech_result, speech_recognized, speech_processed, speech_confidence]
    )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["I love this product! It works amazing and I'm so happy with it."],
            ["The service was terrible and I'm really disappointed."],
            ["The weather is cloudy today."],
            ["This movie is absolutely fantastic! Best film I've seen this year."],
            ["I hate waiting in long queues. It's so frustrating."]
        ],
        inputs=text_input,
        outputs=[text_result, text_processed, text_confidence],
        fn=predict_sentiment_from_text,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False
    )
