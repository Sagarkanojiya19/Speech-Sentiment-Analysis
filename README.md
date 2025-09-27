# 🎤 Speech Sentiment Analysis


A modern React/Next.js frontend with Python TensorFlow backend for real-time speech sentiment analysis.

## Features

- 🎤 Real-time speech recording and transcription
- 🤖 AI-powered sentiment analysis using your trained TensorFlow model
- 📊 Interactive sentiment confidence visualization
- 🎨 Modern dark theme with glassmorphism effects
- 📱 Responsive design for all devices

## Setup Instructions

### 1. Frontend 

The frontend is already set up and ready to run. It will automatically start when you preview the project.

### 2. Backend (Python API)

You need to run the Python API backend separately to get real sentiment predictions.

#### Prerequisites

1. **Model Files**: Place these files in the `api/` directory:
   - `Model.h5` - Your trained TensorFlow model
   - `glove.6B.100d.txt` - GloVe embeddings file

### 3. Using the App

1. **Start the Python API** (as described above)
2. **Open the React app** in v0 preview
3. **Click the microphone button** to start recording
4. **Speak clearly** for up to 5 seconds
5. **View results** - transcription and real sentiment analysis

### Model Loading Issues
- Ensure `Model.h5` is the exact model file from your training
- Verify `glove.6B.100d.txt` is the correct GloVe embeddings file
- Check Python console for specific error messages

### Browser Issues
- Enable microphone permissions when prompted
- Use Chrome/Edge for best Web Speech API support
- Check browser console for JavaScript errors

### Using the Interface

1. **Text Analysis Tab**:
   - Enter text in the input field
   - Click "🔍 Analyze Sentiment"
   - View results with confidence scores

2. **Speech Analysis Tab**:
   - Set recording duration (2-10 seconds)
   - Click "🎙️ Start Recording"
   - Speak clearly into your microphone
   - View sentiment analysis of transcribed speech
