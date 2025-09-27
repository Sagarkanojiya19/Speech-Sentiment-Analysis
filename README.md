# Speech Sentiment Analysis App

A modern React/Next.js frontend with Python TensorFlow backend for real-time speech sentiment analysis.

## Features

- 🎤 Real-time speech recording and transcription
- 🤖 AI-powered sentiment analysis using your trained TensorFlow model
- 📊 Interactive sentiment confidence visualization
- 🎨 Modern dark theme with glassmorphism effects
- 📱 Responsive design for all devices

## Setup Instructions

### 1. Frontend (React/Next.js)

The frontend is already set up and ready to run in v0. It will automatically start when you preview the project.

### 2. Backend (Python API)

You need to run the Python API backend separately to get real sentiment predictions.

#### Prerequisites

1. **Model Files**: Place these files in the `api/` directory:
   - `Model.h5` - Your trained TensorFlow model
   - `glove.6B.100d.txt` - GloVe embeddings file

2. **Python Environment**: Install Python 3.8+ and pip

#### Installation

1. Navigate to the `api/` directory:
   \`\`\`bash
   cd api/
   \`\`\`

2. Install Python dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Make sure your model files are in the `api/` directory:
   \`\`\`
   api/
   ├── sentiment_api.py
   ├── requirements.txt
   ├── Model.h5              # Your trained model
   └── glove.6B.100d.txt     # GloVe embeddings
   \`\`\`

4. Run the Python API server:
   \`\`\`bash
   python sentiment_api.py
   \`\`\`

   You should see:
   \`\`\`
   ✅ Model and embeddings loaded successfully!
   🚀 Starting Flask API server...
   * Running on http://0.0.0.0:5000
   \`\`\`

### 3. Using the App

1. **Start the Python API** (as described above)
2. **Open the React app** in v0 preview
3. **Click the microphone button** to start recording
4. **Speak clearly** for up to 5 seconds
5. **View results** - transcription and real sentiment analysis

## API Endpoints

- `GET /health` - Check if the API and model are loaded
- `POST /predict` - Analyze sentiment for given text

Example API usage:
\`\`\`bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this amazing product!"}'
\`\`\`

## Troubleshooting

### API Connection Issues
- Make sure the Python API is running on port 5000
- Check that both `Model.h5` and `glove.6B.100d.txt` are in the `api/` directory
- Verify no firewall is blocking port 5000

### Model Loading Issues
- Ensure `Model.h5` is the exact model file from your training
- Verify `glove.6B.100d.txt` is the correct GloVe embeddings file
- Check Python console for specific error messages

### Browser Issues
- Enable microphone permissions when prompted
- Use Chrome/Edge for best Web Speech API support
- Check browser console for JavaScript errors

## Architecture

- **Frontend**: Next.js/React with TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Flask API with TensorFlow, scikit-learn, numpy
- **ML Model**: Your trained TensorFlow model with GloVe embeddings
- **Communication**: REST API calls between frontend and backend
