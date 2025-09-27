
# 🎤 Speech Sentiment Analysis

A comprehensive sentiment analysis application that processes both text and speech input using deep learning and natural language processing techniques.

## Features

- **Text Sentiment Analysis**: Analyze sentiment from written text
- **Speech-to-Text + Sentiment**: Record audio, convert to text, and analyze sentiment
- **Real-time Processing**: Live audio recording and immediate sentiment prediction
- **Interactive Web Interface**: Clean, user-friendly Gradio interface
- **Confidence Scores**: Detailed prediction confidence for all sentiment classes

## Sentiment Classes

- 😞 **Negative**: Sad, angry, disappointed emotions
- 😐 **Neutral**: Objective, factual statements  
- 😊 **Positive**: Happy, satisfied, excited emotions

## Technology Stack

- **Deep Learning**: TensorFlow/Keras LSTM model
- **Word Embeddings**: GloVe (Global Vectors for Word Representation)
- **Speech Recognition**: Google Speech-to-Text API
- **Audio Processing**: SoundDevice for real-time recording
- **Web Interface**: Gradio for interactive UI
- **Text Processing**: scikit-learn, NLTK, regex

## Requirements

### Files Needed
- `Model.h5` - Pre-trained LSTM sentiment analysis model
- `glove.6B.100d.txt` - GloVe word embeddings (100-dimensional)
- `Tweets.csv` - Training dataset (for reference)

### System Requirements
- Python 3.8+
- Working microphone for speech analysis
- Internet connection for speech-to-text API

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

## Acknowledgments

- Stanford NLP Group for GloVe embeddings
- Google for Speech-to-Text API
- Gradio team for the amazing interface framework
- Twitter for the sentiment dataset
