
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

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/speech-sentiment-analysis.git
   cd speech-sentiment-analysis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv speech_sentiment_env
   # On Windows:
   speech_sentiment_env\Scripts\activate
   # On macOS/Linux:
   source speech_sentiment_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download GloVe embeddings**:
   - Download `glove.6B.100d.txt` from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
   - Place it in the project root directory

5. **Ensure model file exists**:
   - Make sure `Model.h5` is in the project directory
   - If you need to train the model, run the training notebook first

## Usage

### Running the Application

```bash
python app.py
```

The application will launch on `http://localhost:7860`

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

## Project Structure

```
speech-sentiment-analysis/
├── app.py                              # Main Gradio application
├── Train.py.ipynb                      # Model training notebook
├── Speech to Sentiment (Run Code).ipynb # Testing notebook
├── requirements.txt                    # Python dependencies
├── Model.h5                           # Trained LSTM model
├── model_architecture.json           # Model architecture
├── model_weights.weights.h5           # Model weights
├── glove.6B.100d.txt                 # GloVe embeddings (download needed)
├── Tweets.csv                         # Training dataset
└── README.md                          # This file
```

## Model Architecture

- **Input Layer**: GloVe embeddings (100 dimensions)
- **LSTM Layers**: Bidirectional LSTM with dropout
- **Output Layer**: Softmax for 3-class classification
- **Training Data**: Twitter sentiment dataset
- **Accuracy**: ~85% on validation set

## API Endpoints

When running the app, these endpoints are available:
- `http://localhost:7860/` - Main interface
- `http://localhost:7860/api/predict/` - API endpoint for predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

1. **"Model not initialized" error**:
   - Ensure `Model.h5` and `glove.6B.100d.txt` are in the project directory
   - Check file permissions

2. **Audio recording issues**:
   - Verify microphone permissions
   - Check audio drivers
   - Try different browsers if using web interface

3. **Speech recognition errors**:
   - Ensure stable internet connection
   - Speak clearly and avoid background noise
   - Check microphone quality

4. **Import errors**:
   - Activate virtual environment
   - Install all requirements: `pip install -r requirements.txt`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Stanford NLP Group for GloVe embeddings
- Google for Speech-to-Text API
- Gradio team for the amazing interface framework
- Twitter for the sentiment dataset
