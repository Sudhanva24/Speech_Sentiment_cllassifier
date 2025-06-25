# Streamlit and Python utilities
import streamlit as st
import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import random

# ==============================================================================
# 0. Reproducibility
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ==============================================================================
# 1. Model Definitions
# ==============================================================================

class LuongAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lstm_outputs, last_hidden):
        alignment_scores = torch.bmm(lstm_outputs, last_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(alignment_scores, dim=1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context_vector, attn_weights

class CNN1D_LSTM_Attention_Model(nn.Module):
    def __init__(self, num_features, num_classes, lstm_hidden_size=128, lstm_layers=2):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        self.attention = LuongAttention()
        self.fc = nn.Linear(lstm_hidden_size * 4, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_cnn = self.cnn(x)
        x_cnn = x_cnn.permute(0, 2, 1)
        lstm_outputs, (hidden, _) = self.lstm(x_cnn)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        context_vector, _ = self.attention(lstm_outputs, last_hidden)
        combined_vector = torch.cat((last_hidden, context_vector), dim=1)
        logits = self.fc(self.dropout(combined_vector))
        return logits

# ==============================================================================
# 2. Feature Extraction
# ==============================================================================
def extract_advanced_features(file_path, sample_rate=22050, n_fft=2048, hop_length=512, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

        ref_len = mfccs.shape[1]

        def sync(feat):
            if feat.shape[1] > ref_len:
                return feat[:, :ref_len]
            elif feat.shape[1] < ref_len:
                return np.pad(feat, ((0, 0), (0, ref_len - feat.shape[1])), mode='constant')
            return feat

        features = np.vstack([
            mfccs,
            delta_mfccs,
            delta2_mfccs,
            sync(chroma),
            sync(spectral_contrast),
            sync(tonnetz),
            sync(rms)
        ])
        return features.T
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# ==============================================================================
# 3. Load Model & Preprocessors
# ==============================================================================
@st.cache_resource
def load_model_and_preprocessors():
    st.info("Loading model and preprocessors for the first time...")
    
    model_path = 'final_model_artifacts/best_model.pt'
    scaler_path = 'final_model_artifacts/scaler.joblib'
    encoder_path = 'final_model_artifacts/label_encoder.joblib'

    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)
    num_features = scaler.n_features_in_
    num_classes = len(le.classes_)

    model = model = CNN1D_LSTM_Attention_Model(num_features=num_features, num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, scaler, le

# ==============================================================================
# 4. Streamlit Interface
# ==============================================================================
st.set_page_config(layout="centered")
st.title("🎤 Audio Emotion Recognition")
st.markdown("This app uses a 1D-CNN + LSTM + Attention model trained on RAVDESS to predict emotions from speech.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Analyze Emotion", type="primary"):
        with st.spinner("Processing audio and making a prediction..."):
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())

            features = extract_advanced_features("temp_audio.wav")
            if features is not None:
                model, scaler, le = load_model_and_preprocessors()
                scaled = scaler.transform(features)
                features_tensor = torch.from_numpy(scaled).float().unsqueeze(0)

                with torch.no_grad():
                    logits = model(features_tensor)
                    probs = F.softmax(logits, dim=1)
                    predicted_index = torch.argmax(probs, dim=1).item()
                    predicted_label = le.classes_[predicted_index]
                    confidence = probs[0][predicted_index].item()

                st.success(f"**Predicted Emotion:** {predicted_label.capitalize()} (Confidence: {confidence:.2%})")
                
                emotions = ['angry','calm','disgust','fear','happy','sad','surprise']
                probs = probs.numpy().flatten()
                print(len(emotions))
                print(le.classes_)
                if len(emotions) != len(probs):
                    st.error(f"Mismatch: {len(emotions)} labels vs {len(probs)} probabilities. Fix the encoder or model.")
                else:
                    confidence_df = pd.DataFrame({
                        'Emotion': emotions,
                        'Probability': probs
                    })
                    st.write("### Confidence Distribution")
                    st.bar_chart(confidence_df.set_index("Emotion"))
        
              
            os.remove("temp_audio.wav")
else:
    st.info("Upload a speech audio file to begin.")