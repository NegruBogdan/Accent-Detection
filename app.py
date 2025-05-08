import yt_dlp
import moviepy.editor as mp
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf
import os
from pydub import AudioSegment
import numpy as np
from scipy.signal import resample
import streamlit as st

# Function to extract audio from a YouTube video
def extract_audio_from_video(url):
    try:
        print(f"Extracting audio from: {url}")
        video_path = "downloaded_video.mp4"
        ydl_opts = {
            'format': '137+251',
            'outtmpl': 'downloaded_video.%(ext)s'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        downloaded_path = "downloaded_video.mkv"
        audio_output = "audio.mp3"

        video = mp.VideoFileClip(downloaded_path)
        video.audio.write_audiofile(audio_output)
        return audio_output
    except Exception as e:
        print(f"Failed to extract audio: {e}")
        return None

# Function to load the fine-tuned model and processor from Hugging Face
def load_model(model_name="BoboThePotato/BobosAudioModel"):
    # Load the processor (Wav2Vec2Processor is the same for both the original and fine-tuned models)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")  # Using the same processor as during training
    # Load the model with the fine-tuned state_dict from Hugging Face
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=23, problem_type="single_label_classification")
    model.eval()  # Set model to evaluation mode
    return model, processor

# Load the label mapping (the labels should match your fine-tuned model's labels)
def load_label_mapping():
    return {
        0: "Dutch",
        1: "German",
        2: "Czech",
        3: "Polish",
        4: "French",
        5: "Hungarian",
        6: "Finnish",
        7: "Romanian",
        8: "Slovak",
        9: "Spanish",
        10: "Italian",
        11: "Estonian",
        12: "Lithuanian",
        13: "Croatian",
        14: "Slovene",
        15: "English",
        16: "Scottish",
        17: "Irish",
        18: "Northern Irish",
        19: "Indian",
        20: "Vietnamese",
        21: "Canadian",
        22: "American"
    }

# Function to make predictions using the model
def predict_accent(audio_path, model, processor, label_mapping):
    try:
        # Read and preprocess the audio file
        speech_array, original_sr = sf.read(audio_path)
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)  # Convert stereo to mono if needed

        target_sr = 16000  # Wav2Vec2 expects 16kHz audio
        if original_sr != target_sr:
            num_samples = int(len(speech_array) * target_sr / original_sr)
            speech_array = resample(speech_array, num_samples)

        # Process the audio input
        inputs = processor(speech_array, sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        # Make the prediction
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_id].item()
        predicted_label = label_mapping[predicted_id]

        return predicted_label, confidence
    except Exception as e:
        print(f"Error during audio prediction: {e}")
        return None, None

# Streamlit app
def main():
    st.title("Accent Detection from YouTube Video")

    url = st.text_input("Enter YouTube URL:")

    if url:
        # Load the model and processor
        model, processor = load_model()

        # Load label mapping (same as during training)
        label_mapping = load_label_mapping()
        
        # Extract audio from the video
        audio_path = extract_audio_from_video(url)
        if audio_path:
            # Predict accent using the model
            predicted_label, confidence = predict_accent(audio_path, model, processor, label_mapping)
            if predicted_label is not None:
                st.write(f"Predicted Accent: {predicted_label}")
                st.write(f"Confidence: {confidence:.2f}")
            else:
                st.write("Accent prediction failed.")
        else:
            st.write("Failed to extract audio from the video.")

if __name__ == "__main__":
    main()
