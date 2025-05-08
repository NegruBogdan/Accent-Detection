from yt_dlp import YoutubeDL
import moviepy.editor as mp
from moviepy.editor import VideoFileClip

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor,Wav2Vec2Config
import soundfile as sf
import os
from pydub import AudioSegment
import numpy as np
from scipy.signal import resample
import streamlit as st
from huggingface_hub import hf_hub_download
import streamlit as st

def extract_audio_from_video(url):
    print(f"Extracting audio from: {url}")
    video_path = "downloaded_video.mkv"
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': video_path,
        'merge_output_format': 'mkv',
        'quiet': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile("audio.mp3")
        return "audio.mp3"
    except Exception as e:
        print(f"Failed to extract audio: {e}")
        return None

def load_model(repo_id = "BoboThePotato/BobosAudioModel", filename = "accent_recognition_model_state_dict.pth", model_name = "facebook/wav2vec2-base", num_labels=23):
    state_dict_path = hf_hub_download(repo_id=repo_id, filename=filename)
    config = Wav2Vec2Config.from_pretrained(model_name)
    config.num_labels = num_labels
    config.problem_type = "single_label_classification"

    model = Wav2Vec2ForSequenceClassification(config)

    processor = Wav2Vec2Processor.from_pretrained(model_name)

    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.eval()

    return model, processor

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

def predict_accent(audio_path, model, processor, label_mapping, max_length=16000*5):
    try:
        speech_array, original_sr = sf.read(audio_path)
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)  

        target_sr = 16000  
        if original_sr != target_sr:
            num_samples = int(len(speech_array) * target_sr / original_sr)
            speech_array = resample(speech_array, num_samples)

        if len(speech_array) > max_length:
            speech_array = speech_array[:max_length]  
        else:
            speech_array = np.pad(speech_array, (0, max_length - len(speech_array))) 

        inputs = processor(speech_array, sampling_rate=target_sr, return_tensors="pt", padding=True)
        
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
    
        confidence = torch.softmax(logits, dim=-1)[0][predicted_id].item()
        predicted_label = label_mapping[predicted_id]

        return predicted_label, confidence
    except Exception as e:
        print(f"Error during audio prediction: {e}")
        return None, None

def main():
    st.title("Accent Detection from YouTube Video")

    url = st.text_input("Enter YouTube URL:")

    if url:
        model, processor = load_model()

        label_mapping = load_label_mapping()
        
        audio_path = extract_audio_from_video(url)
        if audio_path:
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
