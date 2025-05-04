import yt_dlp
import moviepy.editor as mp
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import numpy as np
import requests
import soundfile as sf
import os
from io import BytesIO
import pickle
from pydub import AudioSegment
import soundfile as sf
from scipy.signal import resample

def download_video(url, output_path="downloaded_video"):
    ydl_opts = {
        'outtmpl': output_path,  
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])  
    video_file = f"{output_path}.mkv"  
    return video_file

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


def load_model(model_path, num_labels):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_label_mapping(path):
    label_mapping = {
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


    return label_mapping

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

def predict_accent(audio_path, model, label_mapping):
    try:
        speech_array, original_sr = sf.read(audio_path)
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)

        target_sr = 16000
        if original_sr != target_sr:
            num_samples = int(len(speech_array) * target_sr / original_sr)
            speech_array = resample(speech_array, num_samples)

        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
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

def main(url):
    LABEL_MAPPING_PATH = 'D:/Accent Detection/label_mapping.pkl'
    MODEL_PATH = 'D:/Accent Detection/accent_recognition_model_state_dict.pth'

    label_mapping = load_label_mapping(LABEL_MAPPING_PATH)
    num_labels = len(label_mapping)
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = load_model(MODEL_PATH, num_labels)

    audio_path = extract_audio_from_video(url)  
    predicted_label, confidence = predict_accent(audio_path, model, label_mapping)
    
    if predicted_label is not None:
        print(f"Predicted Accent: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")
    else:
        print("Accent prediction failed.")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Accent Detection from YouTube video URL")
    parser.add_argument("--url", type=str, required=True, help="URL of the YouTube video")
    args = parser.parse_args()

    main(args.url)
