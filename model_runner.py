import os
from accent_prediction_script import extract_audio_from_video, predict_accent, load_model, load_label_mapping

def predict_from_url(url):
    try:
        model_path = "accent_recognition_model_state_dict.pth"

        label_mapping = label_mapping = {
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
        num_labels = len(label_mapping)
        model = load_model(model_path, num_labels)

        audio_path = extract_audio_from_video(url)
        if not audio_path:
            return "Failed to extract audio from the video."

        predicted_label, confidence = predict_accent(audio_path, model, label_mapping)

        if predicted_label:
            return f"Predicted Accent: {predicted_label} (Confidence: {confidence:.2f})"
        else:
            return "Prediction failed."

    except Exception as e:
        return f"Error: {e}"
