import nltk
nltk.download('punkt')
from nltk import sent_tokenize

import torch
from transformers import BitsAndBytesConfig, pipeline
from img_to_text import img2txt
from audio_utils import transcribe, text_to_speech
import gradio as gr
import os

# Whisper and Torch setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

# Model and pipeline setup
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quant_config})

# Fallback silent mp3
os.system("ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 10 -q:a 9 -acodec libmp3lame Temp.mp3")

# Gradio logic
def process_inputs(audio_path, image_path):
    speech_to_text_output = transcribe(audio_path)
    chatgpt_output = img2txt(pipe, speech_to_text_output, image_path) if image_path else "No image provided."
    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")
    return speech_to_text_output, chatgpt_output, processed_audio_path

iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="ChatGPT Output"),
        gr.Audio("Temp.mp3")
    ],
    title="Learn OpenAI Whisper: Image processing with Whisper and Llava",
    description="Upload an image and interact via voice input and audio response."
)

iface.launch(debug=True)
