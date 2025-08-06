import whisper
import numpy as np
from gtts import gTTS

model = whisper.load_model("medium")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe(audio_path):
    if not audio_path:
        return ''
    
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return result.text

def text_to_speech(text, file_path="Temp3.mp3", language='en'):
    audioobj = gTTS(text=text, lang=language, slow=False)
    audioobj.save(file_path)
    return file_path
