
# Multimodal RAG – Image + Audio + Language Intelligence

This project combines **OpenAI Whisper**, **LLaVA (LLaMA Visual Language Model)**, and **Gradio** to create a **multimodal assistant** capable of:

- Transcribing spoken questions (via microphone)
- Analyzing and describing uploaded images
- Generating **spoken audio responses** using **text-to-speech**

It’s an interactive pipeline where voice meets vision.

---

##  Example Use Case

You speak:  
> "What color is the object in the image?"

You upload an image.

The assistant:
- Transcribes your question,
- Describes the image based on your prompt,
- Speaks the answer back.

---

## Project Structure

```
multimodal-rag/
├── multimodal_rag.py         # Main Gradio interface
├── img_to_text.py            # LLaVA-based image description logic
├── audio_utils.py            # Audio transcription (Whisper) + TTS
├── logger.py                 # Logging utility
├── requirements.txt          # All Python dependencies
└── README.md                 # This file
```

---

##  Installation

### Create and Activate a Virtual Environment 
```bash
python -m venv venv
venv\Scripts\activate       
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the App

run:

```bash
python multimodal_rag.py
```

Gradio will start a local server, and you’ll see a link like:

```
Running on local URL:  http://127.0.0.1:7860
```

Open the link in your browser.

---


## Requirements

> All handled by `requirements.txt`, but here’s what’s included:

- `transformers==4.37.2`
- `bitsandbytes==0.41.3`
- `accelerate==0.25.0`
- `torch` (with CUDA support recommended)
- `git+https://github.com/openai/whisper.git`
- `gradio`
- `gTTS`
- `nltk`
- `Pillow`
- `numpy`

---

## Notes
- First model load may take a while.
- LLaVA model used: `llava-hf/llava-1.5-7b-hf`
- Whisper model used: `medium`

---
