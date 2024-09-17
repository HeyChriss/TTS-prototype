import torch
from TTS.api import TTS
import PyPDF2
from pydub import AudioSegment
import os
import re

def clean_text(text):
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r'\s([?.!,:](?:\s|$))', r'\1', text)
    # Ensure single space after punctuation
    text = re.sub(r'([?.!,:])\s*', r'\1 ', text)
    # Remove spaces at the beginning and end of the text
    text = text.strip()
    return text

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return clean_text(text)

def split_text_into_chunks(text, min_chunk_size=100, max_chunk_size=1000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        current_text = ' '.join(current_chunk)
        
        if len(current_text) >= max_chunk_size:
            chunks.append(clean_text(current_text))
            current_chunk = []
        elif len(current_text) >= min_chunk_size and (word.endswith('.') or word.endswith('!') or word.endswith('?')):
            chunks.append(clean_text(current_text))
            current_chunk = []

    if current_chunk:
        chunks.append(clean_text(' '.join(current_chunk)))

    return chunks

def generate_audio_for_chunks(tts, chunks, output_dir):
    audio_files = []
    for i, chunk in enumerate(chunks):
        output_file = f"{output_dir}/chunk_{i}.wav"
        try:
            tts.tts_to_file(text=chunk, speaker_wav="/cloning/Recording.wav", file_path=output_file)
            audio_files.append(output_file)
        except RuntimeError as e:
            print(f"Error processing chunk {i}: {str(e)}")
            print(f"Problematic chunk: {chunk}")
            continue
    return audio_files

def concatenate_audio_files(audio_files, output_file):
    combined = AudioSegment.empty()
    for audio_file in audio_files:
        sound = AudioSegment.from_wav(audio_file)
        combined += sound
    combined.export(output_file, format="wav")

# Main process
def create_audiobook(pdf_path, output_file):
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init TTS
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    # Generate audio for each chunk
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    audio_files = generate_audio_for_chunks(tts, chunks, temp_dir)

    # Concatenate audio files
    concatenate_audio_files(audio_files, output_file)

    # Clean up temporary files
    for file in audio_files:
        os.remove(file)
    os.rmdir(temp_dir)

# Usage
create_audiobook(r"C:\Users\USUARIO\Desktop\TTS\TTS-prototype\testing\testing.pdf", "audiobook_output.wav")