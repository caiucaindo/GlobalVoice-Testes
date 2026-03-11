import torch
import whisper
import time
import json

# Initialize Whisper model
model = whisper.load_model('base')

# Function for benchmarking

def benchmark_whisper(input_file):
    start_time = time.time()
    result = model.transcribe(input_file)
    duration = time.time() - start_time
    return result, duration

# Function for chunk processing

def process_chunks(input_file, chunk_size=5):
    audio = whisper.load_audio(input_file)
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
    return chunks

# Main function

def main(input_file):
    # Process audio in chunks
    chunks = process_chunks(input_file)
    benchmarks = []

    for i, chunk in enumerate(chunks):
        result, duration = benchmark_whisper(chunk)
        benchmarks.append({'chunk': i, 'duration': duration, 'result': result})

    # Exporting results to JSON
    with open('benchmark_results.json', 'w') as f:
        json.dump(benchmarks, f, indent=4)

if __name__ == '__main__':
    input_audio_file = 'your_audio_file.mp3'  # Replace with your input audio file
    main(input_audio_file)
