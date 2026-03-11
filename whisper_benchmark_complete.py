import argparse
import json
import sounddevice as sd
import numpy as np
import whisper

class WhisperModel:
    def __init__(self, model_name):
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_data):
        result = self.model.transcribe(audio_data)
        return result['text']

def record_audio(duration, sample_rate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)  # Single channel for mono
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data.flatten()

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)

def main():
    parser = argparse.ArgumentParser(description="Whisper Model CLI")
    parser.add_argument('--model', type=str, required=True, help="Model to use for transcription")
    parser.add_argument('--duration', type=int, default=5, help="Duration of recording in seconds")
    parser.add_argument('--output', type=str, default='output.json', help="Output JSON file")
    args = parser.parse_args()

    audio_data = record_audio(args.duration)
    model = WhisperModel(args.model)
    transcription = model.transcribe_audio(audio_data)

    result = {"transcription": transcription}
    save_to_json(result, args.output)
    print(f'Transcription saved to {args.output}')

if __name__ == '__main__':
    main()