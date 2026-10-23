"""Speech to text from a WAV file: `python transcribe.py speech.wav`. Any sample rate; the library resamples."""
import sys, jinfer

path = sys.argv[1] if len(sys.argv) > 1 else "speech.wav"
with jinfer.load("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf") as asr:
    transcript = asr.transcribe(path)
    print(transcript.text)
    for word in transcript.words[:8]:
        print(f"  {word.start:6.2f}-{word.end:6.2f}  {word.text}")
