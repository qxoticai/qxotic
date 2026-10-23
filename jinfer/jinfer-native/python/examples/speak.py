"""Text to speech: PCM chunks stream out as they are synthesized; `.save` writes a WAV."""
import jinfer

with jinfer.load("simonfxr/kokoro.cpp-GGUF:Q8_0",
                  companions={"voice": "simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf"}) as tts:
    speech = tts.speak("Turns out, matrix multiplications can talk.")
    for chunk in speech:                                   # play as it comes, if you like
        print(f"{len(chunk) / speech.sample_rate:.2f} s chunk")
    speech.save("speech.wav")
    print(f"speech.wav: {len(speech.audio) / speech.sample_rate:.1f} s at {speech.sample_rate} Hz")
