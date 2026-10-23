"""Stream a reply token by token, and stop early: closing the reply cancels the generation."""
import jinfer

with jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0") as model:
    with model.chat("Write a short poem about rivers.", max_tokens=200, temperature=0.7) as reply:
        for i, text in enumerate(reply):
            print(text, end="", flush=True)
            if i == 60:
                print("\n[enough, cancelling]")
                break
