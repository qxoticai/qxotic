"""Ask a question, get the answer. `python chat.py "your question"`"""
import sys, jinfer

question = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France? One sentence."
with jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0") as model:
    reply = model.chat(question, max_tokens=128, temperature=0)
    print(reply.text)
    u = reply.usage
    print(f"[{u.completion_tokens} tokens, {u.completion_tokens / u.completion_ms * 1000:.0f} tok/s, first token in {u.prompt_ms:.0f} ms]")
