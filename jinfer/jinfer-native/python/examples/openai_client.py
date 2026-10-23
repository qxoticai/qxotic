"""Code written for the OpenAI SDK runs unchanged, in-process, with no server: swap the import."""
import jinfer

client = jinfer.OpenAI("LiquidAI/LFM2.5-350M-GGUF:Q8_0")   # instead of OpenAI(api_key=...)

response = client.chat.completions.create(
    messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name three planets."}],
    max_tokens=40, temperature=0)
print(response.choices[0].message.content)

for chunk in client.chat.completions.create(messages=[{"role": "user", "content": "Count from one to ten, separated by commas."}], max_tokens=40, stream=True):
    print(chunk.choices[0].delta.get("content", ""), end="", flush=True)
print()
