"""Corner cases a library has to get right, run against the cached LFM2.5 models: `python qa.py`.
Threads, lifetime at interpreter exit, memory under a soak, offline mode, tool calls, unicode."""
import json, os, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jinfer

LM, EMB = "LiquidAI/LFM2.5-350M-GGUF:Q8_0", "LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0"
checks = 0
def ok(cond, what):
    global checks
    assert cond, what
    checks += 1
    print(f"  ok {checks}: {what[:70]}", flush=True)
def rss_mb():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024

jinfer.configure("-Djinfer.threads=4", "-Djinfer.offline=true")  # never download in QA
m = jinfer.load(LM)

# two threads on one model: the engine serializes, both answer
results, errors = {}, []
def worker(name, q):
    try:
        results[name] = m.chat(q, max_tokens=12, temperature=0).text
    except Exception as e:
        errors.append(e)
ts = [threading.Thread(target=worker, args=(n, q)) for n, q in [("a", "What is the capital of France? One word."), ("b", "What is 2+2? Digits only.")]]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
ok(not errors and "Paris" in results["a"] and "4" in results["b"], f"concurrent completes: {results} {errors}")

# three threads, the same greedy question: every answer is the same
def same(name):
    try:
        results[name] = m.chat("What is the capital of France? One word.", max_tokens=8, temperature=0).text
    except Exception as e:
        errors.append(e)
ts = [threading.Thread(target=same, args=(f"c{i}",)) for i in range(3)]
[t.start() for t in ts]; [t.join() for t in ts]
ok(not errors and len({results["c0"], results["c1"], results["c2"]}) == 1 and "Paris" in results["c0"], f"concurrent greedy replies agree: {errors}")

# two models at once, free one, the other keeps working
e = jinfer.load(EMB)
ok(len(e.embed(["x"])[0]) == e.dimension, "second model works")
e.close()
after = m.chat("What is the capital of France? Answer in one word.", max_tokens=12, temperature=0).text
ok("Paris" in after, f"first model after freeing the second: {after!r}")

# a closed handle is refused, not dereferenced
e2 = jinfer.load(EMB); e2.close()
try:
    e2.embed(["x"]); ok(False, "closed model must be refused")
except jinfer.InvalidArgument as err:
    ok(err.code == -2, f"closed model: {err}")

# offline: an uncached model is refused without the network
t0 = time.time()
try:
    jinfer.load("qxotic/never-downloaded-GGUF:Q8_0"); ok(False, "offline must refuse an uncached model")
except jinfer.JinferError as err:
    ok(time.time() - t0 < 5 and err.code in (-1, -3), f"offline refusal in {time.time() - t0:.1f}s: {type(err).__name__}: {err}")

# unicode through the tokenizer and the model
text = "naïve café 日本語 🙂 - fin"
ok(len(m.tokenizer.encode(text)) >= 6, "unicode tokenizes")
echo = m.chat(f"Repeat exactly: {text}", max_tokens=32, temperature=0).text
ok("日本語" in echo or "café" in echo, f"unicode through the model: {echo!r}")
ok(len(m.tokenizer.encode("word " * 4000)) > 1000, "long text tokenizes")

# multi-turn history
history = [{"role": "user", "content": "My name is Zoltan."}, {"role": "assistant", "content": "Nice to meet you, Zoltan."},
           {"role": "user", "content": "What is my name? One word."}]
single = m.chat(history[-1:], max_tokens=8, temperature=0)
multi = m.chat(history, max_tokens=12, temperature=0)
ok(multi.usage.prompt_tokens > single.usage.prompt_tokens and multi.text.strip(), f"multi-turn history reaches the model: {multi.text!r}")
print("multi-turn recall (model quality, not asserted):", multi.text.strip()[:60])

# tool calling: a forced call comes back as a parsed tool_calls entry
tools = [{"type": "function", "function": {"name": "weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = m.chat("What's the weather in Zurich?", max_tokens=64, temperature=0, tools=tools,
               tool_choice={"type": "function", "function": {"name": "weather"}})
calls = r.tool_calls
ok(r.finish_reason == "tool_calls" and len(calls) == 1 and calls[0].name == "weather", f"forced tool call: {calls}")
ok(isinstance(calls[0].arguments, dict) and "Zurich" in json.dumps(calls[0].arguments), f"tool arguments {calls[0].arguments}")
r = m.chat("What's the weather in Zurich?", max_tokens=64, temperature=0, tools=tools)
ok(r.finish_reason in ("tool_calls", "stop", "length"), f"auto tool choice shape: {r.finish_reason}")

# cancel during a long prefill returns promptly
s = m.chat("Summarize: " + "The quick brown fox jumps over the lazy dog. " * 300, max_tokens=50)
t0 = time.time(); s.close(); ok(time.time() - t0 < 10, f"cancel during prefill in {time.time() - t0:.1f}s")

# soak: memory stays flat over a hundred completions
for _ in range(5): m.chat("Hi", max_tokens=4).text
before = rss_mb()
for i in range(100):
    m.chat(f"Say the number {i}.", max_tokens=6, temperature=0).text
after = rss_mb()
ok(after - before < 64, f"soak RSS {before:.0f} -> {after:.0f} MB")
m.close()

# the interpreter exits cleanly with a live model and a running stream never closed
code = '''
import sys; sys.path.insert(0, %r)
import jinfer
m = jinfer.load(%r)
s = m.chat("Write a long story about the sea.", max_tokens=400)
next(iter(s))
print("exiting with a live stream")
''' % (os.path.dirname(os.path.abspath(__file__)), LM)
t0 = time.time()
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
ok(p.returncode == 0 and "exiting" in p.stdout, f"exit with live stream: rc={p.returncode} in {time.time() - t0:.1f}s {p.stderr[-300:]}")
ok(time.time() - t0 < 60, f"exit took {time.time() - t0:.1f}s")

print(f"ok: {checks} checks")
