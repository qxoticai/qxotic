"""Caches every model the native suites load (toknroll's and jinfer's C tests, test_jinfer.py), through the
libraries themselves: run once per build machine, online, with a python that has toknroll and jinfer installed
(a check-wheel's target/wheelenv, or pip). Kokoro needs espeak-ng on the PATH to load, so the machine needs it
for the speech checks too.

The suites run offline and skip the checks of a model not here; a skip fails the run unless CTEST_ALLOW_SKIP=1."""
import jinfer, toknroll

for ref in ("Qwen/Qwen3-0.6B", "LiquidAI/LFM2.5-350M-GGUF:Q8_0", "microsoft/phi-2", "deepseek-ai/DeepSeek-V3",
            "moonshotai/Kimi-K2.6"):
    toknroll.load(ref).close()
    print("cached", ref)
for ref, companions in (
        ("LiquidAI/LFM2.5-350M-GGUF:Q8_0", None),
        ("LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0", None),
        ("LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0", {"media": "LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"}),
        ("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0", None),
        ("simonfxr/kokoro.cpp-GGUF:Q8_0", {"voice": "simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf"}),
        ("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf", None)):
    jinfer.load(ref, companions=companions).close()
    print("cached", ref)
