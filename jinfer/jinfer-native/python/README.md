# jinfer

**Local AI for Python, in three lines.**
Chat, vision, audio, embeddings, reranking, speech and transcription from one wheel, in your own process.

[![PyPI](https://img.shields.io/pypi/v/jinfer)](https://pypi.org/project/jinfer/)
[![Python](https://img.shields.io/pypi/pyversions/jinfer)](https://pypi.org/project/jinfer/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/qxoticai/qxotic/blob/v0.4.0/LICENSE)

```sh
pip install jinfer
```

```python
import jinfer

model = jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0")   # downloaded once, then cached
print(model.chat("Why is the sky blue?").text)

for text in model.chat("Tell me a joke."):               # streamed
    print(text, end="", flush=True)

from dataclasses import dataclass

@dataclass
class Person:
    name: str
    born: int

model.chat("Ada Lovelace, born 1815.", schema=Person).parsed
# Person(name='Ada Lovelace', born=1815)
```

- **Simple.** `load`, `chat`, `.text`: replies are typed objects, a schema comes back as your dataclass, and tools are plain Python functions.
- **Fast.** Reads prompts faster than llama.cpp on the same CPU, see the [benchmarks](https://qxotic.ai/#benchmarks), and starts at once: the engine is compiled ahead of time.
- **Light.** One 9 MB wheel and one dependency: no server, no `torch`, no GPU, no JVM.

## What it runs

GGUF models, `Q8_0` recommended, of these families:

| Capability | Families |
|---|---|
| chat | Gemma 4, Qwen 3.5+, LFM 2.5, LFM3, gpt-oss, Llama 3+, Granite 4.1+, Ministral 3, SmolLM3, MiniCPM5, Laguna, Mellum 2, Ling 3, Nemotron-H, Ternary Bonsai 2 |
| vision | Gemma 4, Qwen 3.5+, LFM 2.5, LFM3, Ternary Bonsai 2 |
| audio input | Gemma 4, LFM3-Omni |
| embeddings | EmbeddingGemma 2 (text, images, audio, video), Qwen 3, LFM 2.5 |
| reranking | Qwen 3, LFM 2.5 |
| speech | Kokoro, Inflect |
| transcription | Parakeet, with timed words |

## Requirements

- Python 3.10 or later.
- Linux x86-64 or aarch64 with glibc 2.28 or later, macOS 14 or later on Apple silicon, or Windows 10 or later on x86-64.
- On x86-64, a CPU with AVX2.
- On the PATH: `curl` to download models, `ffmpeg` for images and audio in a chat, `espeak-ng` (or `espeak`) for Kokoro voices.

## The API at a glance

`jinfer.load(ref)` takes a Hub reference (`owner/repo:QUANT`) or a local path, and returns the class the model calls for:

| Class | Call |
|---|---|
| `LanguageModel` | `chat(prompt, schema=..., tools=[...])`, `conversation(system=...)` |
| `Embedder` | `embed(texts)`: a float32 vector per text |
| `Reranker` | `rerank(query, documents)`: a score per document |
| `Speaker` | `speak(text)`: PCM, or `.save("out.wav")` |
| `Transcriber` | `transcribe("speech.wav")`: `.text` and timed `.words` |

The [full API reference](https://github.com/qxoticai/qxotic/blob/v0.4.0/jinfer/jinfer-native/python/API.md) covers every option, conversations, errors, threads and the C library.
The [comparison](https://github.com/qxoticai/qxotic/blob/v0.4.0/docs/jinfer/python-comparison.md) sets jinfer beside ollama, transformers and llama-cpp-python, use case by use case.

## License

Apache 2.0, by [Quixotic AI](https://qxotic.ai).
The library in the wheel is built with Oracle GraalVM and carries its runtime under the [GraalVM Free Terms and Conditions](https://www.oracle.com/downloads/licenses/graal-free-license.html): free to use, commercially too, but a product sold for a fee that bundles it is not covered.
