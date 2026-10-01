# Tok'n'Roll Vector

The fast encoder of Tok'n'Roll's tiktoken family, written with the Java Vector API.
Same API, same token ids, several times the speed of core's own encoder.

## Maven

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-vector</artifactId>
  <version>0.2.0</version>
  <scope>runtime</scope>
</dependency>
```

Needs Java 25 and the incubator module on the command line:

```
java --add-modules jdk.incubator.vector ...
```

Nothing in the code changes.
`toknroll-core` finds the engine through `ServiceLoader` the first time a tokenizer of the family encodes, checks it against its own encoder on a sample that covers every kind of piece, and uses it from then on.
Without the module, or without the flag, core encodes by itself and produces the same ids.
To rule the engine out, say when chasing a suspected difference, set `-Dtoknroll.engine=none`.
A tokenizer's `toString()` names the engine once it has encoded through it.

## What it covers

The tiktoken regex family behind a byte-level BPE model: r50k (GPT-2), cl100k (GPT-4, Llama 3), Qwen2, o200k (GPT-4o, gpt-oss) and Tekken (Ministral).
Other tokenizers are untouched.

## How

- The text goes to UTF-8 with vectors: a vector of ASCII chars narrows to bytes in one step.
- The pre-tokenization split runs as a bitstream program, 64 bytes per step, after the design of Hugging Face's `bitcannon`: one bitmap per character class, every rule as boolean algebra and carry-propagating adds over the bitmaps, no branch per piece or per character.
- A word seen before is a probe into the thread's word cache, which grows with use up to a million words, and its ids are written straight into the output; a run of such words is a loop of its own.
- A new word is a token as it is when the merge loop would give it back whole, else merged: by hand up to three bytes, by a scan over flat arrays up to 95, by a heap beyond.

The split is proven exact against the regex on a fuzz corpus and piece for piece over all of enwik8, and the whole encoder against core's on the same.
