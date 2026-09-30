# Tok'n'Roll Vector API Kernels

The hot loops of Tok'n'Roll's encoder written with the Java Vector API.
Same API, same token ids, several times the speed on the tiktoken family of tokenizers.

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
`toknroll-core` finds the kernels through `ServiceLoader`, checks them against its own versions on a sample when they load, and uses them from then on.
Without the module, or without the flag, core runs its plain Java versions and produces the same ids.

## What it speeds up

- The conversion of a `String` to UTF-8, which every encode starts with.
- The pre-tokenization split of the tiktoken regex family: r50k (GPT-2), cl100k (GPT-4, Llama 3), Qwen2, o200k (GPT-4o, gpt-oss) and Tekken (Ministral).
  Each regex runs as a bitstream program, 64 bytes per step, after the design of Hugging Face's `bitcannon`: one bitmap per character class, every rule as boolean algebra and carry-propagating adds over the bitmaps, no branch per piece or per character.

The split is proven exact against the regex the same way core's own scanner is: on a fuzz corpus of every kind of character the regexes tell apart, and piece for piece over all of enwik8.
