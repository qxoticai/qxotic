# Tok'n'Roll Safetensors

`toknroll-hf` is an optional module that builds tokenizers from local HuggingFace
tokenizer files.

## What it does

- Loads `tokenizer.json` (plus optional `special_tokens_map.json` and `tokenizer_config.json`)
- Builds a Tok'n'Roll tokenizer from local files
- Keeps model files and remote downloading out of core `toknroll`

## Current scope

- Supports BPE `tokenizer.json`
- Fails fast on unsupported `normalizer` or `pre_tokenizer` definitions

## Entry points

- `SafetensorsTokenizers.fromDirectory(Path directory)`
- `SafetensorsTokenizers.fromTokenizerJson(Path tokenizerJson)`
