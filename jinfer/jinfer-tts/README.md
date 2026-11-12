# Jinfer text-to-speech CLI

`jinfer-tts` runs every speech model included by `jinfer-models-all`. Models and companions can be
ordinary files or uncompressed entries in a ZIP overlay appended to the executable.

```bash
mvn -pl jinfer/jinfer-tts -am -DskipTests package   # from the repository root

java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-tts/target/jinfer-tts.jar \
  kokoro.gguf --with voice=af_heart.gguf --text "Hello." --output hello.wav

java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-tts/target/jinfer-tts.jar \
  inflect.gguf --with lexicon=lexicon.bin --play
```

`--play` synthesizes the full waveform before playback.
`--stream` starts after the first text chunk and synthesizes later chunks while earlier ones play.
Both modes use the same cross-platform playback support as `jinfer-cli`:

- **macOS:** built-in `afplay`, synthesizing one WAV clip ahead when streaming.
- **Windows:** built-in Windows PowerShell's `.NET SoundPlayer`, with the same clip streaming.
- **Linux:** install `aplay` (ALSA utilities) or `ffplay` (FFmpeg); streaming uses a persistent PCM pipe.

All platforms can fall back to `ffplay` if the native player cannot be launched.
Players must be on `PATH`; a player that launches but fails reports its error.

Build the native executable with GraalVM 25.0.4 or later:

```bash
make -C jinfer/jinfer-tts native   # -> bin/jinfer-tts
```

When the Inflect lexicon is in the models directory beside the checkout (or `MODELS=dir` names
another), it is bundled into the image, so an Inflect GGUF speaks without a file beside it and
without espeak-ng. Build the ZIP separately, append it, and adjust its offsets; updating the Mach-O
executable directly with `zip` is not reliable:

```bash
zip -0 payload.zip models/kokoro.gguf voices/af_heart.gguf
cp bin/jinfer-tts jinfer-tts-kokoro
dd if=payload.zip bs=1048576 >> jinfer-tts-kokoro
zip -A jinfer-tts-kokoro

./jinfer-tts-kokoro z://models/kokoro.gguf \
  --with voice=z://voices/af_heart.gguf --play
```

The model's data must start at a 4-byte aligned offset within the executable, because the weights are read in place.
ZIP does not align entries, so the packager must pad when necessary.
`jinfer-tts` refuses a model whose data offset is not 4-byte aligned, naming the offset.
Companion entries have no such requirement: they are extracted before loading.

Use `--archive <path>` when reading `z://` entries from another executable, including when running
the CLI through `java -jar`. Run `--help` for all options.
