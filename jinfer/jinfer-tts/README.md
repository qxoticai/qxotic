# Jinfer text-to-speech CLI

`jinfer-tts` runs every speech model included by `jinfer-models-all`. Models and companions can be
ordinary files or uncompressed entries in a ZIP overlay appended to the executable.

```bash
mvn -pl jinfer-tts -am -DskipTests package

java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -jar jinfer-tts/target/jinfer-tts.jar \
  kokoro.gguf --with voice=af_heart.gguf --text "Hello." --output hello.wav

java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -jar jinfer-tts/target/jinfer-tts.jar \
  inflect.gguf --with lexicon=lexicon.bin --play
```

`--play` synthesizes the full waveform before playback. `--stream` starts after the first text chunk
and synthesizes later chunks while earlier ones play. It uses an ordered `afplay` queue on macOS and
a persistent raw PCM stream through `ffplay` or `aplay` elsewhere.

Build the native executable with GraalVM 25 or later:

```bash
mvn -pl jinfer-tts -am -Pnative -DskipTests package
```

The executable is `target/jinfer-tts`. Build the ZIP separately, append it, and adjust its offsets;
updating the Mach-O executable directly with `zip` is not reliable:

```bash
zip -0 payload.zip models/kokoro.gguf voices/af_heart.gguf
cp target/jinfer-tts jinfer-tts-kokoro
dd if=payload.zip bs=1048576 >> jinfer-tts-kokoro
zip -A jinfer-tts-kokoro

./jinfer-tts-kokoro z://models/kokoro.gguf \
  --with voice=z://voices/af_heart.gguf --play
```

The model entry's data offset must be divisible by 4. ZIP does not guarantee this alignment, so
the packager must add padding when necessary; `jinfer-tts` rejects an unaligned model instead of
running inference over misaligned weights. Companion entries do not have this restriction because
they are extracted before loading.

Use `--archive <path>` when reading `z://` entries from another executable, including when running
the CLI through `java -jar`. Run `--help` for all options.
