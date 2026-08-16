# jinfer-inflect2-cli

Small executable example for `jinfer-inflect2`. It writes a WAV file or streams raw audio to
`aplay`/`ffplay`; the reusable model module contains none of these CLI dependencies.

## Run

```bash
mvnd -pl jinfer/jinfer-inflect2-cli -am -DskipTests package

java \
  --enable-preview \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-inflect2-cli/target/inflect.jar \
  /path/to/inflect_nano_v2_q8_0.gguf \
  --text "Hello world." \
  --output hello.wav
```

Place `lexicon.bin` beside the GGUF for the fast phonemizer. Without one, the CLI falls back to
`espeak-ng`. Use `--play` instead of `--output` to play as synthesis progresses.

Run with `--help` for all options.

## Native executable

```bash
mvnd -pl jinfer/jinfer-inflect2-cli -am -Pnative -DskipTests package
```

The executable is `jinfer/jinfer-inflect2-cli/target/inflect`.
