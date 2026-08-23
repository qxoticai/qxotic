# Inflect speech CLI

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

A command-line interface for the `jinfer-inflect2` speech provider. It writes WAV files or streams
raw audio to `aplay` or `ffplay`. Playback dependencies remain in this example module, not the
reusable provider.

## Run

From the repository root:

```bash
mvn -pl jinfer/jinfer-inflect2-cli -am -DskipTests package

java \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-inflect2-cli/target/inflect.jar \
  models/inflect_nano_v2_q8_0.gguf \
  --text "Hello world." \
  --output hello.wav
```

Place `lexicon.bin` beside the model file to use the built-in phonemizer. Without it, the CLI uses
`espeak-ng`. Replace `--output` with `--play` to play audio while it is synthesized.

Run with `--help` for all options.

## Native executable

Build with GraalVM Native Image 25.0.3 or later:

```bash
mvn -pl jinfer/jinfer-inflect2-cli -am -Pnative -DskipTests package
```

The executable is `jinfer/jinfer-inflect2-cli/target/inflect`.
