# Gemma 4 multimodal examples

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)

These single-file Java programs run tasks from Google's Gemma guides for
[images](https://ai.google.dev/gemma/docs/capabilities/vision/image) and
[video](https://ai.google.dev/gemma/docs/capabilities/vision/video) with Jinfer. They cover image
questions, multiple images, OCR, object detection, video understanding and variable image token
budgets.

## Run

Install [JBang](https://www.jbang.dev/), then run the scripts from a repository checkout:

```bash
cd jinfer/examples
```

Each Java file declares Java 25, its dependencies and its default model references. No shared source
files are required. Models download on first use and remain in the Jinfer cache.

| Task | JBang command |
|---|---|
| Single image Q&A | `jbang GemmaVision.java img.png "What is shown in this image?"` |
| Multiple images | `jbang GemmaVisionMulti.java "Caption these images." a.png b.jpg` |
| OCR | `jbang GemmaVision.java sign.png "What does the sign say?"` |
| Object detection | `jbang GemmaVision.java street.jpg "detect person and cat, output only json"` |
| Token-budget comparison | `./gemma-budget-sweep.sh city.jpg "detect person and car, output only json"` |
| Video understanding | `jbang GemmaVideo.java clip.mp4 "Summarize the main events."` |

Description, OCR and detection use the same script and differ only by the prompt. Detection returns
normalized 0-1000 box coordinates as JSON. For an annotated PNG instead, run
[`Detect.java`](scripts/Detect.java):

```bash
cd scripts
jbang Detect.java ../street.jpg "person, bicycle, traffic light"
```

`GemmaVideo.java` requires `ffmpeg` on `PATH` and samples 16 frames by default. Override the count
with `-Djinfer.video.frames=<count>`.

`GemmaVision.java` and `GemmaVideo.java` accept model and media references as trailing arguments.
E variants use the E2B projector:

```bash
# E2B (default)
jbang GemmaVision.java cat.jpg "Describe it"

# E4B
jbang GemmaVision.java cat.jpg "Describe it" \
  unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
  unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf

# 12B
jbang GemmaVision.java cat.jpg "Describe it" \
  unsloth/gemma-4-12b-it-GGUF:Q8_0 \
  unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf
```

Set the image token budget to trade speed for visual detail. Supported values are 70, 140, 280,
560 and 1120:

```bash
jbang -Djinfer.gemma4.imageTokenBudget=1120 GemmaVision.java chart.png "Read every value"
```
