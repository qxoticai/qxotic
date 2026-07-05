# Gemma 4 vision examples (jbang)

Runnable equivalents of every snippet on
<https://ai.google.dev/gemma/docs/capabilities/vision/image>, using jinfer instead of
`transformers`. One-time setup publishes jinfer to your local Maven repo so jbang can resolve it:

```bash
cd .. && ./mvnw -q -DskipTests install     # installs com.qxotic:jinfer-gemma4 to ~/.m2
```

Pass `-Djam.native.library.path=/path/to/libjam.so` for native-speed matmul (otherwise the Java
Vector backend is used automatically).

| Google snippet | jbang equivalent |
|---|---|
| 1. Single image Q&A ("What is shown in this image?") | `jbang GemmaVision.java img.png "What is shown in this image?"` |
| 2. Multiple images ("Caption these images.") | `jbang GemmaVisionMulti.java "Caption these images." a.png b.jpg` |
| 3. OCR ("What does the sign say?") | `jbang GemmaVision.java sign.png "What does the sign say?"` |
| 4. Object detection ("detect person and cat") | `jbang GemmaVision.java street.jpg "detect person and cat, output only json"` |
| 5-6. Token-budget comparison (70/140/280/560) | `./gemma-budget-sweep.sh city.jpg "detect person and car, output only json"` |

1, 3 and 4 are the same script — Gemma's vision is prompt-driven, so describe / OCR / detect differ
only by the prompt (detection returns normalized 0-1024 box coordinates as JSON).

Model sizes (any Gemma 4 works — only the paths change; E-variants share the E2B projector):

```bash
# E2B (default, fastest)   E4B                                  12B (sharpest)
jbang GemmaVision.java cat.jpg "Describe it"
jbang GemmaVision.java cat.jpg "Describe it" ~/m/gemma-4-E4B-it-Q8_0.gguf ~/m/gemma-4-E2B-it-GGUF/mmproj-F32.gguf
jbang GemmaVision.java cat.jpg "Describe it" ~/m/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf ~/m/gemma-4-12b-it-GGUF/mmproj-F32.gguf
```

Image token budget (trade detail for speed, per the docs' 70/140/280/560/1120):

```bash
jbang -Djinfer.gemma4.imageTokenBudget=1120 GemmaVision.java chart.png "Read every value"
```
