# jinfer-gemma-embedding2

[Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) for the JVM: text, images, audio and video in one 768-dimensional space, Matryoshka down to 128.
GGUF weights: [unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF).

The LangChain4j and Spring AI integrations use this model for search: their query and document roles.
This module adds the model's other tasks, for code that calls jinfer directly.

## Task prompts

EmbeddingGemma 2 is trained with a short prompt in front of the text, one per task.
`GemmaEmbedding2` builds them: each method takes your text and returns it framed, ready to embed.

| Task | Query | What it is compared with |
|---|---|---|
| Search | `searchQuery(text)` | `document(title, text)` |
| Question answering | `question(text)` | `document(title, text)` |
| Fact checking | `claim(text)` | `document(title, text)` |
| Code retrieval | `codeQuery(text)` | `document(filename, code)` |
| Classification | `classification(text)` | `classification(...)` on the labels or examples |
| Clustering | `clustering(text)` | `clustering(...)` on every other text |
| Sentence similarity | `similarity(text)` | `similarity(...)` on the other side |

- The four retrieval tasks share one document framing, so one index serves all of them; a null title is `none`, as the card says.
- Images, audio and video take no prompt: embed them as they are, whatever the task.
- Framed text goes only to `LoadedEmbedder.embed` and `embedAll`, which embed text as given; the integrations add their own search prompt.

The strings are the card's, from `config_sentence_transformers.json`; `GemmaEmbedding2ParityTest` holds every method to the official pipeline's tokens and vectors.

## Examples

These are the bodies of `GemmaEmbedding2ExamplesTest`, which runs them against real images, audio and video.

```java
import static com.qxotic.jinfer.models.gemmaembedding2.GemmaEmbedding2.*;

ModelStore hub = ModelStore.standard();
Arena weights = Arena.ofShared();
LoadedEmbedder<?> embedder = Models.loadEmbedder(
        hub.resolve("unsloth/embeddinggemma-2-GGUF:Q8_0"), weights,
        Map.of("media", hub.resolve("unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf")), null);
ContextState state = embedder.model().newState(8192);   // shared by text and media

/** One vector for one input: Strings and media, in order. */
float[] embed(Object... parts) {
    float[][] v = new float[1][];
    embedder.embed(state, List.of(parts), embedder.dimension(), x -> v[0] = x);
    return v[0];
}

float score(float[] a, float[] b) {     // vectors are L2-normalized: dot = cosine
    float s = 0;
    for (int i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}
```

Search, one text query over text, a photo, a recording and a video:

```java
List<float[]> index = List.of(
        embed(document("Trip notes", "We hiked to the glacier at sunrise.")),
        embed(ImageCodec.load(Path.of("cat.png"))),
        embed(AudioCodec.load(Path.of("speech.wav"))),   // "Hello world one two three four five"
        embed(VideoCodec.ffmpeg().every(Path.of("dice.mp4"), Duration.ofSeconds(1), 32)));

float[] q = embed(searchQuery("a person counting from one to five"));   // best: the recording
```

Question answering over page scans:

```java
List<float[]> pages = scans.stream().map(p -> embed(ImageCodec.load(p))).toList();
float[] q = embed(question("What was the revenue in the third quarter?"));   // best: the report page
```

Fact checking against photos and reports; it finds evidence, it does not judge it:

```java
List<float[]> evidence = List.of(
        embed(ImageCodec.load(Path.of("cat.png"))),
        embed(document("City report", "The bridge reopened in March after repairs.")));
float[] q = embed(claim("The bridge was closed all winter."));   // best: the report
```

Media as the query, against text:

```java
float[] photo = embed(ImageCodec.load(Path.of("cat.png")));
List<float[]> entries = List.of(
        embed(document("Cat", "A small domesticated feline with soft fur.")),
        embed(document("Dice", "Small cubes with numbered faces, used in games.")));   // best: "Cat"

float[] recording = embed(AudioCodec.load(Path.of("speech.wav")));
List<float[]> transcripts = List.of(
        embed(document(null, "The weather will be sunny tomorrow.")),
        embed(document(null, "Hello world, one two three four five.")));   // best: the second
```

Classification of images against labels written as text:

```java
List<float[]> labels = List.of(
        embed(classification("a photo of a cat")),
        embed(classification("a photo of dice")));
float[] x = embed(ImageCodec.load(Path.of("dice.png")));   // best: "a photo of dice"
```

Clustering captions and photos together: each photo lands next to its own caption.

```java
List<float[]> captions = List.of(
        embed(clustering("A cat sitting and looking at the camera.")),
        embed(clustering("A pair of dice on a table.")));
float[] photo = embed(ImageCodec.load(Path.of("cat.png")));   // nearest: the cat caption
```

Sentence similarity:

```java
float[] guitar = embed(similarity("A man is playing a guitar."));
float[] strums = embed(similarity("Someone strums an instrument."));
float[] stocks = embed(similarity("The stock market fell today."));
// score(guitar, strums) > score(guitar, stocks)
```

Text and media in one input make one vector; as in the card's example, it takes no prompt:

```java
float[] listing = embed("Our pet of the month: ", ImageCodec.load(Path.of("cat.png")), " Ready for adoption.");
float[] q = embed(searchQuery("a cat to adopt"));
```

## Notes

- Every input shares one 8,192-token window: an image takes 280 tokens, a video frame 140, a second of audio 25.
- For many texts, `embedder.embedAll(state, texts, consumer)` packs them into fewer forward passes than one `embed` each.
- Pass a width such as 256 instead of `embedder.dimension()` for shorter Matryoshka vectors; the card calls 256 close to lossless, and 128 costs media noticeably.
