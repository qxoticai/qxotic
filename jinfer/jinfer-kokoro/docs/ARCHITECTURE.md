# jinfer-kokoro — class architecture

Package `com.qxotic.jinfer.models.kokoro` (12 classes, ~3030 lines). It is the `jinfer-<model>` provider for GGUF
architecture `kokoro` — a **speech-synthesis-only** family: hexgrad's Kokoro v1.0 (StyleTTS2-style PL-BERT +
prosody predictor + iSTFTNet vocoder) driven by one separate **voice GGUF** (`general.architecture = kokoro-voice`).
Text goes through espeak (required, from `jinfer-codecs`) rewritten into misaki's IPA dialect, then through the
model to 24 kHz mono PCM. Everything runs on the CPU kernels of `jinfer-kernels` plus host-side Java DSP.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
`*--` composition (nested weight records); classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b `Kokoro` and its sub-networks, 2c reusable layers
3. [Loading sequence](#3-loading-sequence)
4. [Text front end: `speak`](#4-text-front-end-speak)
5. [Synthesis pipeline: `Kokoro.synthesize`](#5-synthesis-pipeline)
6. [Sub-network forwards](#6-sub-network-forwards) — 6a text encoder, 6b PL-BERT, 6c prosody, 6d decoder, 6e generator + DSP
7. [State, memory and validation](#7-state-memory-and-validation)

| Class | Role |
|---|---|
| `KokoroProvider` | `ModelProvider` SPI: architecture `kokoro`, declares companion `voice`, `loadSpeech` → `KokoroTTS.load` |
| `KokoroTTS` | `SpeechSynthesisModel<Configuration, Weights, State>`: binds espeak + `Misaki` into a `Phonemizer`; text chunking (`speak`) and `synthesize` → `Media.Audio` |
| `Kokoro` | The model: reads/validates GGUF config + voice pack, owns all sub-network weights, runs the whole pipeline in `synthesize`; nested `Configuration`, `Voice`, `Weights`, `State` |
| `Misaki` | `UnaryOperator<String>`: rewrites tied espeak IPA into misaki's dialect (English US/GB tables, "other" table) |
| `TextEncoder` | Token embedding + 3× (conv1d, LayerNorm, LeakyReLU) + bidirectional LSTM → `[steps, 512]` |
| `PlBert` | ALBERT-style parameter-shared PL-BERT (12 repeats of one layer) + `bert_proj` → `[steps, 512]` |
| `ProsodyPredictor` | Duration encoder (3× BiLSTM + AdaLayerNorm), duration head, alignment, shared LSTM, F0 / noise branches |
| `KokoroDecoder` | Pre-generator decoder body: ASR gather, F0/N down-convs, encode block + 4 decode `AdainResBlock`s |
| `KokoroGenerator` | iSTFTNet generator: 2 upsample stages with harmonic-source injection, `AdaINResBlock1` stacks, `conv_post` → spectrum |
| `KokoroDsp` | Host DSP: nine-harmonic HnNSF source + STFT (`sourceStft`), inverse STFT (`istft`); n_fft 20, hop 5 |
| `KokoroLayers` | Reusable layer records (`Linear`, `Conv1d`, `ConvTranspose1d`, `DepthwiseUpsample`, `AdaIN`, `AdaLayerNorm`, `Snake`, `AdainResBlock`) and tensor loaders |
| `KokoroOps` | `LstmWeights`, `lstm` / `bidirectionalLstm`, `durations` (sigmoid-sum duration decoding) |

SPI registration: `src/main/resources/META-INF/services/com.qxotic.jinfer.chat.ModelProvider` →
`com.qxotic.jinfer.models.kokoro.KokoroProvider`.

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / tests"]
    subgraph chat["jinfer-chat"]
        Models["Models.loadSpeech (ServiceLoader)"]
        MP["ModelProvider"]
    end
    subgraph core["jinfer-core"]
        SSM["SpeechSynthesisModel / Model"]
        Ph["Phonemizer.ipa"]
        RS["RuntimeState"]
        WS["Workspace / Views / Arenas / Parallel"]
        Au["Media.Audio"]
    end
    subgraph codecs["jinfer-codecs"]
        Es["Espeak (espeak-ng on PATH)"]
    end
    subgraph kk["jinfer-kokoro"]
        P["KokoroProvider"]
        T["KokoroTTS"]
        K["Kokoro"]
        Mi["Misaki"]
        Sub["TextEncoder · PlBert · ProsodyPredictor<br/>KokoroDecoder · KokoroGenerator"]
        Dsp["KokoroDsp"]
        L["KokoroLayers · KokoroOps"]
    end
    subgraph kern["jinfer-kernels"]
        Kn["MatMul · Convolutions · Norms · FlashAttention<br/>Activations · Ops · Convert · ModelLoader"]
    end
    G["gguf: GGUF / TensorEntry"]
    J["jota-memory: MemoryView / MemoryArena"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|creates| T
    T -.->|implements| SSM
    T --> K
    T --> Ph
    T --> Es
    T --> Mi
    K --> Sub --> L --> Kn
    K --> Dsp
    K --> G
    K --> J
    T -->|returns| Au
    K -.->|State extends| RS
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider {
        <<interface, external>>
        +architectures() Set~String~
        +companionFiles() Map~String,String~
        +loadSpeech(channel, gguf, path, arena, companions) SpeechSynthesisModel
    }
    class Model~C,W,S~ {
        <<interface, external>>
        +configuration() C
        +weights() W
    }
    class SpeechSynthesisModel~C,W,S~ {
        <<interface, external>>
        +sampleRate() int
        +phonemizer() Phonemizer
        +newState() S
        +newState(arena) S
        +synthesize(state, phonemes, options) Audio
        +speak(state, text, options, sink)
    }
    class Phonemizer {
        <<interface, external>>
        +phonemize(text) int[]
        +ipa(symbols, ipa)$ Phonemizer
    }
    class RuntimeState {
        <<abstract, external>>
        +exclusively(op)
        #checkResourcesAlive()
        #releaseResources()
        +close()
    }
    class Espeak {
        <<external>>
        +find()$ Optional~Espeak~
        +tiedIpa(run, language) String
    }
    class UnaryOperator~String~ {
        <<interface, external>>
    }
    class KokoroProvider {
        +architectures() Set~String~
        +companionFiles() Map~String,String~
        +loadSpeech(...) SpeechSynthesisModel
    }
    class KokoroTTS {
        -CHUNK_LIMIT = 200
        -Kokoro model
        -Phonemizer phonemizer
        +load(model, voice, arena)$ KokoroTTS
        +load(channel, gguf, voice, arena)$ KokoroTTS
        -wrap(kokoro)$ KokoroTTS
        +synthesize(state, phonemes, options) Audio
        +speak(state, text, options, sink)
        ~chunks(text)$ List~String~
        -splitAtWord(text, preferred, earliest)$ int
    }
    class Kokoro
    class Kokoro_State["Kokoro.State"]
    class Misaki {
        -regional List
        ~forLanguage(language)$ Misaki
        +apply(tiedIpa) String
    }

    Model <|-- SpeechSynthesisModel
    ModelProvider <|.. KokoroProvider
    SpeechSynthesisModel <|.. KokoroTTS
    RuntimeState <|-- Kokoro_State
    UnaryOperator <|.. Misaki
    KokoroProvider ..> KokoroTTS : load
    KokoroTTS --> Kokoro : model
    KokoroTTS --> Phonemizer : phonemizer
    KokoroTTS ..> Espeak : find
    KokoroTTS ..> Misaki : forLanguage
    Kokoro ..> Kokoro_State : newState
```

`KokoroProvider.companionFiles()` returns `{"voice": "voice"}`; `loadSpeech` throws `IllegalArgumentException` when the
`voice` companion is missing. `KokoroTTS.wrap` throws `IOException` when `Espeak.find()` is empty — Kokoro has no lexicon,
so espeak is mandatory. The phonemizer is `Phonemizer.ipa(tokens, run -> misaki.apply(espeak.tiedIpa(run, language)))`.

### 2b. `Kokoro` and its sub-networks

```mermaid
classDiagram
    direction TB
    class Kokoro {
        +ARCHITECTURE = "kokoro"
        +VOICE_ARCHITECTURE = "kokoro-voice"
        +MAX_PHONEMES = 510
        -float[] sourceWeight
        -float sourceBias
        -long parameterCount
        +load(model, voice, arena)$ Kokoro
        +load(channel, gguf, voice, arena)$ Kokoro
        ~readConfig(gguf)$ Configuration
        ~readVoice(gguf)$ Voice
        ~readVoiceLanguage(gguf)$ String
        -style(rawPhonemes, offset, allocator) MemoryView
        +newState() State
        +newState(arena) State
        +synthesize(state, rawPhonemes, speed, seed) float[]
    }
    class Configuration {
        <<record>>
        dimIn, hiddenDim, styleDim, maxConvDim, maxDuration
        tokenCount, melCount, decoderLayers, sampleRate
        istftInitialChannels, istftFftSize, istftHopSize
        int[] upsampleRates, resblockKernelSizes, ...
        plbertHiddenSize, plbertLayers, plbertHeads, ...
        String[] tokens
    }
    class Voice {
        <<record>>
        int maxPhonemes
        int styleDimensions
    }
    class Weights {
        <<record>>
        Map model
        MemoryView voicePack
    }
    class State {
        -MemoryArena owned
        -MemoryArena allocator
        -Workspace scratch
        -Runnable disarm
        -Random random
    }
    class TextEncoder_Weights["TextEncoder.Weights"]
    class PlBert_Weights["PlBert.Weights"]
    class ProsodyPredictor_Weights["ProsodyPredictor.Weights"]
    class KokoroDecoder_Weights["KokoroDecoder.Weights"]
    class KokoroGenerator_Weights["KokoroGenerator.Weights"]
    class Workspace { <<external>> }
    class LeakWatch { <<external>> }
    class MemoryView { <<external>> }

    Kokoro *-- Configuration
    Kokoro *-- Voice
    Kokoro *-- Weights
    Kokoro *-- TextEncoder_Weights
    Kokoro *-- PlBert_Weights
    Kokoro *-- ProsodyPredictor_Weights
    Kokoro *-- KokoroDecoder_Weights
    Kokoro *-- KokoroGenerator_Weights
    Kokoro ..> State : creates
    State --> Workspace : scratch
    State ..> LeakWatch : arm
    Weights --> MemoryView
```

```mermaid
classDiagram
    direction LR
    class TextEncoder_Weights["TextEncoder.Weights"] {
        embedding
        List~Conv~ convolutions
        LstmWeights forward, reverse
    }
    class TextEncoder_Conv["TextEncoder.Conv"] {
        float[] taps
        bias, gamma, beta
    }
    class PlBert_Weights["PlBert.Weights"] {
        tokenEmbedding, positionEmbedding, tokenType
        embeddingGamma, embeddingBeta
        Linear embeddingProjection
        Layer layer
        Linear outputProjection
    }
    class PlBert_Layer["PlBert.Layer"] {
        Linear query, key, value, attentionOutput
        Linear ffnUp, ffnDown
        attention/ffn gamma, beta
    }
    class PP_Weights["ProsodyPredictor.Weights"] {
        List~DurationLayer~ durationEncoder
        LstmWeights durationForward/Reverse
        Linear durationProjection
        LstmWeights sharedForward/Reverse
        Branch f0, noise
    }
    class PP_DurationLayer["ProsodyPredictor.DurationLayer"] {
        LstmWeights forward, reverse
        AdaLayerNorm normalization
    }
    class PP_Branch["ProsodyPredictor.Branch"] {
        List~AdainResBlock~ blocks
        Conv1d projection
    }
    class PP_Output["ProsodyPredictor.Output"] {
        int[] alignmentIndices
        float[] f0
        float[] noise
    }
    class Dec_Weights["KokoroDecoder.Weights"] {
        Conv1d f0Convolution, noiseConvolution, asrResidual
        AdainResBlock encoder
        List~AdainResBlock~ decoder
    }
    class Gen_Weights["KokoroGenerator.Weights"] {
        List~Stage~ stages
        Conv1d postConvolution
    }
    class Gen_Stage["KokoroGenerator.Stage"] {
        Conv1d noiseConvolution
        AdaINResBlock1 noiseResidual
        ConvTranspose1d upsample
        List~AdaINResBlock1~ residuals
        boolean leftReflectionPad
    }
    class Gen_Block["KokoroGenerator.AdaINResBlock1"] {
        List~Step~ steps
        forward(input, time, style, scratch)
    }
    class Gen_Step["KokoroGenerator.Step"] {
        AdaIN adain1, adain2
        Snake snake1, snake2
        Conv1d convolution1, convolution2
        int dilation
    }
    class Spectrum["KokoroDsp.Spectrum"] {
        float[][] magnitude
        float[][] phase
    }

    TextEncoder_Weights *-- TextEncoder_Conv
    PlBert_Weights *-- PlBert_Layer
    PP_Weights *-- PP_DurationLayer
    PP_Weights *-- PP_Branch
    Gen_Weights *-- Gen_Stage
    Gen_Stage *-- Gen_Block
    Gen_Block *-- Gen_Step
```

### 2c. Reusable layers — `KokoroLayers` and `KokoroOps`

```mermaid
classDiagram
    direction LR
    class KokoroLayers {
        ~adainResBlock(...)$ AdainResBlock
        ~linear(...)$ Linear
        ~conv1d(...)$ Conv1d
        ~convTranspose1d(...)$ ConvTranspose1d
        ~depthwiseUpsample(...)$ DepthwiseUpsample
        ~nearestUpsample(...)$ MemoryView
        ~vector(...)$ MemoryView
        ~matrix(...)$ MemoryView
        ~lstm(...)$ LstmWeights
        ~convolutionTaps(...)$ float[]
    }
    class Linear {
        weight, bias, inputSize, outputSize
        forward()
    }
    class Conv1d {
        float[] taps, bias, kernel, inChannels, outChannels
        forward()
        forward(stride, padding)
    }
    class ConvTranspose1d {
        weight, bias, kernel, inChannels, outChannels
        forward()
    }
    class DepthwiseUpsample {
        float[] taps, bias, channels
        forward()
    }
    class AdaIN {
        Linear style, channels
        forward()
    }
    class AdaLayerNorm {
        Linear style, channels
        forward()
    }
    class Snake {
        alpha, channels
        forward()
    }
    class AdainResBlock {
        AdaIN normalization1, normalization2
        DepthwiseUpsample pool
        Conv1d convolution1, convolution2, shortcut
        forward(input, time, style, scratch)
    }
    class KokoroOps {
        ~bidirectionalLstm(input, fwd, rev, output, scratch)$
        ~lstm(...)$
        ~durations(logits, tokenCount, maxDuration, speed, scratch)$ int[]
    }
    class LstmWeights {
        input, hidden, inputBias, hiddenBias
    }
    class Kernels["MatMul / Convolutions / Ops / Convert / ModelLoader"] { <<external>> }

    KokoroLayers ..> Linear
    KokoroLayers ..> Conv1d
    KokoroLayers ..> ConvTranspose1d
    KokoroLayers ..> DepthwiseUpsample
    KokoroLayers ..> AdainResBlock
    KokoroLayers ..> LstmWeights : lstm()
    AdainResBlock *-- AdaIN
    AdainResBlock *-- DepthwiseUpsample
    AdainResBlock *-- Conv1d
    AdaIN --> Linear
    AdaLayerNorm --> Linear
    KokoroOps *-- LstmWeights
    Linear ..> Kernels
    Conv1d ..> Kernels
    KokoroLayers ..> Kernels
```

`AdainResBlock.forward`: AdaIN → LeakyReLU(0.2) → optional depthwise ×2 upsample → conv1 → AdaIN → LeakyReLU → conv2,
plus a shortcut path (nearest ×2 upsample when pooled, 1×1 conv when channel counts differ), summed and scaled by
1/√2. `vector` dequantizes non-FP32 tensors to FP32 in the persistent allocator; `matrix` keeps the file-backed
(possibly quantized) view for `MatMul.gemm`; `convolutionTaps` copies conv weights to a host `float[]`.

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant C as Caller
    participant M as Models (jinfer-chat)
    participant P as KokoroProvider
    participant T as KokoroTTS
    participant K as Kokoro
    participant L as ModelLoader
    participant S as Sub-network loaders
    participant E as Espeak

    C->>M: loadSpeech(path, arena, companions{voice})
    M->>P: loadSpeech(channel, gguf, path, arena, companions)
    P->>T: load(channel, gguf, voicePath, arena)
    T->>K: load(channel, gguf, voice, arena)
    K->>K: readConfig(gguf) - fixed-layout checks
    K->>L: readGguf(voiceChannel)
    K->>K: readVoice / readVoiceLanguage
    K->>L: loadTensors(channel, gguf, arena)
    K->>S: TextEncoder / PlBert / ProsodyPredictor / KokoroDecoder / KokoroGenerator .load(tensors, ...)
    K->>K: m_source weight[9] + bias to host floats
    K->>L: loadTensors(voice) + requireF32("voice.pack")
    K-->>T: Kokoro
    T->>E: Espeak.find()
    T->>T: Misaki.forLanguage(language)
    T->>T: Phonemizer.ipa(tokens, misaki over espeak.tiedIpa)
    T-->>C: KokoroTTS
```

`readConfig` refuses anything but the exact v1.0 layout (hidden 512, style 128, 178 tokens, 24 kHz, n_fft 20 / hop 5,
upsample rates `{10, 6}`, PL-BERT 768×12 layers ×12 heads, …). `readVoice` requires `voice.pack` F32 of GGUF shape
`[256, 1, max_phonemes ≥ 510]`. `readVoiceLanguage` maps the first letter of `kokoro_voice.name` to an espeak
language (`a`→`en-us`, `b`→`en-gb`, `e`→`es`, `f`→`fr-fr`, `h`→`hi`, `i`→`it`, `j`→`ja`, `p`→`pt-br`, `z`→`cmn`).
Weights live in the caller's `Arena`; a `MemoryAllocators.ofArena(arena)` "persistent" allocator holds dequantized
vectors.

---

## 4. Text front end: `speak`

```mermaid
flowchart TB
    A["speak(state, text, options, sink)"] --> B{"text blank?"}
    B -- yes --> X["IllegalArgumentException"]
    B -- no --> C["state.exclusively"]
    C --> D["chunks(text): collapse whitespace,<br/>split after . ! ? ; :, cap 200 chars at a word"]
    D --> E["deque of parts"]
    E --> F["phonemizer.phonemize(part)<br/>espeak tiedIpa -> Misaki -> symbol ids"]
    F --> G{"length"}
    G -- "0" --> E
    G -- "> 510" --> H["splitAtWord(part, len/2)<br/>push both halves back"] --> E
    G -- "1..510" --> I["synthesize(state, ids, options)"]
    I --> J{"sink.test(audio)"}
    J -- false --> K["stop"]
    J -- true --> E
```

`KokoroTTS.synthesize` reads `options.speed()` (default 1) and calls `Kokoro.synthesize(state, ids, speed, seed=0)`,
wrapping the PCM in `Media.Audio(pcm, 24000, 1)`. If no chunk produced phonemes, `speak` throws.

`Misaki.apply`: for non-English voices one `OTHER` table (ties → single symbols such as `ʤ ʦ A I O W Y Q`);
for English, the `ENGLISH` table, a syllabic-mark → `ᵊ` rewrite, then the regional table (`AMERICAN` or `BRITISH`),
then `ENGLISH_TAIL` (flap `ɾ`→`T`, glottal `ʔ`→`t`, tie removed).

---

## 5. Synthesis pipeline

### 5.0 Global architecture

```mermaid
flowchart LR
    ids["raw phoneme ids<br/>pad 0 at both ends"] --> PB["PlBert.encode<br/>[N, 512]"]
    ids --> TE["TextEncoder.encode<br/>[N, 512]"]
    VP["voice pack row min(len,510)-1"] --> ds["decoder style [1,128]"]
    VP --> ps["predictor style [1,128]"]
    PB --> PP["ProsodyPredictor.forward"]
    ps --> PP
    PP -->|"alignment, f0, noise"| DEC["KokoroDecoder.forward<br/>[512, 2T]"]
    TE --> DEC
    ds --> DEC
    PP -->|f0| SRC["KokoroDsp.sourceStft<br/>harmonic Spectrum"]
    DEC --> GEN["KokoroGenerator.forward<br/>Spectrum"]
    SRC --> GEN
    ds --> GEN
    GEN --> IST["KokoroDsp.istft"] --> PCM["clamp to [-1,1]<br/>float[] PCM 24 kHz"]
```

### 5a. `Kokoro.synthesize` sequence

```mermaid
sequenceDiagram
    participant T as KokoroTTS
    participant K as Kokoro
    participant St as Kokoro.State
    participant PB as PlBert
    participant PP as ProsodyPredictor
    participant TE as TextEncoder
    participant D as KokoroDecoder
    participant DSP as KokoroDsp
    participant G as KokoroGenerator

    T->>K: synthesize(state, ids, speed, 0)
    K->>St: exclusively(...)
    K->>K: validate length 1..510, ids on table, speed in [0.5, 2]
    K->>St: scratch.rewind(), tokens = [0, ids..., 0]
    K->>K: style(len, styleDim) predictor / style(len, 0) decoder
    K->>PB: encode(plBert, tokens)
    K->>PP: forward(predictor, bert, predictorStyle, N, speed)
    PP-->>K: Output(alignment, f0, noise)
    K->>TE: encode(textEncoder, tokens, 512, 5)
    K->>D: forward(decoder, text, alignment, f0, noise, decoderStyle)
    K->>DSP: sourceStft(f0, sourceWeight, sourceBias, state.random(seed))
    K->>G: forward(generator, decoded, harmonic, decoderStyle)
    G-->>K: Spectrum(magnitude, phase)
    K->>DSP: istft(magnitude, phase)
    K-->>T: float[] pcm clamped
```

All intermediate tensors come from the state's `Workspace` scratch (rewound per call); sub-steps use
`Workspace.scope(scratch)` to release temporaries. `Reference.reachabilityFence(this)` keeps the model's file-backed
weights alive through the call.

---

## 6. Sub-network forwards

### 6a. `TextEncoder.encode`

```mermaid
flowchart LR
    t["tokens"] --> g["Convert.gatherToF32<br/>embedding [N,512]"]
    g --> loop["3x: transpose -> Convolutions.conv1dRows k=5<br/>-> transpose -> Norms.layerNormRows -> LeakyReLU 0.2"]
    loop --> l["KokoroOps.bidirectionalLstm<br/>256 + 256"] --> o["[N, 512]"]
```

### 6b. `PlBert.encode`

```mermaid
flowchart LR
    t["tokens"] --> e["token emb + position emb + token-type<br/>LayerNorm eps 1e-12 [N,128]"]
    e --> p["embd_proj -> [N,768]"]
    p --> tr["transform: 12 repeats of ONE shared layer<br/>QKV -> FlashAttention.bidirectionalPrefill 12 heads<br/>-> attn_o + residual + LN -> ffn_up GELU ffn_down + residual + LN"]
    tr --> o["bert_proj -> [N,512]"]
```

### 6c. `ProsodyPredictor.forward`

```mermaid
flowchart TB
    inp["bert [N,512] ++ style -> [N,640]"] --> de["3x DurationLayer:<br/>BiLSTM -> AdaLayerNorm(style) -> append style"]
    de --> dl["BiLSTM pred.lstm -> dur_proj [N,50]"]
    dl --> du["KokoroOps.durations: sum sigmoid / speed,<br/>round, min 1"]
    du --> al["repeatAlignment -> frame indices T"]
    de --> ga["gatherRows by alignment [T,640]"]
    al --> ga
    ga --> sh["BiLSTM pred.shared -> transpose [512,T]"]
    sh --> f0["F0 Branch: 3 AdainResBlocks, 2nd upsamples x2<br/>-> conv1x1 -> f0[2T]"]
    sh --> nn["N Branch: same layout -> noise[2T]"]
```

### 6d. `KokoroDecoder.forward`

```mermaid
flowchart TB
    te["text encoding [N,512]"] --> asr["gatherAsr by alignment -> [512,T]"]
    f0["f0[2T]"] --> fd["F0_conv k3 stride 2 -> [1,T]"]
    no["noise[2T]"] --> nd["N_conv k3 stride 2 -> [1,T]"]
    asr --> c1["concat [514,T]"]
    fd --> c1
    nd --> c1
    c1 --> enc["encode AdainResBlock 514->1024"]
    asr --> res["asr_res conv1x1 -> [64,T]"]
    enc --> loop["4x: concat cur ++ asr_res ++ f0 ++ N [1090,T]<br/>-> decode block; last block 1090->512 with x2 upsample"]
    res --> loop
    loop --> out["[512, 2T]"]
```

### 6e. `KokoroGenerator.forward` and `KokoroDsp`

```mermaid
flowchart TB
    dec["decoder [512, 2T]"] --> s0
    harm["harmonic Spectrum -> [22, frames]<br/>frames = 60*2T + 1"] --> n0["noise_convs.0 k12 s6 + noise_res.0"]
    harm --> n1["noise_convs.1 k1 + noise_res.1"]
    subgraph s0["Stage 0"]
        a0["LeakyReLU 0.1 -> ups.0 ConvTranspose x10 -> 256 ch"]
    end
    subgraph s1["Stage 1"]
        a1["LeakyReLU 0.1 -> ups.1 ConvTranspose x6 -> 128 ch<br/>-> reflection pad left 1"]
    end
    s0 --> add0["+ injected source"] --> r0["mean of 3 AdaINResBlock1, k 3/7/11"] --> s1
    n0 --> add0
    s1 --> add1["+ injected source"] --> r1["mean of 3 AdaINResBlock1"]
    n1 --> add1
    r1 --> post["LeakyReLU 0.01 -> conv_post k7 -> [22, frames]"]
    post --> ot["outputTransform: magnitude = exp, phase = sin"]
    ot --> istft["KokoroDsp.istft -> (frames-1)*5 samples"]
```

`KokoroDsp.sourceStft`: per-harmonic phase accumulation over F0 (`2T` samples), 300:1 centered interpolation to
the sample rate, sine (amplitude 0.1, voiced when F0 > 10) plus seeded Gaussian noise for nine harmonics, mixed by
`m_source` linear weights + bias and `tanh`, then a reflect-padded, Hann-windowed 20-point DFT (11 bins, hop 5).
`istft` inverts with precomputed twiddles and overlap-add normalised by the window-square sum. Noise draws are taken
sequentially from one `Random` before the `Parallel.forLoop`s, so output is deterministic for a seed.

---

## 7. State, memory and validation

```mermaid
classDiagram
    direction LR
    class RuntimeState { <<abstract, external>> }
    class State["Kokoro.State"] {
        -MemoryArena owned
        -MemoryArena allocator
        -Workspace scratch
        -Runnable disarm
        -Random random
        -random(seed) Random
        ~scratchAllocations() int
        #checkResourcesAlive()
        #releaseResources()
    }
    class Arenas { <<external>> }
    class LeakWatch { <<external>> }
    class Workspace { <<external>> }
    RuntimeState <|-- State
    State --> Workspace
    State ..> Arenas : requireCrossThread / close
    State ..> LeakWatch : arm
```

- `newState()` creates an owned cross-thread arena (`Arenas.newCrossThreadMemoryArena`) closed on `close()`;
  `newState(arena)` borrows a caller arena (must be cross-thread, checked by `Arenas.requireCrossThread`).
- There is no KV cache or checkpoint: each `synthesize` call is stateless apart from scratch memory and the
  reseeded `Random`; `RuntimeState.exclusively` serialises use of one state.
- Trust boundary in `Kokoro.synthesize`: non-empty, ≤ 510 ids, each id on a non-empty slot of the 178-entry symbol
  table, speed finite in `[0.5, 2.0]`, voice pack still alive (`Views.checkAlive`).
- Voice style row is chosen by input length: row `min(len, maxPhonemes) - 1`; the first 128 values are decoder
  style, the next 128 predictor style.
