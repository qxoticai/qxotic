# toknroll-core — Architecture

`toknroll-core` (JPMS module `com.qxotic.toknroll`, Java 11, zero dependencies) is the heart of
Tok'n'Roll: the public tokenizer API, the two BPE models (tiktoken byte-level BPE and
SentencePiece BPE), the pre-tokenizers (splitters), vocabularies, special-token matching and the
SPI through which a faster engine (toknroll-vector) can take over the tiktoken hot path.

| Package | Exported to | Content |
|---|---|---|
| `com.qxotic.toknroll` | everyone | API: `Tokenizer`, `TokenizationModel`, `TokenizationPipeline`, `Normalizer`, `Splitter`, `Vocabulary`, `Specials`, `IntSequence`, `MergeRule`, `TokenType`, `ByteLevel`, `Toknroll` factory, exceptions |
| `com.qxotic.toknroll.spi` | everyone | `TiktokenEngine` (ServiceLoader SPI), `TiktokenTables`, `CharClasses` |
| `com.qxotic.toknroll.impl` | `toknroll.gguf`, `toknroll.hf` only | Implementations: models, splitters, vocabularies, int sequences, `ImplAccessor` bridge |

`module-info` declares `uses com.qxotic.toknroll.spi.TiktokenEngine`.

## 1. The tokenizer contract

Everything that encodes or decodes is a `Tokenizer`. A `TokenizationModel` is a `Tokenizer` that
works on one pre-split piece (a "word"); a `TokenizationPipeline` chains
normalizer → splitter → model and is itself a `Tokenizer`.

```mermaid
classDiagram
    direction TB
    class Tokenizer {
        <<interface>>
        +vocabulary() Vocabulary
        +encodeInto(CharSequence text, int start, int end, Builder out)
        +encodeUtf8Into(byte[] utf8, int start, int end, Builder out)
        +countTokens(CharSequence text, int start, int end) int
        +decodeBytesInto(IntSequence tokens, int from, ByteBuffer out) int
        +encode(CharSequence) IntSequence
        +encodeToArray(CharSequence) int[]
        +encodeUtf8(byte[]) IntSequence
        +decode(IntSequence) String
        +decode(int[]) String
        +decodeBytes(IntSequence) byte[]
        +countBytes(IntSequence) int
        +expectedTokensPerChar() float
    }
    class TokenizationModel {
        <<interface>>
    }
    class TokenizationPipeline {
        <<final>>
        -TokenizationModel model
        -Normalizer normalizer
        -Splitter splitter
        -boolean scanned
        +model() TokenizationModel
        +normalizer() Optional~Normalizer~
        +splitter() Optional~Splitter~
    }
    class Normalizer {
        <<interface>>
        +apply(CharSequence) CharSequence
        +identity()$ Normalizer
        +lowercase()$ Normalizer
        +unicode(Form)$ Normalizer
        +sequence(Normalizer...)$ Normalizer
    }
    class Splitter {
        <<interface>>
        +splitAll(CharSequence text, int start, int end, SplitConsumer c)
        +splitAllToListEagerly(CharSequence) List
        +identity()$ Splitter
        +regex(Pattern)$ Splitter
        +sequence(Splitter...)$ Splitter
    }
    class SplitConsumer {
        <<functional>>
        +accept(CharSequence source, int start, int end)
    }
    class Toknroll {
        <<factory>>
        +vocabulary(String...)$ Vocabulary
        +vocabulary(Map specials, String...)$ Vocabulary
        +tiktokenModel(Vocabulary, List~MergeRule~)$ TokenizationModel
        +sentencePieceBpeModel(Vocabulary, List~MergeRule~)$ TokenizationModel
        +sentencePieceBpeModel(Vocabulary, float[] scores)$ TokenizationModel
        +pipeline(Normalizer, Splitter, TokenizationModel)$ TokenizationPipeline
    }
    class UnicodeNormalizer {
        <<package-private>>
    }

    Tokenizer <|-- TokenizationModel
    Tokenizer <|.. TokenizationPipeline
    TokenizationPipeline o-- TokenizationModel
    TokenizationPipeline o-- Normalizer : optional
    TokenizationPipeline o-- Splitter : optional
    Normalizer <|.. UnicodeNormalizer
    Splitter *-- SplitConsumer : nested
    Toknroll ..> TokenizationPipeline : builds
    Toknroll ..> TokenizationModel : builds
```

`Tokenizer` only has four abstract methods (`vocabulary`, `encodeInto` on a range,
`countTokens` on a range, `decodeBytesInto`); every other method is a `default` built on them.
The zero-copy calls (`encodeInto(..., Builder)`, `decodeBytesInto(..., ByteBuffer)`) are the hot
loop APIs.

## 2. Vocabulary, token types, merges and special tokens

```mermaid
classDiagram
    direction LR
    class Vocabulary {
        <<interface>>
        +size() int
        +token(int id) String
        +id(String text) int
        +contains(int id) boolean
        +contains(String text) boolean
        +findToken(int id) Optional~String~
        +findId(String text) OptionalInt
        +isTokenOfType(int id, TokenType t) boolean
    }
    class TokenType {
        <<interface>>
    }
    class StandardTokenType {
        <<enumeration>>
        NORMAL
        UNKNOWN
        CONTROL
        USER_DEFINED
        UNUSED
        BYTE
        +getId() int
    }
    class MergeRule {
        <<final>>
        +of(int left, int right, int rank)$ MergeRule
        +leftId() int
        +rightId() int
        +rank() int
    }
    class Specials {
        <<interface>>
        +tokens() Map~String,Integer~
        +isEmpty() boolean
        +encodeInto(Tokenizer t, CharSequence text, Builder out)
        +encodeUtf8Into(Tokenizer t, byte[] utf8, int s, int e, Builder out)
        +encode(Tokenizer t, CharSequence text) IntSequence
        +none()$ Specials
        +of(Tokenizer)$ Specials
    }
    class ByteLevel {
        <<final utility>>
        +encode(byte[])$ String
        +decode(String)$ byte[]
        +encodeSingle(byte)$ char
        +decodeSingle(char)$ byte
        +isValidEncoding(CharSequence)$ boolean
    }
    class VocabularyImpl {
        <<impl>>
        -String[] tokens
        -int[] tokenTypes
    }
    class VocabularyWithSpecials {
        <<impl>>
        -Vocabulary inner
        -Map specialToIndex
    }
    class SpecialsImpl {
        <<impl>>
        -Map tokens
        -Trie chars
        -Trie bytes
    }

    TokenType <|.. StandardTokenType
    Vocabulary ..> TokenType
    Vocabulary <|.. VocabularyImpl
    Vocabulary <|.. VocabularyWithSpecials
    VocabularyWithSpecials o-- Vocabulary : inner
    Specials <|.. SpecialsImpl
    SpecialsImpl ..> Vocabulary : CONTROL / USER_DEFINED / UNKNOWN ids
```

* `Specials.of(tokenizer)` scans the vocabulary once for tokens typed CONTROL, USER_DEFINED or
  UNKNOWN, and builds two tries (chars and UTF-8 bytes). Encoding with it matches special tokens
  whole and hands the text between them to the tokenizer. Plain `Tokenizer.encode` never matches
  specials (`Specials.none()` behaviour).
* `ByteLevel` is GPT-2's bytes ↔ printable-unicode mapping used by byte-level BPE vocabularies.

## 3. Token id sequences

```mermaid
classDiagram
    direction TB
    class IntSequence {
        <<interface>>
        +intAt(int) int
        +length() int
        +subSequence(int, int) IntSequence
        +toArray() int[]
        +copyTo(int src, int[] dest, int off, int n)
        +stream() IntStream
        +concat(IntSequence) IntSequence
        +of(int...)$ IntSequence
        +wrap(int[])$ IntSequence
        +newBuilder()$ Builder
    }
    class Builder {
        <<interface>>
        +size() int
        +ensureCapacity(int)
        +add(int) Builder
        +addAll(IntSequence) Builder
        +build() IntSequence
        +snapshot() IntSequence
        +asSequenceView() IntSequence
    }
    class AbstractIntSequence {
        <<abstract>>
        +equals / hashCode / compareTo / toString
    }
    class ArrayIntSequence {
        -int[] array
        -int offset
        -int length
    }
    class ListIntSequence {
        -List~Integer~ list
    }
    class IntSequenceBuilder {
        -int[] buffer
        -int size
    }
    IntSequence *-- Builder : nested
    IntSequence <|.. AbstractIntSequence
    AbstractIntSequence <|-- ArrayIntSequence
    AbstractIntSequence <|-- ListIntSequence
    Builder <|.. IntSequenceBuilder
    IntSequenceBuilder ..> ArrayIntSequence : build / view
```

`IntSequence` is the immutable, `Comparable` view of token ids; `Builder` is the append-only sink
every encode writes into (no boxing, no copy until `build()`).

## 4. Models (impl)

```mermaid
classDiagram
    direction TB
    class TokenizationModel {
        <<interface>>
    }
    class AbstractTokenizationModel {
        <<abstract>>
        #Vocabulary vocabulary
        #float expectedTokensPerChar
        #encodeImpl(CharSequence)* IntSequence
        #encodeImplInto(CharSequence, int, int, Builder)
        #heapLess(int, long, int, long)$ boolean
        +countTokens(...) int
    }
    class TiktokenModel {
        <<byte-level BPE>>
        -LongLongMap merges
        -int[] singleByteTokenId
        -byte[][] tokenBytesById
        -Word[] memo
        -List~TiktokenEngine~ ENGINES$
        +encode(text, s, e, TiktokenSplitter, Builder)
        +encodeUtf8(utf8, s, e, TiktokenSplitter, Builder)
        +decodeBytesInto(...) int
    }
    class Tables {
        <<inner>>
    }
    class SentencePieceBpeModel {
        <<SentencePiece BPE>>
        -String[] tokensById
        -int[] byteTokenIds
        -LongLongMap packedMerges
        -int[] bmpSingletonIds
        -AtomicReference~WordCache~ cache
        +fromVocabularyAndMerges(...)$
        +fromVocabulary(Vocabulary, float[] scores)$
    }
    class LongLongMap {
        -char[] gridIndex
        -long[] gridValues
        -long[] table
        +get(long key) long
    }
    class TiktokenReconstruction {
        <<utility>>
        +vocabulary(mergeableRanks, specials, userDefined)$ Vocabulary
        +mergeRules(mergeableRanks)$ List~MergeRule~
        ~packTiktokenMerges(...)$ LongLongMap
        ~packSentencePieceMerges(...)$ LongLongMap
    }
    class MetaspaceTokenizer {
        <<Tokenizer decorator>>
        -Tokenizer base
        -Normalizer normalizer
        -boolean prependAlways
        -boolean prependFirst
        -boolean trimLeadingSpace
    }
    class TiktokenTables {
        <<interface spi>>
    }

    TokenizationModel <|.. AbstractTokenizationModel
    AbstractTokenizationModel <|-- TiktokenModel
    AbstractTokenizationModel <|-- SentencePieceBpeModel
    TiktokenModel *-- LongLongMap
    SentencePieceBpeModel *-- LongLongMap
    TiktokenModel *-- Tables
    TiktokenTables <|.. Tables
    TiktokenReconstruction ..> LongLongMap : packs
    MetaspaceTokenizer o-- Tokenizer : base pipeline
    Tokenizer <|.. MetaspaceTokenizer
```

* **TiktokenModel** – reference byte-level BPE (GPT-2/cl100k/o200k/Llama 3/Qwen…): a word's bytes
  map to single-byte tokens, then the lowest-rank (leftmost on ties) pair merges until none
  applies. Short words (≤ 32 bytes) are memoized in a lock-free 64K-slot memo. Merge pairs live in
  `LongLongMap` (dense 512×512 grid for early merges + open-addressing table for the rest).
* **SentencePieceBpeModel** – SentencePiece BPE with byte fallback (`<0x00>`…`<0xFF>`), merges by
  rank or by scores; O(n²) scan for short words, heap for words over 128 symbols (worst case
  `O(n log n)`); a 24 MB lock-free seqlock word cache built lazily by one thread.
* **MetaspaceTokenizer** – wraps a SentencePiece pipeline to handle `▁` (U+2581): normalizes,
  prepends `▁` per `prepend_scheme` (`always` / `first` / `never`), and trims the leading space
  on decode.
* **TiktokenReconstruction** – rebuilds vocabulary + merge rules from a `.tiktoken` mergeable-ranks
  file.

## 5. Splitters (pre-tokenizers, impl)

```mermaid
classDiagram
    direction LR
    class Splitter {
        <<interface>>
    }
    class RegexSplitter {
        -Pattern pattern
        -boolean emitGaps
        +create(Pattern)$ RegexSplitter
    }
    class TiktokenSplitter {
        <<hand-written scanner>>
        -int contraction
        -int word
        -int numbers
        -int otherTail
        -int rules
        -int cuts
        -TiktokenSplitter base
        ~forPattern(Pattern)$ Splitter
        ~forStages(Splitter...)$ Splitter
        ~sample()$ String
        ~knownPatterns()$ Iterable
    }
    class SentencePieceSplitter {
        -BitSet glue
        -boolean isolateNewlines
    }
    class StagedSplitter {
        <<Splitter.sequence>>
    }
    class CharClasses {
        <<spi utility>>
        +SPACE, LETTER, NUMBER, MARK
        +UPPER, LOWER, NEWLINE
        +of(int codePoint)$ int
    }
    Splitter <|.. RegexSplitter
    Splitter <|.. TiktokenSplitter
    Splitter <|.. SentencePieceSplitter
    Splitter <|.. StagedSplitter
    TiktokenSplitter o-- TiktokenSplitter : base (no cuts)
    TiktokenSplitter ..> CharClasses
```

* **TiktokenSplitter** recognizes the known tiktoken-family regexes (r50k/GPT-2, cl100k, Llama 3,
  Qwen2, Qwen3.5, o200k, Tekken, DeepSeek V3, number/newline "cut" stages…) by their *source*
  (`forPattern`) and replaces `java.util.regex` with a hand-written, allocation-free scanner. Unknown
  patterns fall back to `RegexSplitter`.
* When the splitter is a `TiktokenSplitter` and the model a `TiktokenModel`, the pipeline is
  **scanned**: `ImplAccessor.encodeScanned` lets the model split and merge in one pass, and that
  is where a `TiktokenEngine` can take over.
* **SentencePieceSplitter** cuts text on spaces that are not "glued" to a vocabulary token, optionally
  isolating newline runs.

## 6. Engine SPI (`com.qxotic.toknroll.spi`)

```mermaid
classDiagram
    direction LR
    class TiktokenEngine {
        <<interface ServiceLoader>>
        +CL100K = 0
        +CL100K_ASCII = 1
        +QWEN2 = 2
        +QWEN2_ASCII = 3
        +R50K = 4
        +O200K = 5
        +TEKKEN = 6
        +R50K_NUMBERS = 7
        +encoder(TiktokenTables model, int rules) Encoder
        +words(TiktokenTables model) WordEncoder
    }
    class Encoder {
        <<interface>>
        +encode(CharSequence, int, int, Builder)
        +encodeUtf8(byte[], int, int, Builder)
    }
    class WordEncoder {
        <<interface>>
        +encodeWords(byte[] utf8, int start, int[] ends, int count, Builder)
    }
    class TiktokenTables {
        <<interface>>
        +tokenCount() int
        +tokenBytes(int id) byte[]
        +mergeCount() int
        +mergeLeft(int) int
        +mergeRight(int) int
        +mergeResult(int) int
        +mergeRank(int) int
        +ignoreMerges() boolean
    }
    class TiktokenModel
    TiktokenEngine *-- Encoder
    TiktokenEngine *-- WordEncoder
    TiktokenEngine ..> TiktokenTables : reads
    TiktokenModel ..> TiktokenEngine : ServiceLoader, verified on sample
    TiktokenModel ..> TiktokenTables : exposes (inner Tables)
```

On its first encode, `TiktokenModel` asks every engine for an `Encoder` for its rule set,
**verifies it against its own encoder** on `TiktokenSplitter.sample()` (covers every kind of piece),
and only then uses it. An engine that declines, disagrees or throws is skipped.
`-Dtoknroll.engine=none` disables engines.

## 7. The `ImplAccessor` bridge

`impl` is exported only to `toknroll-gguf` and `toknroll-hf`; they never touch the package-private
classes directly but go through `ImplAccessor`:

```mermaid
classDiagram
    direction LR
    class ImplAccessor {
        <<public final, qualified export>>
        +createVocabulary(String[] tokens, int[] types)$ Vocabulary
        +createVocabularyWithSpecials(...)$ Vocabulary
        +createTiktokenModel(Vocabulary, merges, ignoreMerges)$ TokenizationModel
        +createSentencePieceBpeModel(...)$ TokenizationModel
        +createRegexSplitter(Pattern, emitGaps)$ Splitter
        +createSentencePieceSplitter(String[] tokens, isolateNewlines)$ Splitter
        +createStagedSplitter(Splitter[])$ Splitter
        +metaspaceTokenizer(...)$ Tokenizer
        +loadTiktokenMergeableRanks(BufferedReader)$ Map
        +reconstructTiktokenVocabulary(...)$ Vocabulary
        +reconstructTiktokenMergeRules(...)$ List
        +isScanned(model, splitter)$ boolean
        +encodeScanned(...)$
        +packMerge(rank, mergedId)$ long
        +pairKey(left, right)$ long
    }
    ImplAccessor ..> VocabularyImpl
    ImplAccessor ..> VocabularyWithSpecials
    ImplAccessor ..> TiktokenModel
    ImplAccessor ..> SentencePieceBpeModel
    ImplAccessor ..> RegexSplitter
    ImplAccessor ..> TiktokenSplitter
    ImplAccessor ..> SentencePieceSplitter
    ImplAccessor ..> MetaspaceTokenizer
    ImplAccessor ..> TiktokenReconstruction
    ImplAccessor ..> SpecialsImpl
```

## 8. Encode flow

```mermaid
sequenceDiagram
    participant C as caller
    participant P as TokenizationPipeline
    participant N as Normalizer
    participant S as Splitter
    participant M as TokenizationModel
    participant E as TiktokenEngine (optional)
    C->>P: encodeInto(text, start, end, out)
    P->>N: apply(text) (if configured)
    alt scanned (TiktokenSplitter + TiktokenModel)
        P->>M: encode(text, splitter, out)
        M->>E: Encoder.encode (if verified engine)
        E-->>M: ids appended to out
    else general
        P->>S: splitAll(text, consumer)
        S-->>P: piece [s, e)
        P->>M: encodeInto(piece, out)
    end
    P-->>C: out holds token ids
```

Other small types: `TokenizerLoadException` (I/O while loading), `UnsupportedTokenizerException`
(a valid tokenizer that Tok'n'Roll does not implement), `IntPair` / `IteratorCombiner` (internal
helpers).
