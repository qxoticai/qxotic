# toknroll-testkit — Architecture

`toknroll-testkit` (JPMS module `com.qxotic.toknroll.testkit`) is the shared **test support**
library of Tok'n'Roll: golden fixtures (ids produced by the Python reference tokenizers),
parity / invariant harnesses, test corpora and cache paths. It is a `main` artifact so that every
module's tests (core, gguf, hf, vector, incubator, benchmarks) can depend on it. It does **not**
depend on `toknroll-core`: harnesses take lambdas (`int[] encode(String)`, …), which keeps them
usable for any implementation (Tok'n'Roll, JTokkit, native).

Dependencies: `com.qxotic.format.json`, `java.net.http`.

## 1. Golden fixtures

```mermaid
classDiagram
    direction TB
    class FixtureJsonLoader {
        <<final>>
        +loadMap(Class owner, String resource, String context, String generator)$ Map
    }
    class FamilyGoldenFixture {
        <<final>>
        +load()$ FamilyGoldenFixture
        +families() Map~String,Family~
    }
    class Family {
        +modelRef() String
        +revision() String
        +cases() List~CaseData~
    }
    class CaseData {
        +caseId
        +text
        +decoded
        +int[] tokens
    }
    class EdgeParityGolden {
        <<final>>
        +load(Class owner, String resource, String generator)$ EdgeParityGolden
        +texts() List~String~
        +models() List~Model~
    }
    class Model {
        +name() String
        +revision
        +cases List~Expected~
    }
    class Expected {
        int[] ids
        int[] specials
        String decoded
    }
    class FamilyTestSpecs {
        <<final>>
        families of generate_ground_truth.py
    }
    FamilyGoldenFixture ..> FixtureJsonLoader
    EdgeParityGolden ..> FixtureJsonLoader
    FamilyGoldenFixture *-- Family
    FamilyGoldenFixture *-- CaseData
    Family o-- CaseData
    EdgeParityGolden *-- Model
    EdgeParityGolden *-- Expected
    Model o-- Expected
    FamilyTestSpecs ..> FamilyGoldenFixture : must be covered
```

* `ground_truth_model_families.json` — per model family, test texts with expected ids and
  decoding (`toknroll-benchmarks/generate_ground_truth.py`).
* `hf_edge_parity.json` / `gguf_edge_parity.json` — edge texts × models, each with plain-encode ids,
  `Specials.encode` ids and (HF) decoded text.
* `FixtureJsonLoader` reads from the test classpath, else from the repository's `test-fixtures`,
  with a "regenerate with …" message when missing.

## 2. Harnesses

```mermaid
classDiagram
    direction LR
    class TokenizerParityHarness {
        <<final>>
        +loadCorpusBytes(Path)$ byte[]
        +runParity(label, Path golden, byte[] corpus, chunkSize, maxChunks, encode, countTokens, decode, tokenSurface)$
    }
    class TokenizerInvariantHarness {
        <<final>>
        +runSmokeChecks(label, inputs, encode, EncodeIntoFn, countTokens, decode, decodeBytes, countBytes, DecodeBytesIntoFn)$
        +runSlicingInvariants(label, inputs, encode, EncodeSliceFn, countTokens, CountTokensSliceFn)$
    }
    class EncodeIntoFn {
        <<interface>>
        +encodeInto(String) int[]
    }
    class DecodeBytesIntoFn {
        <<interface>>
        +decodeBytesInto(int[] tokens, int from, ByteBuffer out) int
    }
    class EncodeSliceFn {
        <<interface>>
        +encodeSlice(CharSequence, int, int) int[]
    }
    class CountTokensSliceFn {
        <<interface>>
        +countTokensSlice(CharSequence, int, int) int
    }
    TokenizerInvariantHarness *-- EncodeIntoFn
    TokenizerInvariantHarness *-- DecodeBytesIntoFn
    TokenizerInvariantHarness *-- EncodeSliceFn
    TokenizerInvariantHarness *-- CountTokensSliceFn
```

* **Parity**: a corpus (enwik8) is cut into chunks identified by hash; each chunk's ids must equal
  the golden row's tokens, `countTokens` must agree and decoding must round-trip.
* **Invariants** (`runSmokeChecks`): determinism, count = length of encode, `encodeInto` /
  `decodeBytes` / `countBytes` / `decodeBytesInto` agreeing with their plain counterparts,
  round-trip on small strings. `runSlicingInvariants` checks the range overloads
  (`encodeInto(text, start, end, out)`, `countTokens(text, start, end)`) on the full range.

## 3. Corpora, tiktoken files and paths

```mermaid
classDiagram
    direction LR
    class Enwik8Corpus {
        <<final>>
        +data() byte[]
        fetched once into the corpus cache
    }
    class TestCorpora {
        <<final>>
        static test strings
    }
    class TiktokenFiles {
        <<final>>
        +loadMergeableRanks(String blobPath, String sha)$ Map
        +loadMergeableRanks(BufferedReader)$ Map
    }
    class FileCache {
        <<package>>
        +readFileCached(String blobPath, String expectedHash)$ byte[]
    }
    class TestCachePaths {
        <<final>>
        +cacheRoot()$ Path
        +corpusDir()$ Path
        +resolveUnderTestArtifacts(String, String...)$ Path
    }
    class GoldenFixturePaths {
        +resolveGoldenEnwik8Dir()$ Path
    }
    class GitRepoPaths {
        +findGitRoot()$ Path
        +findGitRoot(Path)$ Path
    }
    class TestSystemProperties {
        property names
    }
    Enwik8Corpus ..> TestCachePaths : corpusDir
    TiktokenFiles ..> FileCache : download + SHA check
    GoldenFixturePaths ..> GitRepoPaths
    GoldenFixturePaths ..> TestCachePaths
```
