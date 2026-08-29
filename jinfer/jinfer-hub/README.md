# Jinfer Hub

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

`jinfer-hub` resolves model references into cached local files. It downloads only missing files and
supports Hugging Face and ModelScope repositories.

## Add the library

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.2.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>

<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-hub</artifactId>
</dependency>
```

## The reference grammar

One string identifies a file in a supported model repository:

```text
[host/]owner/repo[@revision][/path][:quant]
```

| Reference | Meaning |
|-----------|---------|
| `unsloth/Qwen3.5-4B-GGUF` | Select the default quant from a Hugging Face repository |
| `unsloth/Qwen3.5-4B-GGUF:Q8_0` | Select a quant explicitly |
| `LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf` | Select a companion file |
| `unsloth/gemma-4-E2B-it-GGUF/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf` | Select a file in a subdirectory |
| `unsloth/Qwen3.5-4B-GGUF@a1b2c3d:Q8_0` | Pin a branch, tag or commit |
| `hf.co/unsloth/Qwen3.5-4B-GGUF:Q8_0` | Name the host explicitly |
| `modelscope.cn/unsloth/Qwen3.5-4B-GGUF:Q8_0` | Resolve from ModelScope |

`host/` is optional and defaults to `hf.co`, which is also the form every resolved reference prints
back as. Name a host to reach another source. `@revision` is optional. `/path` selects an exact
file or subdirectory. `:quant` selects a file by quant name. A reference is not a URL: it has no
scheme, query or `/blob/` browser path.

Three rules keep a host-less reference from swallowing a local path:

- A file that already exists under that name wins, so no working local path changes meaning.
- Anything spelled like a path is a path: a leading `/`, `.` or `~`, a backslash, a drive letter.
- `owner/model.gguf` is a path, because no repository is named after a model file. The
  file-in-repository form has three segments or more, as in `owner/repo/mmproj-F32.gguf`.

A dot in the first segment reserves it for the host table, so a misspelled host is reported rather
than read as an owner.

## Quant shorthand

A reference with no `:quant` selects `Q4_K_M`, matching llama.cpp's default so
the same shorthand selects the same file in both tools. Jinfer's best-supported quant is `Q8_0`, so
examples pin it explicitly. When a repository ships exactly one GGUF, that file is selected without
a quant; otherwise the quant name is matched against the file names in the repository listing.

An invalid reference reports the information needed to correct it. A quant that matches nothing
lists what the repository ships; a quant that matches several files prints a menu to pick from:

```text
no Q8_0 in owner/repo. Available: repo-Q4_K_M.gguf, repo-Q5_K_M.gguf
Q4_K matches 2 files in owner/repo - name the intended one:
  repo-Q4_K_M.gguf
  repo-Q4_K_S.gguf
```

An explicit quant never falls back to a different file.

## Cache layout

Resolution checks the cache before any network request. Files land under the root, one path
component per reference field, so the layout identifies the source:

```text
$JINFER_MODELS/
└── hf.co/
    └── unsloth/
        └── gemma-4-E2B-it-GGUF/
            └── gemma-4-E2B-it-Q8_0.gguf
```

The root is `-Djinfer.models` or `JINFER_MODELS`, falling back to the platform cache
(`~/.cache/jinfer`, `~/Library/Caches/jinfer` or `%LOCALAPPDATA%\jinfer`). A pinned revision joins
the repository directory as `repo@revision`.

## Downloads

- **Resumable.** Downloads use a sibling `.part` file and chunk map. An interrupted download resumes
  from completed chunks. Files larger than 64 MB use 4 to 8 parallel range requests.
- **Checksum-verified.** The file is verified against the repository's `sha256` before it is
  renamed into place. A mismatch deletes the partial file and restarts the download. The resolver
  never returns an unverified file.
- **Concurrent.** `resolveAll` downloads missing files in parallel. Cached files require no work.
- **Process-safe.** Threads and separate JVMs downloading the same file queue on a lock file
  instead of corrupting each other's partial download.

## Environment

| Variable | Effect |
|----------|--------|
| `JINFER_MODELS` | Moves the cache root (property: `-Djinfer.models`) |
| `JINFER_OFFLINE=1` | Forbids network access; anything uncached fails fast (property: `-Djinfer.offline`) |
| `JINFER_DOWNLOAD_THREADS` | Sets parallel range connections per file, 4 to 8 by CPU count (property: `-Djinfer.downloadThreads`) |
| `JINFER_SKIP_DISK_CHECK=1` | Skips the free-space check (some network mounts report no free space) |
| `HF_TOKEN` | Authenticates access to gated Hugging Face repositories |
| `HF_ENDPOINT` | Points Hugging Face at a mirror |
| `MODELSCOPE_API_TOKEN` | Authenticates access to gated ModelScope repositories |
| `MODELSCOPE_ENDPOINT` | Points ModelScope at a mirror |

The Hugging Face token may also be read from `$HF_TOKEN_PATH`, then `$HF_HOME/token`.

Warm resolution makes no request. A cached reference resolves locally. In offline mode, a missing
file reports its expected cache path:

```text
hf.co/unsloth/gemma-4-E2B-it-GGUF is not cached at ~/.cache/jinfer/... and JINFER_OFFLINE forbids downloading
```

`find(...)` does not access the network. It also checks the Hugging Face hub cache, so files fetched
by `hf download` or `llama-server -hf` are reused.

## From Java

`ModelStore.standard()` is the entry point:

```java
String ref = "LiquidAI/LFM2.5-350M-GGUF:Q8_0";

Path model = ModelStore.standard().resolve(ref);

List<Path> files = ModelStore.standard().resolveAll(List.of(
        ref,
        "LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf"));

Optional<Path> hit = ModelStore.standard().find(ref);            // cached file only, no network
List<ModelStore.Cached> cached = ModelStore.standard().cached();  // list what the cache holds
```

`ModelStore.isRef(...)` and `requireRef(...)` implement the grammar; `evict(...)` removes a cached
file. The CLI's `jinfer pull` and `jinfer list` run over the same store.

Resolution happens before loading, and inference never fetches: the path returned here is what
`Models.load(path, arena)` and the framework builders (`model(...)` / `companion(...)`) consume.

## A plain URL is not a model reference

A URL does not provide a repository listing, quant, revision or published checksum. Jinfer can
validate only the content length reported by the server and warns when downloading a URL.
`https://example.org/models/x.gguf` is cached at `<root>/example.org/models/x.gguf`. Model builders
accept a reference through `model(...)` or a local file through `modelPath(...)`. Download a URL
first, then pass its local path.

## See also

- [Models from a hub](https://qxotic.ai/docs/jinfer#models-from-a-hub): how the framework builders accept refs and
  companions
- [CLI](../README.md#cli-and-server): `jinfer pull` and `jinfer list` on the same store
