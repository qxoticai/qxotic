<h1 align="center">jinfer Hub</h1>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
</p>

`jinfer-hub` resolves Hugging Face and ModelScope references, `owner/repo[:quant]`, into verified, cached local files: the names model cards already use, the checksums the hubs publish, and a cache laid out by reference.
A local path is used as it is, and a URL is refused; the error shows the reference to use instead.

## Add the library

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.4.0</version>
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

`host/` is optional and defaults to `hf.co`, which is also the form every resolved reference prints back as.
Name a host to reach another source.
`@revision` is optional, `/path` selects an exact file or subdirectory, and `:quant` selects a file by quant name.

Three rules keep a host-less reference from swallowing a local path:

- A file that already exists under that name wins, so no working local path changes meaning.
- Anything spelled like a path is a path: a leading `/`, `.` or `~`, a backslash, a drive letter.
- `owner/model.gguf` is a path, because no repository is named after a model file.
  The file-in-repository form has three segments or more, as in `owner/repo/mmproj-F32.gguf`.

A dot in the first segment reserves it for the host table, so a misspelled host is reported rather than read as an owner.

Owner and repository match the cache in any case, as both hubs read repository ids case-insensitively.
`LiquidAI/d1-2B-gguf` and `LiquidAI/d1-2B-GGUF` share one cache folder and one download, online or offline.

## Quant shorthand

A reference with no `:quant` selects `Q4_K_M`, matching llama.cpp's default, so the same shorthand selects the same file in both tools.
jinfer's best-supported quant is `Q8_0`, so the examples pin it explicitly.
When a repository ships exactly one GGUF, that file is selected without a quant.
Otherwise the quant selects the file named `<model>-<QUANT>.gguf`, the quant being the whole rest of the name after the model's, which is the repository's name without `-GGUF`.
`:Q4_0` is `LFM2.5-350M-Q4_0.gguf` and never `LFM2.5-350M-QAD-Q4_0.gguf`, which is `:QAD-Q4_0`; `:Q4_K` names neither `Q4_K_M` nor `Q4_K_XL`.
The cache follows the same rule, so a downloaded file never answers for a quant that is not downloaded.

An invalid reference reports the information needed to correct it.
A quant that matches nothing lists what the repository ships; a quant that several models in one repository share prints a menu to pick from:

```text
no Q4 in LiquidAI/LFM2.5-350M-GGUF; available: BF16, F16, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, QAD-Q4_0
Q8_0 matches 2 files in acme/models - name the one you want:
  hf.co/acme/models/a-Q8_0.gguf   1.2 GB
  hf.co/acme/models/b-Q8_0.gguf   2.4 GB
```

An explicit quant never falls back to a different file.

## Cache layout

Resolution checks the cache before any network request.
With the default root, Hugging Face files are written into the Hugging Face cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`) in its own layout, so `hf download` and llama.cpp see the same bytes.
Everything else, and every file under an explicit root, lands under the root, one path component per reference field, so the layout identifies the source:

```text
$JINFER_MODELS/
└── hf.co/
    └── unsloth/
        └── gemma-4-E2B-it-GGUF/
            └── gemma-4-E2B-it-Q8_0.gguf
```

The root is `-Djinfer.models` or `JINFER_MODELS`, falling back to the platform cache (`~/.cache/jinfer`, `~/Library/Caches/jinfer` or `%LOCALAPPDATA%\jinfer`).
A pinned revision joins the repository directory as `repo@revision`.

## Downloads

- **Resumable.** Interrupted downloads can resume when the server supports it.
- **Verified.** Downloads are checked against repository-provided checksums when available.
- **Concurrent.** Download several files in parallel with `resolveAll` or `pullAll`.
- **Safe updates.** Failed updates leave the previously usable model available.

## Use, update, repair

| Operation | Behavior |
|---|---|
| `resolve` / `resolveAll` | Use cached files without network access; download on a miss |
| `pullAll(refs, false)` / `jinfer pull` | Check upstream; reuse matching content, download changed or missing files |
| `pullAll(refs, true)` / `jinfer pull --force` | Download again, even if cached |

Existing local paths are returned as-is, including with force enabled.

## Environment

| Variable | Effect |
|----------|--------|
| `JINFER_MODELS` | Moves the cache root (property: `-Djinfer.models`) |
| `JINFER_OFFLINE` | `1`, `true`, `on`, `yes`: offline; `0`, `false`, `off`, `no`: online (case-insensitive). `-Djinfer.offline` takes the same values and wins over the variable when set. |
| `JINFER_PROGRESS=off` | Hides download progress (property: `-Djinfer.progress`) |
| `JINFER_DOWNLOAD_THREADS` | Sets parallel download connections per file (property: `-Djinfer.downloadThreads`) |
| `JINFER_SKIP_DISK_CHECK=1` | Skips the free-space check (some network mounts report no free space) |
| `HF_TOKEN` | Authenticates access to gated Hugging Face repositories |
| `HF_ENDPOINT` | Points Hugging Face at a mirror |
| `MODELSCOPE_API_TOKEN` | Authenticates access to gated ModelScope repositories |
| `MODELSCOPE_ENDPOINT` | Points ModelScope at a mirror |

The Hugging Face token may also be read from `$HF_TOKEN_PATH`, then `$HF_HOME/token`.

Update checks and forced downloads require network access.
An exact cached file pinned to a full commit can be pulled offline.

A cached reference resolves without a request.
In offline mode, a missing file reports its expected cache path:

```text
hf.co/unsloth/gemma-4-E2B-it-GGUF is not cached in ~/.cache/jinfer/..., and JINFER_OFFLINE forbids downloading
```

`find(...)` does not access the network.
It also checks the Hugging Face hub cache, so files fetched by `hf download` or `llama-server -hf` are reused.

A source that names nothing throws `ModelNotFoundException`, an `IllegalArgumentException`: a missing local file, a repository that does not exist or is gated, a file or quant it lacks, or, offline, a miss.
`source()` is the path or ref; a malformed or ambiguous ref stays a plain `IllegalArgumentException`.

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

To update explicitly or repair a cached file:

```java
List<Path> updated = ModelStore.standard().pullAll(List.of(ref), false);
List<Path> repaired = ModelStore.standard().pullAll(List.of(ref), true);
```

`ModelStore.isRef(...)` and `requireRef(...)` implement the grammar; `find(...)` is the offline lookup; `remove(...)` explicitly deletes the one cached file `find(...)` returns; `detached()` lists the Hugging Face cache versions no branch or tag names, with what removing each frees.
The CLI's `jinfer pull`, `list`, `find`, `remove` and `prune` run over the same store.

Resolution happens before loading, and inference never fetches: the path returned here is what `Models.load(path, arena)` and the framework builders (`model(...)` / `companion(...)`) consume.

## A URL is not a model reference

A link to a model's page, pasted from the browser, is refused with the exact reference to use instead, and the publisher it names, to check:

```text
'https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/blob/main/LFM2.5-350M-Q8_0.gguf' is a URL, not a model reference.
Use: hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0
That is LiquidAI's repository on hf.co: check it is the publisher you meant to trust.
```

Any other URL carries no repository, revision or published checksum: download the file and pass its local path.

## See also

- [Models from a hub](https://github.com/qxoticai/qxotic/blob/main/docs/jinfer/index.md#models-from-a-hub): how the framework builders accept refs and
  companions
- [CLI guide](../jinfer-cli/README.md): commands and examples
