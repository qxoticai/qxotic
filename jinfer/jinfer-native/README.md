# libjinfer

jinfer as a shared library with a C API: the engine, every ported model architecture and its native kernels compiled into one library, for C, Python and any language with a foreign function interface.
One header, [`include/jinfer.h`](include/jinfer.h), twenty-nine functions: load a GGUF model, chat with the OpenAI request and response shapes, stream the reply, embed, rerank, speak and transcribe, and drive the raw door of states, logits and embeddings.
A model's tokenizer comes out as a `toknroll_tokenizer`, so [`toknroll.h`](../../toknroll/toknroll-native/include/toknroll.h) is the second header, and libjinfer carries its thirteen functions too.
The Python package in [`python/`](python/README.md) is the first binding, and the wheel on PyPI carries the library.

```c
#include "jinfer.h"

jinfer_model* model = jinfer_model_load("LiquidAI/LFM2.5-350M-GGUF:Q8_0", NULL);   /* NULL: jinfer_last_error says why */
jinfer_stream* s = jinfer_chat(model, "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}");
while (jinfer_stream_step(s) == JINFER_OK) printf("%s", jinfer_stream_text(s));
printf("%s\n", jinfer_stream_result(s));   /* the chat.completion response */
jinfer_stream_free(s);
jinfer_model_free(model);
```

[`examples/c/chat.c`](examples/c/chat.c) is the same with its errors handled and a log callback.
It is built and tested on Linux x86-64 and aarch64 (glibc 2.28 or later), macOS 14 or later on Apple silicon, and Windows 10 or later on x86-64.
x86-64 means x86-64-v3, a CPU with AVX2; below it the kernels are not usable.
The models are jinfer's ports, the [Python README](python/README.md#models) lists the families and the [jinfer README](../README.md#supported-architectures) keeps the authoritative table.

## Getting the library

- `make dist` here builds the release bundle, `target/dist/libjinfer-<version>-<platform>.zip`: both headers, the shared library, this README, the license and the wheel.
- `pip install jinfer`: the [wheel](https://pypi.org/project/jinfer/) carries the shared library, both headers and a pkg-config file in the installed package, and `python -m jinfer --cflags --libs` (or `--pkgconfig`) says where, so a C build in a virtual environment needs nothing else.
- Or build it, see below.

## From C

The header's opening comment states the whole contract: errors, buffers, text, ownership, threads, fork, settings, stability.
In short: pointer-returning functions return NULL and int functions a negative code, and `jinfer_last_error` has the message and the code, per thread.
Output buffers `(out, capacity)` follow `snprintf`: the return is the whole result's size, the buffer holds what fits, and a capacity of 0 just counts.
Everything returned is borrowed until the next call on that handle.
A model is reference-counted by its states and streams, so freeing it under a live stream is safe.
Any thread may call, one call at a time per state or stream; a second at once is `JINFER_EBUSY`.
A forked child is refused: the runtime's threads stayed in the parent, so every call there fails with the reason and every free does nothing; start children with spawn or exec.

Linking the shared library:

```sh
cc -Iinclude -I../../toknroll/toknroll-native/include chat.c -L<dir> -ljinfer     # Linux, macOS: the soname is libjinfer.so.0.4, @rpath/libjinfer.0.4.dylib
cl /Iinclude /I..\..\toknroll\toknroll-native\include chat.c /link /LIBPATH:<dir> jinfer.lib   # Windows: the import library beside jinfer.dll
```

On Linux and macOS the library exports the two headers' functions and libjam's kernels, nothing else.
A Windows DLL has no export list, so `jinfer.dll` exports the runtime's symbols too; a program binds by name as on any DLL.
There is no static archive: libjam, the kernel library, is linked into the image whole, and the image is the unit.

## Configuration

The library is a Java runtime compiled ahead of time; it starts at `jinfer_configure` or at the first call, and reads its settings then.

| What | How |
|---|---|
| runtime options | `jinfer_configure` takes them as a java command line would, once, before the first call; `JINFER_OPTIONS` in the environment adds to them and wins, split on whitespace; `-Dtoknroll.*` options reach the models' tokenizers too |
| log lines | `JINFER_LOG` picks them, the `toknroll_*` functions' included: `off`, `error`, `warn` (the default), `info` (downloads) or `debug` (loads, kernels); they go to stderr, or to `jinfer_log_set`'s callback, set once before any other call, NULL for nowhere |
| download progress | shown on stderr whatever `JINFER_LOG` says: a live board when stderr is a terminal, a line at each tenth otherwise; `JINFER_PROGRESS=off`, or `-Djinfer.progress=off`, turns it off entirely, and the download is then a start and a done line at info |
| the model store | `-Djinfer.models=<dir>`, else `JINFER_MODELS`, else the platform cache directory; a file the Hugging Face hub cache holds (`HF_HUB_CACHE`, `HF_HOME`) is read in place |
| offline | `-Djinfer.offline=true` serves the store and never touches the network; `false`, `0`, `off` and `no` mean online, anything else offline |
| threads | `-Djinfer.threads=<n>` sizes the inference pool; the default is the machine's physical cores |
| the prompt cache | `-Djinfer.cache.sessions=<n>` and `-Djinfer.cache.budget=<size>` bound what the engine keeps between turns |
| memory | RAM for the model file, which is read whole, plus its context; `-XX:MaxHeapSize=<size>` caps the runtime's own heap |

Downloads go through `curl` on the PATH, so the image carries no TLS stack of its own; local models need nothing.
An image or an audio file given to a chat is decoded with `ffmpeg` on the PATH, and speech synthesis with Kokoro voices needs `espeak-ng` (or `espeak`); a WAV file for transcription needs nothing.
[C-API.md](../../C-API.md#configuration) has the rules behind the table.

## Build and check

GraalVM 25.0.4 or later (`native-image`; `JAVA_HOME` picks it), a C compiler and Maven.
`make libjinfer` builds at `-O3` for x86-64-v3; the binaries we ship come from Oracle GraalVM, whose compiler adds its inferred profile.
From this directory:

```sh
make libjinfer            # GraalVM Native Image -> target/libjinfer.so; NATIVE_IMAGE_OPTIONS=-march=x86-64-v4 for AVX-512
make libjinfer QUICK=1    # this machine, in seconds: -Ob -march=native, kernels unoptimized, never measure it
make check                # versions agree, headers vs exports, both C suites, the Python test on the cached models
make qa                   # the corner cases: threads, lifetime at interpreter exit, soak, offline, tools, unicode
make python               # stage the library into the Python package
make wheel                # the platform-tagged wheel -> target/wheel/
make check-wheel          # the wheel as a user gets it: twine, a fresh venv, pip install, the test from outside the tree
make dist                 # the release bundle -> target/dist/
```

`make libjinfer` from the repository root does the same.
On Windows, `.\build.ps1` in this directory has the same targets.
The checks need the models they name cached, `make cache-models` at the root fetches them, and the network for the C suites' progress scenarios.
This machine's wheels from a clean export of a ref, and the upload to PyPI, are `make wheels` and `make publish-wheels` at the repository root (`wheels.ps1` on Windows; [RELEASING.md](../../RELEASING.md)).

## Inside

The Java side is `com.qxotic.jinfer.libjinfer`: one `@CEntryPoint` per header function (`jinfer__*`, isolate thread first), a `Model` of whatever kind the file is, a `State` for the raw door and a `Stream` for every generation.
`src/main/c/jinfer_shim.c` is the C face: it forwards to the entry points through the runtime that toknroll-native's `toknroll_isolate.c` starts, one for the `jinfer_*` and `toknroll_*` symbols alike, which hands it the log callback set before that and probes the terminal for the download board.
`jinfer_model_options` is read through the header itself (`Options.Directives`), so the layout cannot drift.
The chat dialect is jinfer-server's own: `OpenAiChat` is the seam the server exposes, so the library and `/v1/chat/completions` lower and answer identically.

GraalVM links shared libraries with an export list that names its own runtime API and entry points, so toknroll-native's `src/main/c/native-cc.sh` wraps the C compiler for both libraries and makes the list the shims' globals and libjam's names at the link step.
The wrapper also gives the library its soname, derived from the version, `libjinfer.so.0.4`; `target/libjinfer.so.0.4` links to the library, and the bundle carries the file under that name.
libjam, the kernel library, is linked whole (jam-native's static archive, the pom's whole-archive line) and loaded from the library's own file: the isolate shim hands the runtime its path as `jam.native.library.path`, so nothing is extracted at run time.
`NATIVE_CC_LOG=<file> make libjinfer` records every compiler and linker line the build ran.

The image is built for one ISA level, `-march`, and `VectorWidth` pins the kernels' Vector API width to that level from the target CPU features the builder reports.
Left to its default, the width would be the build host's, and an AVX2 image built on an AVX-512 machine would carry 512-bit species and emulate them, a hundred times slower.

The image is lean the way libtoknroll's is: curl as a process instead of the JDK's HTTP client and TLS stack, the image's own `System.LoggerFinder` instead of `java.util.logging`, the shim's terminal probes instead of the Foreign Function API, staging files named without `SecureRandom`, each a service provider or a plain code path and never a substitution of a JDK class.
The shipped shared library is 42 MB on Linux and Windows and 39 MB on macOS, most of it the kernels of every architecture at this ISA level.

`python/` is the `jinfer` package, see its [README](python/README.md).
