# libjinfer

jinfer as a shared library with a C API: `include/jinfer.h` is the contract, every language binds to it.
The Python package in `python/` is the first binding.

```c
jinfer_model* model = jinfer_model_load("LiquidAI/LFM2.5-350M-GGUF:Q8_0", NULL);
jinfer_stream* s = jinfer_chat(model, "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}");
while (jinfer_stream_step(s) == JINFER_OK) printf("%s", jinfer_stream_text(s));
printf("%s\n", jinfer_stream_result(s));   /* the OpenAI chat.completion response */
jinfer_stream_free(s);
jinfer_model_free(model);
```

The header's opening comment states the whole contract: errors, buffers, ownership, threads, stability.
In short: pointer-returning functions return NULL and int functions a negative code, `jinfer_last_error(&code)` has the message and the code;
`(out, capacity)` buffers follow `snprintf`; everything returned is borrowed until the next call on that handle;
a model is reference-counted by its states and streams; any thread may call, one thread at a time per state or stream.

## Build

GraalVM 25.0.4 or later (`native-image`; `JAVA_HOME` picks it, GraalVM CE for the binaries we ship), a C compiler and Maven. From this directory:

```sh
make libjinfer            # -O3 -march=x86-64-v3 (GraalVM's default on x86-64, AVX2), what ships -> target/libjinfer.so; MARCH=x86-64-v4 for AVX-512
make libjinfer QUICK=1    # -Ob -march=native: links and runs in seconds, kernels unoptimized, never measure it
make check                # versions agree, header vs exports, the C example, the Python test (cached models)
make qa                   # threads, lifetime at interpreter exit, soak, offline, tools, unicode
make python               # stage the library into the Python package
make wheel                # the platform-tagged wheel -> target/wheel/
make check-wheel          # the wheel as a user gets it: twine and auditwheel, a fresh venv, pip install, the test from outside the checkout
make publish              # twine upload to PyPI, after check-wheel; TWINE_REPOSITORY=testpypi rehearses
make dist                 # release bundle: headers, library, README, LICENSE, wheel -> target/dist/
```

`make libjinfer` from the repository root does the same.

## How it is built

The Java side is `com.qxotic.jinfer.libjinfer`: one `@CEntryPoint` per header function (`jinfer__*`, isolate thread first),
a `Model` of whatever kind the file is, a `State` for the raw door and a `Stream` for every generation.
Tokenization is toknroll's API, `toknroll.h` in `toknroll/toknroll-native`: libjinfer carries its `toknroll_*` entry points too, and `jinfer_model_tokenizer` hands out a model's tokenizer as a `toknroll_tokenizer`.
`src/main/c/jinfer_shim.c` is the C face: it forwards to the entry points through the runtime `toknroll_isolate.c` starts at `jinfer_configure` or at the first call, one for the `jinfer_*` and `toknroll_*` symbols alike, and holds a log callback installed before that.
GraalVM links shared libraries with a version script that names only its own entry points,
so `src/main/c/jinfer-cc.sh` wraps the C compiler and adds `jinfer_*` to that list at the link step.

`jinfer_model_options` is read through the header itself (`Options.Directives`), so the layout cannot drift.

The chat dialect is jinfer-server's own: `OpenAiChat` is the public seam the server exposes, so the library and
`/v1/chat/completions` lower and answer identically. Downloads run through `curl` on the PATH (the `fetch` module's process transport), so the image carries no TLS stack of its own; local models need nothing. Download progress is jinfer-hub's: a board on stderr at a terminal, log lines otherwise. libjam is linked into the image (`StaticJam`, jam-native's `jam_static` archive linked
whole, its symbols exported by the wrapper), so nothing is extracted at run time.
`JINFER_CC_LOG=<file> make libjinfer` records every compiler and linker line the build ran.

The image is built for one ISA level, `-march`, and `VectorWidth` pins the kernels' Vector API width to that level
from the target CPU features the builder reports: left to its default, the width would be the build host's, and an AVX2
image built on an AVX-512 machine would carry 512-bit species and emulate them, a hundred times slower.
x86-64-v3 is the floor; below it the kernels are not usable.
