# libtoknroll

Tok'n'Roll as a shared library with a C API, for Python, C and any language with a foreign function interface.
One header, [`include/toknroll.h`](include/toknroll.h), twelve functions: load a tokenizer from a `tokenizer.json`, a GGUF file or a hub reference, encode, decode, encode a batch on all cores, and look the vocabulary up, token flags included; `toknroll_configure` passes the runtime options a java command line would, before first use.
Remote sources fetch tokenizer metadata only, never weights.

```c
toknroll_tokenizer* t = toknroll_tokenizer_load("Qwen/Qwen3-0.6B");
const char* text = "Hello, world!";
int32_t n = toknroll_encode(t, text, strlen(text), 0, NULL, 0);   /* capacity 0 counts */
int32_t* ids = malloc(n * sizeof *ids);
toknroll_encode(t, text, strlen(text), 0, ids, n);
char back[256];
toknroll_decode(t, ids, n, back, sizeof back);
toknroll_tokenizer_free(t);
```

The contract is the one `jinfer.h` keeps: pointer-returning functions return NULL and int functions a negative code, `toknroll_last_error` has the message and the code per thread; `(out, capacity)` buffers follow snprintf; strings are UTF-8 and borrowed, text with its length in bytes; any thread may call and a tokenizer may be shared.
`libjinfer` carries the same entry points, so `jinfer_model_tokenizer` hands out a model's tokenizer as a `toknroll_tokenizer` and one header serves both libraries.

## Build and check

GraalVM 25.0.4 or later (`native-image`; `JAVA_HOME` picks it), a C compiler and Maven.
With Oracle GraalVM, `make libtoknroll` first records a profile (PGO) on enwik8 and a tokenizer per fast path, fetched once, and the binaries we ship are built that way; GraalVM CE builds without one.
From this directory:

```sh
make libtoknroll            # GraalVM Native Image -> target/libtoknroll.so and target/libtoknroll.a
make check                  # versions agree, exports vs header, the C example against both, the Python test
make python                 # stage the shared library into python/toknroll/lib/<platform>/
make wheel                  # the platform-tagged wheel -> target/wheel/
make check-wheel            # the wheel as a user gets it: twine and auditwheel, a fresh venv, pip install, the suite from outside the checkout
make dist                   # release bundle: header, both libraries, README, LICENSE, wheel -> target/dist/
```

On Windows, `.\build.ps1` in this directory has the same targets.
This machine's two wheels from a clean export of a ref, and the upload to PyPI, are `make wheels` and `make publish-wheels` at the root (`wheels.ps1` on Windows; RELEASING.md).

Hub references are fetched through `curl` on the PATH (the `fetch` module's process transport), so the image carries no TLS stack of its own; local files need nothing.
Without a profile, at `-O2`, the builder's default, the shared library is 17 MB and the archive 18 MB; with the JDK's HTTPS stack inside it was 32 MB.
The size follows the code: `-O3` is 12 MB larger for 18% more encode throughput, `-Os` 6 MB smaller for 19% less, and the floor, one entry point and the runtime, is 8 MB.
What ships is the PGO build, which Oracle GraalVM compiles at `-O3` but only on the paths the profile ran, so it stays near the `-O2` size: 16 MB on Windows.

The shared library's soname follows the version, `libtoknroll.so.0.4` (`@rpath/libtoknroll.0.4.dylib` on a Mac); the build links `target/libtoknroll.so.0.4` to the library, and the release bundle carries the file under that name.
The shared library exports the header's twelve functions and nothing else, neither GraalVM's `graal_*` runtime API nor the entry points behind the header.
`libtoknroll.a` is the same image as one relocatable object with every symbol but the header's twelve made local: nothing of the runtime behind it is visible to the program that links it, and it needs only the system libraries the image does, `-lz -ldl -lpthread -lm`, then `-lrt` on Linux or `-framework Foundation` on macOS.
A program that links it defines `TOKNROLL_STATIC` before including the header, so `TOKNROLL_API` asks for no import.
It is for programs that want no shared library; a GraalVM host image cannot link it (two runtimes in one image), so libjinfer compiles the entry points into its own image instead.

`src/main/c/toknroll_shim.c` forwards each header function to the `toknroll__*` entry points in `Native.java`; `toknroll_isolate.c` owns the runtime: it starts at `toknroll_configure` or at the first call, with configure's options and then `TOKNROLL_OPTIONS` from the environment, and libjinfer compiles both files into its own image so one runtime serves both APIs.
`python/` is the `toknroll` package, see its [README](python/README.md); `python/tests` is its suite, `python3 -m unittest discover -s tests -t tests` from `python/`.
