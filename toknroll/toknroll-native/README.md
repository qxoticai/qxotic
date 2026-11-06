# libtoknroll

Tok'n'Roll as a shared library with a C API, for C, Python and any language with a foreign function interface.
One header, [`include/toknroll.h`](include/toknroll.h), thirteen functions: load a tokenizer from a `tokenizer.json`, a GGUF file or a hub reference, encode, decode, encode a batch on all cores, and look the vocabulary up, token flags included.
The same ids as the model's own tokenizer, or the load is refused with the reason; remote sources fetch tokenizer metadata only, never weights.
The Python package in [`python/`](python/README.md) is the first binding, and the wheel on PyPI carries the library.

```c
#include "toknroll.h"

toknroll_tokenizer* t = toknroll_tokenizer_load("Qwen/Qwen3-0.6B");   /* NULL: toknroll_last_error says why */
const char* text = "Hello, world!";
int32_t n = toknroll_encode(t, text, strlen(text), 0, NULL, 0);        /* capacity 0 counts */
int32_t* ids = malloc(n * sizeof *ids);
toknroll_encode(t, text, strlen(text), 0, ids, n);
char back[256];
toknroll_decode(t, ids, n, 0, back, sizeof back, NULL);
toknroll_tokenizer_free(t);
```

[`examples/c/tokenize.c`](examples/c/tokenize.c) is the same with its errors handled.
It is built and tested on Linux x86-64 and aarch64 (glibc 2.28 or later), macOS 14 or later on Apple silicon, and Windows 10 or later on x86-64.
x86-64 means x86-64-v3, a CPU with AVX2.

## Getting the library

- The release zip, `libtoknroll-<version>-<platform>.zip`, an asset of the release on GitHub, built here by `make dist`: a prefix tree, `include/toknroll.h`, `lib/` with the shared library under its soname, the linker's name linked to it, the static archive and `lib/pkgconfig/toknroll.pc`, this README and the license. Unzipped into `/usr/local` it is an installed library pkg-config finds with no setup; elsewhere, `PKG_CONFIG_PATH=<prefix>/lib/pkgconfig`. On Windows the DLL sits in `bin\` and the import library in `lib\`.
- `pip install toknroll`: the [wheel](https://pypi.org/project/toknroll/) carries the shared library, `toknroll.h` and a pkg-config file in the installed package, and `toknroll-config --cflags --libs`, its console script, says where (`--pkgconfigdir` for pkg-config), so a C build in a virtual environment needs nothing else.
- Or build it, see below.

## From C

The contract is [C-API.md](../../C-API.md)'s, restated in the header's opening comment.
In short: pointer-returning functions return NULL and int functions a negative code, and `toknroll_last_error` has the message and the code, per thread.
Output buffers `(out, capacity)` follow `snprintf`: the return is the whole result's size, the buffer holds what fits, and a capacity of 0 just counts.
Strings are UTF-8 and borrowed, text comes with its length in bytes.
Any thread may call, and a tokenizer may be shared; a fork keeps every tokenizer.

Linking the shared library:

```sh
cc -Iinclude tokenize.c -L<dir> -ltoknroll            # Linux, macOS: the soname is libtoknroll.so.0.4.0, @rpath/libtoknroll.0.4.0.dylib
cl /Iinclude tokenize.c /link /LIBPATH:<dir> toknroll.lib   # Windows: the import library beside toknroll.dll
```

On Linux and macOS the library exports the header's thirteen functions and nothing else, neither GraalVM's runtime API nor the entry points behind the header.
A Windows DLL has no export list, so `toknroll.dll` exports the runtime's symbols too; a program binds the thirteen by name as on any DLL.

`libtoknroll.a` is the same image as one relocatable object with every symbol but the thirteen made local, for a program that wants no shared library.
It needs the system libraries the image does, `-lz -ldl -lpthread -lm`, then `-lrt` on Linux or `-framework Foundation` on macOS, and the program defines `TOKNROLL_STATIC` before including the header, so `TOKNROLL_API` asks for no import.
A GraalVM host image cannot link it, two runtimes in one image, which is why libjinfer compiles these entry points into its own image instead.

`libjinfer` carries the same thirteen entry points, so `jinfer_model_tokenizer` hands out a model's tokenizer as a `toknroll_tokenizer`, and one header serves both libraries.

## Configuration

The library is a Java runtime compiled ahead of time; it starts at `toknroll_configure` or at the first call, and reads its settings then.

| What | How |
|---|---|
| runtime options | `toknroll_configure` takes them as a java command line would, once, before the first call; `TOKNROLL_OPTIONS` in the environment adds to them and wins, split on whitespace |
| log lines | `TOKNROLL_LOG` picks them: `off`, `error`, `warn` (the default), `info` (downloads) or `debug`; they go to stderr, or to `toknroll_log_set`'s callback, set once before any other call, NULL for nowhere |
| download progress | shown on stderr whatever `TOKNROLL_LOG` says: a live board when stderr is a terminal, a line at each tenth otherwise; `TOKNROLL_PROGRESS=off`, or `-Dtoknroll.progress=off`, turns it off entirely, and the download is then a start and a done line at info |
| the cache | `-Dtoknroll.cache.root=<dir>`, else `TOKNROLL_CACHE_ROOT`, else jinfer's store under the platform cache directory; a file the Hugging Face hub cache holds (`HF_HUB_CACHE`, `HF_HOME`) is read in place |
| a rate-limited hub | waited out as it asks, a listing for a minute at most and a download for half an hour, the first wait logged at warn; past that the 429 is the error |
| offline | `-Dtoknroll.offline=true` serves the cache and never touches the network; `false`, `0`, `off` and `no` mean online, anything else offline |
| GGUF pre-tokenizers | `-Dtoknroll.gguf.pre.<name>=alias:<known>` names a scheme the file does not |
| threads, heap | `-XX:ActiveProcessorCount=<n>` caps what `toknroll_encode_batch` spreads over, `-XX:MaxHeapSize=<size>` the heap, about 256 MB for a 262k-token vocabulary; the runtime reserves the address space its heap may grow to, so under a virtual memory limit (`ulimit -v`) cap the heap, or the start fails saying so |

Hub references are fetched through `curl` on the PATH, so the image carries no TLS stack of its own; local files need nothing.
[C-API.md](../../C-API.md#configuration) has the rules behind the table.

## Build and check

GraalVM 25.0.4 or later (`native-image`; `JAVA_HOME` picks it), a C compiler and Maven.
With Oracle GraalVM, `make libtoknroll` first records a profile on enwik8 and a tokenizer per fast path, fetched once, and compiles with it at `-O3`; that is the build we ship.
GraalVM CE builds without a profile.
From this directory:

```sh
make libtoknroll            # GraalVM Native Image -> target/libtoknroll.so and target/libtoknroll.a
make libtoknroll QUICK=1    # this machine, in seconds: -Ob -march=native, never measure it
make check                  # versions agree, exports vs header, the C suite against both libraries, valgrind, the Python suite
make python                 # stage the shared library into python/toknroll/lib/<platform>/
make wheel                  # the platform-tagged wheel -> target/wheel/
make check-wheel            # the wheel as a user gets it: twine, a fresh venv, pip install, the suite from outside the tree
make dist                   # the release zip, a prefix tree -> target/dist/
make check-dist             # the zip unzipped to a prefix of its own, the C example built with pkg-config alone
make publish-dist TAG=<tag> # the zip and its checksum as assets of that GitHub release
```

On Windows, `.\build.ps1` in this directory has the same targets.
The C suite downloads one tokenizer from the hub for its progress scenarios, so `make check` needs the network, as the Python suite does.
This machine's wheels from a clean export of a ref, and the upload to PyPI, are `make wheels` and `make publish-wheels` at the repository root (`wheels.ps1` on Windows; [RELEASING.md](../../RELEASING.md)).

## Inside

`src/main/c/toknroll_shim.c` forwards each header function to the `toknroll__*` entry points in `Native.java`.
`toknroll_isolate.c` owns the runtime: it starts it with configure's options and then `TOKNROLL_OPTIONS`, hands it the log callback, gates calls across a fork, and probes the terminal for the download board, so the image needs no Foreign Function API.
libjinfer compiles both files into its own image, and one runtime serves both APIs there.

The build links the shared library with an export list of the header's functions and gives it its soname, derived from the version, `libtoknroll.so.0.4.0` while the major is 0, as the ABI is not guaranteed between 0.x releases; `target/libtoknroll.so.0.4.0` links to the library, and the bundle carries the file under that name.

The image is lean on purpose.
Downloads go through `curl` as a process, not the JDK's HTTP client, which with its TLS stack is 18 MB; logging is the image's own `System.LoggerFinder`, not `java.util.logging`; the hub's terminal probes come from the shim, not the Foreign Function API; staging files are named without `SecureRandom`.
Each of those is a service provider or a plain code path, never a substitution of a JDK class.
The shipped shared library is 13.8 MB on every platform, the static archive 15.2 MB; the floor, one entry point and the runtime, is 5 MB.
`-Os` would save 3 MB for 9% less single-threaded encode throughput, so it is not used.

`python/` is the `toknroll` package, see its [README](python/README.md); its suite is `python3 -m unittest discover -s tests -t tests` from `python/`.
