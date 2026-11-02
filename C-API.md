# C API convention

libjam, libtoknroll and libjinfer follow one convention, although they face opposite ways.
libjam is C that Java calls; libtoknroll and libjinfer are Java that C calls.
A caller who knows one of the three knows the other two.

The goals, in order: correct by default, hard to misuse, simple, portable.
The platforms are Windows x86-64, Linux x86-64 and aarch64, and macOS aarch64, all 64-bit.
The libraries are in alpha: until 1.0 any release may break them, and the versioning rules below are practiced, not enforced.
From 1.0 they are enforced, and compatibility is checked on every release (see Versions).

## Names

Every exported name starts with the library's prefix: `jam_`, `toknroll_`, `jinfer_` for functions and types, `JAM_`, `TOKNROLL_`, `JINFER_` for macros and constants.
A handle is an opaque struct, `typedef struct jinfer_model jinfer_model;`, and its functions are named `prefix_noun_verb`.
A handle is made by `_new` (from arguments) or `_load` (from a file, path or hub reference), and released by `_free`.
`_free(NULL)` does nothing.
There is no `_create`, `_destroy` or `_release`.

```c
jinfer_model* jinfer_model_load(const char* ref, const jinfer_model_options* options);
void          jinfer_model_free(jinfer_model* model);
jam_context*  jam_context_new(const jam_context_options* options);
void          jam_context_free(jam_context* context);
```

A function that does not change its handle takes it `const`.

## Types at the boundary

Only these types cross the boundary:

| Type | Used for |
|---|---|
| `int32_t` | element counts, indices, positions, enumerated values, statuses |
| `int64_t` | byte sizes the library returns, and offsets that may pass 2^31 |
| `size_t` | byte lengths the caller passes in (`text`, `length`) |
| `uint32_t` | flag words |
| `float` | samples, logits, embeddings |
| `const char*` | UTF-8 text, NUL-terminated unless a length comes with it |
| pointers to the above, opaque handles, function pointers | |

`int`, `long`, `unsigned`, `bool`, enum types, structs passed by value and varargs never appear in a signature.
`int` and `long` change size between compilers, `bool` and enum types are a size other languages guess, and a struct by value has no portable calling convention.
Named constants are still welcome, as `#define` or as an anonymous `enum { ... }` with every value spelled out; the parameter that takes them is `int32_t`.
Parameters come in one order: the handle, the inputs, the flags, then the outputs.
A pointer comes before its count (`text, length`, `ids, count`, `out, capacity`), as in `memcpy` and `snprintf`, `jinfer_configure(options, count)` included.
A pointer with a count may be NULL when the count is 0.
Products of counts (`m * k`, `count * dimension`) are computed in 64 bits inside the library.

## Flags

A function that may grow options takes a `uint32_t flags` word, never a `bool`.
Each flag is named after its function, `JAM_SHARE_IMMUTABLE`, `TOKNROLL_ENCODE_SPECIALS`, and the call site reads as a sentence.
A bit the library does not define is refused with `EINVAL`, so it stays free to assign later.

## Results and errors

A function returns its result.
A count, index or position is a non-negative `int32_t`, a byte size a non-negative `int64_t`, and a negative value is an error.
A constructor returns the handle, or NULL on failure.
Only a function that cannot fail (`_free`, `_cancel`) returns `void`.

The three libraries share one table of codes, each with its prefix, and a header defines the ones its library can return:

| Code | Value | The caller |
|---|---|---|
| `OK` | 0 | carries on |
| `ERROR` | -1 | gives up and shows the message |
| `EINVAL` | -2 | fixes its own call: a NULL, a bad size, a flag that does not exist |
| `ENOTFOUND` | -3 | reports a missing file, model or token to the user |
| `EUNSUPPORTED` | -4 | falls back: valid request, nothing here can do it |
| `ECANCELLED` | -5 | stops; it asked for this |
| `EBUSY` | -6 | retries, or uses another handle |
| `ENOMEM` | -7 | frees something, or gives up |

A positive status exists only where a function documents one, such as `JINFER_DONE` for a finished stream.
A caller treats any negative code it does not know as `ERROR`, so codes can be added.
A shape the library refuses is `EINVAL`; `EUNSUPPORTED` is only for a valid request it cannot serve, because callers fall back on it.
A hint the library ignores (`jam_share` on a CPU context) returns `EUNSUPPORTED` and is not an error.

The message behind the last result is per thread:

```c
const char* prefix_last_error(int32_t* code);   /* "" and OK after a success; *code unless NULL */
```

Every call sets it, a success clears it, and the string stays valid until the thread's next call into the library.
A library never reports through `errno`, never calls `abort` or `exit`, and never prints unless asked to (see Configuration).
No Java or C++ exception crosses the boundary: each entry point catches everything and returns a code.

## Memory and buffers

Memory the caller passes in is borrowed for the call, unless the function says otherwise (`jam_share`).
Memory the library hands out belongs to a handle and is released by that handle's `_free`; a library never returns memory the caller must release with `free()`, which would mix C runtimes on Windows.
A borrowed result (`jinfer_model_info`, `jinfer_stream_text`) stays valid until the next call on the same handle, or its `_free`; each such function says so.
A function that returns a string never returns NULL: `""` stands for none, so a caller can print it as it comes.

A function that fills caller memory takes the buffer and its capacity, and returns the full length the result needs, as `snprintf` does:

```c
int32_t toknroll_encode(const toknroll_tokenizer* t, const char* text, size_t length, uint32_t flags,
                        int32_t* ids, int32_t capacity);                 /* ids needed */
int64_t toknroll_decode(const toknroll_tokenizer* t, const int32_t* ids, int32_t count,
                        char* out, int64_t capacity);                    /* bytes needed, NUL excluded */
```

It writes at most `capacity` elements, and the result is complete when the return is at most `capacity`.
Text is NUL-terminated whenever `capacity` is not 0, so a string is complete when the return is below `capacity`.
`out = NULL, capacity = 0` asks for the size.
A size the library knows up front (`count * dimension`) is answered before any work, so asking costs nothing.

## Options

A function with more than a few optional settings takes a pointer to an options struct.
Every field's zero value means its default, so `jinfer_model_options options = {0};` in C (`= {};` in C++) is a complete initialization, and a NULL pointer means all defaults.
Fields are ordered largest first, so the struct has no padding for a binding to mirror.
Before 1.0 a struct simply changes, since the soname changes with it.
From 1.0 it can only grow: it starts with a `uint32_t size` that the caller sets to `sizeof`, and the library reads the fields that fit and defaults the rest.

## Handles and NULL

NULL never stands for a default object: there is no global context to fall back on.
A NULL, freed or foreign handle returns `EINVAL` (a borrowed string returns `""`, a pointer NULL) wherever the library can tell.
libtoknroll and libjinfer keep their handles in tables and tag each with its library, so a handle from the other library is refused instead of naming a different object.

## Threads, fork and callbacks

Any function may be called from any thread, at any time, unless its comment says otherwise.
The exceptions take one of two forms:
- one call at a time per handle, enforced with `EBUSY` (a jam context, a jinfer state or stream);
- not during another call on the same handle (every `_free`).

After `fork`, a library either works in the child or refuses with `ERROR`; it never hangs.

A callback takes a `void* user` as its last parameter, and is registered together with it.
It runs on the threads its registration names, must return normally (no `longjmp`, no C++ exception), and may call only the functions documented as callback-safe.
Once a call that replaces or clears a callback returns, the old one is not running and is never called again, so its `user` can be freed.

## Configuration

Settings come through the API: an options struct, or `prefix_configure(options, count)` before first use (later it returns `EINVAL`).
One environment variable per library, `PREFIX_DEBUG`, turns on diagnostics on stderr; a library is otherwise silent.
`jinfer_log_set` routes jinfer's log lines to the host instead.
A library reads its environment once, at first use.

## Headers

One header per library, `include/prefix.h`, guarded by `PREFIX_H` and wrapped in `extern "C"` for C++.
It is ASCII, at most 120 columns, compiles without a warning as C99 and C++ (`-Wall -Wextra -pedantic`, clang `-Weverything`), and defines nothing outside its prefix.
It opens with one comment of short sections, in this order, leaving out those that do not apply and adding its own before Stability: Errors, Buffers, Text, Ownership, Threads, Fork, Settings, Stability.
Every function has a comment: what it does, what it returns, its errors, its threading, and ownership when that is not obvious.
The header is the reference; the README and the examples only show it in use, and the examples are compiled in CI.

## Exports

A library exports exactly the functions its header declares, and no other library's.
`PREFIX_API` is `__declspec(dllexport)` while building on Windows (`PREFIX_BUILD`), `__declspec(dllimport)` for a Windows caller, nothing with `PREFIX_STATIC`, and `__attribute__((visibility("default")))` elsewhere, where everything else is built hidden.
`make check-exports` fails when the exported functions differ from the header's.
One exception: on Windows, GraalVM marks its own runtime functions `dllexport`, and nothing can take them back out of libtoknroll's and libjinfer's DLLs, so there the check covers the prefix's names only.

## Versions

```c
#define PREFIX_VERSION_MAJOR 0
#define PREFIX_VERSION_MINOR 4
#define PREFIX_VERSION_PATCH 0
const char* prefix_version(void);   /* "0.4.0", the release the library was built from */
```

The version is the project's release version; `make check-version` keeps the header, the POM and the Python package equal.
The soname is derived from it by the build, never set by hand, so a program built against one ABI refuses to start against another instead of misreading it.

Before 1.0 (alpha), as practice:
- the soname carries the minor version, `libprefix.so.0.4` on Linux and `@rpath/libprefix.0.4.dylib` on macOS;
- any release may break the API or the ABI, and nothing checks it.

From 1.0, enforced:
- the soname carries the major version alone, `libprefix.so.1` and `@rpath/libprefix.1.dylib`;
- a patch release changes no signature, constant, struct or behavior a caller relies on;
- a minor release only adds: new functions, new flags, new error codes, new fields at the end of a struct with a size;
- anything else is a new major version;
- every release is checked against the previous one before it ships: the exported functions, the header's declarations and constants, and the binary layout (`abidiff` on Linux), and the release fails on a break its version does not allow.

## Bindings

Each binding mirrors the header exactly, and is checked against it rather than trusted.
- Java FFM (libjam): descriptors are written with layouts named after the C types (`C_INT32`, `C_SIZE_T`, `C_POINTER`), each under its C prototype as a comment.
- JNI (libjam): the shim includes the header, so the compiler checks every conversion.
- Python ctypes (libtoknroll, libjinfer): `argtypes` and `restype` for every function.
- Each binding's tests pass sizes and pointers with high bits set, which a narrowed type would cut.
- The C sources assert the 64-bit assumptions: `_Static_assert(sizeof(void*) == 8 && sizeof(size_t) == 8, "...")`.
