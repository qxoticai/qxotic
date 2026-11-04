/* toknroll.h's promise on unloading: once started the library stays loaded, as its threads run its code,
 * so an unload (dlclose, FreeLibrary) succeeds and leaves it in place. A program of its own, not linked
 * against the library: it opens the library itself, by the path it is given.
 *
 *   ./unload_test <libtoknroll.so | toknroll.dll> */
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS   /* fopen, in fixtures.h */
#elif defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE               /* RTLD_NOLOAD */
#endif
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "fixtures.h"
#ifdef _WIN32
#  include <windows.h>
typedef HMODULE library;
static library open_library(const char* path) { return LoadLibraryA(path); }
static int mapped(const char* path) { return GetModuleHandleA(path) != NULL; }
static int close_library(library l) { return FreeLibrary(l) ? 0 : -1; }
typedef void (*any_fn)(void);
static any_fn symbol(library l, const char* name) { return (any_fn) GetProcAddress(l, name); }
#else
#  include <dlfcn.h>
typedef void* library;
static library open_library(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static int mapped(const char* path) {   /* still in this process, without loading it again */
    void* l = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    if (l) dlclose(l);
    return l != NULL;
}
static int close_library(library l) { return dlclose(l); }
typedef void (*any_fn)(void);
/* dlsym's object pointer, copied into a function pointer: ISO C has no cast between the two */
static any_fn symbol(library l, const char* name) {
    void* p = dlsym(l, name);
    any_fn f;
    memcpy(&f, &p, sizeof f);
    return f;
}
#endif

typedef const char* (*version_fn)(void);
typedef void* (*load_fn)(const char*);
typedef int32_t (*size_fn)(const void*);
typedef void (*free_fn)(void*);

static int failures;
#define CHECK(cond) do { if (!(cond)) { printf("    %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: unload_test <library>\n"); return 2; }
    const char* json = "target/unload-tiny.json";
    write_tiny_json(json);
    /* opened and closed before any call: nothing started, the close succeeds (whether the image then
     * leaves is the platform's: macOS keeps one that links the Objective-C runtime, as Foundation does) */
    library l = open_library(argv[1]);
    CHECK(l != NULL);
    if (!l) return 1;
    CHECK(close_library(l) == 0);
    /* opened, started, closed: the close succeeds, and the library stays in place */
    l = open_library(argv[1]);
    CHECK(l != NULL);
    if (!l) return 1;
    load_fn load = (load_fn) symbol(l, "toknroll_tokenizer_load");
    size_fn size = (size_fn) symbol(l, "toknroll_vocabulary_size");
    free_fn release = (free_fn) symbol(l, "toknroll_tokenizer_free");
    version_fn version = (version_fn) symbol(l, "toknroll_version");
    CHECK(load && size && release && version);
    if (!load || !size || !release || !version) return 1;
    void* t = load(json);
    CHECK(t != NULL);
    int32_t vocabulary = size(t);
    CHECK(vocabulary == VOCAB_JSON);
    CHECK(close_library(l) == 0);
    CHECK(mapped(argv[1]));                    /* still loaded: the code, the runtime and the tokenizer */
    CHECK(size(t) == vocabulary);
    CHECK(*version());
    /* opened again: the same library, the same runtime, the old tokenizer valid */
    library again = open_library(argv[1]);
    CHECK(again != NULL);
    size_fn size_again = again ? (size_fn) symbol(again, "toknroll_vocabulary_size") : NULL;
    CHECK(size_again == size);
    if (size_again) CHECK(size_again(t) == vocabulary);
    release(t);
    if (again) CHECK(close_library(again) == 0);
    printf("%s unload\n", failures ? "FAIL" : "ok  ");
    return failures ? 1 : 0;
}
