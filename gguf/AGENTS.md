# GGUF Agent Guide

## Scope

`gguf` is a Java library for GGUF metadata and tensor descriptors.

- It reads/writes GGUF structure and metadata.
- It computes tensor offsets/sizes.
- It does **not** read/write tensor payload bytes for users.
- It does **not** do quantization or inference.

## Key Sources

- Public API: `src/main/java/com/qxotic/format/gguf/`
- Internal IO: `src/main/java/com/qxotic/format/gguf/impl/`
- Tests: `src/test/java/com/qxotic/format/gguf/`
- Docs: `README.md`, `docs/`

## Build & Test

Run from repository root.

```bash
mvnd -pl gguf -am compile
mvnd -pl gguf -am test
mvnd -pl gguf -am spotless:check
mvnd -pl gguf -am spotless:apply
```

Run one test class:

```bash
mvnd -pl gguf -am test -Dtest=TensorEntryTest
```

External corpus tests (network + cache):

```bash
GGUF_EXTERNAL_TESTS=true mvnd -pl gguf -am test -Dtest=ExternalCorpusTest
```

- Default cache: `~/.cache/qxotic/gguf-metadata`
- Override cache: `GGUF_FIXTURES_DIR=/path/to/cache`

## Release Preflight

```bash
mvnd -pl gguf -am spotless:check
mvnd -pl gguf -am test
mvnd -pl gguf -am -Prelease -Dgpg.skip=true -DskipTests verify
```

Real deploy uses `-Prelease` **without** `-Dgpg.skip=true`.

## Code Conventions

- Java 11 compatible.
- Keep public API stable and explicit.
- Use `long` for offsets/sizes and element counts.
- Use `Math.multiplyExact` for overflow-safe size math.
- Throw:
  - `GGUFFormatException` for malformed GGUF content.
  - `IllegalArgumentException` for invalid API inputs.
  - `IOException` for IO failures.
- Prefer small, behavior-focused tests over implementation-coupled tests.

## Change Checklist

When changing GGUF format behavior:

1. Update reader/writer and any impacted API methods.
2. Add/adjust tests (positive + negative paths).
3. Run `spotless:apply`, then full `test`.
4. If applicable, run `ExternalCorpusTest` and update docs.
