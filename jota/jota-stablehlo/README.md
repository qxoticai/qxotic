# jota-stablehlo

A Java model of [StableHLO](https://github.com/openxla/stablehlo) programs, with readers and writers for every format StableHLO uses, a verifier, and conversions to and from jota's TIR.
No native code: MLIR and StableHLO are not needed at runtime.

| format | read | write |
|---|---|---|
| MLIR text, pretty (custom assembly) | `MlirParser.parse` | `MlirPrinter.print` |
| MLIR text, generic form | `MlirParser.parse` | `MlirPrinter.printGeneric` |
| StableHLO portable artifact (VHLO bytecode, StableHLO 0.9.0 → 1.27.x) | `PortableArtifact.deserialize` | `PortableArtifact.serialize(program, target)` |
| plain MLIR bytecode of the `stablehlo` dialect | `MlirBytecodeReader.read` | `MlirBytecodeWriter.writePlain` |

## Using it

```java
Program p = MlirParser.parse(Path.of("model.mlir"));          // or PortableArtifact.deserialize(...)
List<Diagnostic> errors = Verifier.verify(p);                  // spec constraints (Cn) + type inference
byte[] artifact = PortableArtifact.serialize(p, StableHloVersion.parse("1.0.0"));
String text = MlirPrinter.print(p);                            // as stablehlo-opt prints it
TirConversion tir = StableHloToTirConverter.convert(p, "main"); // run it on jota
List<Object> results = Interpreter.run(p, "main", args);         // or interpret it directly
Program exported = TirToStableHlo.export(tracedGraph);          // and back
```

`Interpreter` follows the spec on statically shaped tensors and runs the `check.*` assertions of StableHLO's tests: it reproduces XLA's expected values for 2270 of the 2325 `stablehlo/testdata` programs; implementation-defined integer results follow XLA by default (`Interpreter.Semantics`).

The model is a generic IR (`Operation`, `Value`, `Block`, `Region`, `Program`) with typed views of every StableHLO op (`op.asOp()`, `OpVisitor`).
`StructuralEquality.diff` compares two programs modulo value names.

## Command line

`StableHloTool` detects the input format from the file content:

```
stablehlo-tool info    FILE                  format, versions, functions, op counts
stablehlo-tool print   FILE [--generic] [-o OUT]
stablehlo-tool convert FILE --to text|generic|portable|bytecode -o OUT [--target X.Y.Z] [--bytecode-version N]
stablehlo-tool verify  FILE                  exit status 1 on a diagnostic, with source lines for text
stablehlo-tool to-tir  FILE [--function NAME]
stablehlo-tool run     FILE [--function NAME]  interpret a function without arguments, print its results
```

From `jota/` (relative paths are resolved from there):

```bash
mvn -q -pl jota-stablehlo compile org.codehaus.mojo:exec-maven-plugin:3.6.1:java \
  -Dexec.mainClass=com.qxotic.jota.stablehlo.StableHloTool \
  -Dexec.args="info jota-stablehlo/src/test/resources/stablehlo-samples/portable/stablehlo_legalize_to_vhlo.1_0_0.mlir.bc"
```

Plain bytecode is not versioned the way portable artifacts are: an op's property layout changes when StableHLO adds an attribute.
`--bytecode-version 4` or older stores attributes in a dictionary, which any StableHLO reads; 5 and later use the layouts of `--target`.
Use a portable artifact to exchange programs between StableHLO versions.

## Tests

`mvn -pl jota-stablehlo test` runs the default suite on the bundled samples (`src/test/resources/stablehlo-samples`, Apache-2.0, see `NOTICE`).
Optional properties widen it, on the command line or in an active `~/.m2/settings.xml` profile:

- `-Dstablehlo.repo=/path/to/openxla/stablehlo`: the whole StableHLO corpus (2332 testdata programs, interpreter tests, every portable artifact version), the verifier corpus, and XLA's expected values on jota's TIR interpreter.
- `-Dstablehlo.jax.python=tools/jax-python.sh`: cross-checks against StableHLO itself through a pinned jaxlib (it reads our portable artifacts and plain bytecode, parses and verifies our pretty text) and runs exported programs on XLA. The script creates its venv with `uv` on first use.

## Tools

| script | what it does |
|---|---|
| `tools/refresh-samples.sh` | copies the curated text and portable samples from a StableHLO checkout |
| `tools/gen-bytecode-samples.py` | writes the plain bytecode samples with jaxlib (`tools/jax-python.sh -I tools/gen-bytecode-samples.py`) |
| `tools/genops.py` | generates the typed op records and `OpVisitor` |
| `tools/genvhlo.py` | generates the VHLO op schemas from `VhloOps.td` |
| `tools/genprops.py` | generates the per-version op property layouts from `StablehloOps.td` / `ChloOps.td` (needs a full-history clone) |
| `tools/jax-python.sh` | Python with the pinned jax / jaxlib for the opt-in tests |

Design notes, the TIR ⇄ StableHLO parity analysis and the remaining work are in `../TODO-parity-TIR-StableHLO.md`.
