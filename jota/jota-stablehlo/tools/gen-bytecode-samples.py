"""Writes plain (non-portable) MLIR bytecode samples of the stablehlo dialect.

Usage (from jota/jota-stablehlo, with the pinned jaxlib):
  tools/jax-python.sh -I tools/gen-bytecode-samples.py

Every text/testdata/*.mlir sample is parsed by MLIR (jaxlib) and written as
bytecode/<name>.mlirbc at jaxlib's current bytecode version; the first few are also
written at older bytecode versions as bytecode/<name>.v<N>.mlirbc. The source text of
each sample stays in text/testdata, and the jaxlib version goes to bytecode/GENERATED_BY.
"""
import pathlib

import jaxlib
from jax._src.interpreters import mlir as jmlir
from jax._src.lib.mlir import ir

here = pathlib.Path(__file__).resolve().parent.parent / "src/test/resources/stablehlo-samples"
out = here / "bytecode"
out.mkdir(exist_ok=True)
for old in out.glob("*.mlirbc"):
    old.unlink()

OLD_VERSIONS = (1, 4)  # before properties (v5) and before the current layout
OLD_SAMPLES = 8

sources = sorted((here / "text/testdata").glob("*.mlir"))
for i, src in enumerate(sources):
    with jmlir.make_ir_context():
        module = ir.Module.parse(src.read_text())
        assert module.operation.verify()
        with open(out / (src.stem + ".mlirbc"), "wb") as f:
            module.operation.write_bytecode(f)
        if i < OLD_SAMPLES:
            for v in OLD_VERSIONS:
                with open(out / f"{src.stem}.v{v}.mlirbc", "wb") as f:
                    module.operation.write_bytecode(f, desired_version=v)

(out / "GENERATED_BY").write_text(
    f"tools/gen-bytecode-samples.py with jaxlib {jaxlib.__version__}, "
    f"from text/testdata ({len(sources)} files)\n"
)
print(f"wrote {len(list(out.glob('*.mlirbc')))} files to {out}")
