#!/usr/bin/env bash
# Python with the pinned jax/jaxlib used by the opt-in jaxlib tests (JaxInteropTest, XlaExecutionTest):
#   mvn -pl jota-stablehlo test -DskipTests=false -Dstablehlo.jax.python=tools/jax-python.sh
# The tests compare against the StableHLO bundled with this jaxlib, so the version is pinned.
# First use creates the venv with uv (override its location with JOTA_JAX_ENV); later uses just run it.
set -euo pipefail
JAX_VERSION=0.12.1
env="${JOTA_JAX_ENV:-$HOME/.local/share/jota-stablehlo/jax-$JAX_VERSION}"
if [[ ! -x "$env/bin/python" ]]; then
  command -v uv >/dev/null || { echo "jax-python.sh: needs uv (https://docs.astral.sh/uv/)" >&2; exit 1; }
  uv venv -q "$env" >&2
  VIRTUAL_ENV="$env" uv pip install -q "jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" >&2
fi
exec "$env/bin/python" "$@"
