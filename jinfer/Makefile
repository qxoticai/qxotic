# jinfer developer conveniences - `make` (or `make help`) lists them. Builds are aliases into
# the ROOT Makefile (../Makefile), the single Maven entry point, because the jinfer subtree is
# not dependency-closed (gguf, json, toknroll, jota and jam are sibling trees only the root
# reactor sees on a clean clone); plain `mvn` inside this directory works only after a root
# `mvn install`. The poms are the source of truth: jinfer-parent's pluginManagement owns the
# shared shade + native-image configuration, jinfer-cli and jinfer-bench name their executables
# via the jinfer.image.* properties.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

JAVA ?= java
JAR_FILE = jinfer.jar

ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif
NATIVE_FILE := jinfer$(EXE)

# Command-line variables (MAVEN_FLAGS=-o, ...) ride MAKEFLAGS into the sub-make automatically.

default: help

##@ Build (aliases into the root Makefile)

jar: ## The CLI jar, copied to ./jinfer.jar (incremental - `make clean jar` for from-scratch)
	$(MAKE) -C $(ROOT) jinfer-jar

test: ## The jinfer subtree's tests (model-backed suites tag-gated, see pom.xml to opt in)
	$(MAKE) -C $(ROOT) jinfer-test

native: ## GraalVM native image -> ./jinfer; PRELOAD_GGUF=model.gguf embeds tokenizer metadata
	$(MAKE) -C $(ROOT) native PRELOAD_GGUF=$(PRELOAD_GGUF)

clean: ## Wipe the subtree's build output (and the sibling output its closure built)
	$(MAKE) -C $(ROOT) jinfer-clean

##@ Run

# Force-inline the hot Vector API helpers (see ./hotspot_compile_commands). Without the hint a weaker
# inliner (GraalVM CE / plain C2 under budget) leaves a FloatVector-returning helper out-of-line,
# so the vector escapes the call and is boxed (heap-allocated) instead of scalar-replaced into a
# SIMD register - measured ~45% slower prefill on GraalVM CE. A strong inliner (Oracle GraalVM)
# does it anyway. The file form keeps the (growing) list in one place.
INLINE_HINTS = -XX:CompileCommandFile=$(DIR)/hotspot_compile_commands
JAVA_RUNTIME_OPTIONS = --add-modules jdk.incubator.vector,jdk.httpserver \
	--enable-native-access=ALL-UNNAMED $(INLINE_HINTS)

run: jar ## Build, then launch the CLI: make run ARGS="--model hf.co/... --chat"
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(DIR)/$(JAR_FILE) $(ARGS)

run-jar-command: ## Print the full java invocation for running the jar by hand
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(DIR)/$(JAR_FILE)

# Two identical greedy runs off the jar must be byte-identical (set MODEL=path/to/model.gguf).
MODEL ?= $(ROOT)/../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf

test-golden: jar ## Determinism check: two greedy runs, byte-identical (MODEL=...)
	@test -f $(MODEL) || { echo "test-golden: ERROR: model not found ($(MODEL)); set MODEL=..." >&2; exit 1; }
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(DIR)/$(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --temperature 0 --max-output-tokens 64 2>/dev/null > /tmp/jinfer-golden-1.txt
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(DIR)/$(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --temperature 0 --max-output-tokens 64 2>/dev/null > /tmp/jinfer-golden-2.txt
	@diff /tmp/jinfer-golden-1.txt /tmp/jinfer-golden-2.txt && echo "test-golden: deterministic"

help: ## Show this help
	@awk 'BEGIN { FS = " *## *" } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		NF >= 2 && $$1 ~ /^[a-zA-Z_-]+:/ { t = $$1; sub(/:.*/, "", t); \
			printf "  \033[36m%-16s\033[0m %s\n", t, $$2 }' $(MAKEFILE_LIST)

.PHONY: default help jar test native clean run run-jar-command test-golden
