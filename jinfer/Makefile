# jinfer developer conveniences - `make` (or `make help`) lists them. Builds are aliases into
# the ROOT Makefile (../Makefile), the single Maven entry point, because the jinfer subtree is
# not dependency-closed (gguf, json, toknroll, jota and jam are sibling trees only the root
# reactor sees on a clean clone); plain `mvn` inside this directory works only after a root
# `mvn install`. The poms are the source of truth: jinfer-parent's pluginManagement owns the
# shared shade + native-image configuration, jinfer-cli and jinfer-bench name their executables
# via the jinfer.image.* properties.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

JAVA ?= java
JAR_FILE = $(ROOT)/bin/jinfer.jar

# Command-line variables (MAVEN_FLAGS=-o, ...) ride MAKEFLAGS into the sub-make automatically.

default: help

##@ Build (aliases into the root Makefile)

jar: ## The CLI jar -> ../bin/jinfer.jar
	$(MAKE) -C $(ROOT) jinfer-jar

test: ## The jinfer subtree's tests (model-backed suites tag-gated, see pom.xml to opt in)
	$(MAKE) -C $(ROOT) jinfer-test

native: ## GraalVM native image of the CLI -> ../bin/jinfer; PRELOAD_GGUF=model.gguf embeds tokenizer metadata (the others: make -C jinfer-tts native, make -C jinfer-bench native)
	$(MAKE) -C $(ROOT) native PRELOAD_GGUF=$(PRELOAD_GGUF)

clean: ## Wipe the subtree's build output (and the sibling output its closure built), bin/ included
	$(MAKE) -C $(ROOT) jinfer-clean

##@ Run

# ./hotspot_compile_commands is a measurement instrument for C2's inliner, never a requirement
# (its header says how to measure with it); opt in with INLINE_HINTS=-XX:CompileCommandFile=...
INLINE_HINTS ?=
JAVA_RUNTIME_OPTIONS = --add-modules jdk.incubator.vector,jdk.httpserver \
	--enable-native-access=ALL-UNNAMED $(INLINE_HINTS)

run: jar ## Build, then launch the CLI: make run ARGS="--model hf.co/... --chat"
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE) $(ARGS)

run-jar-command: ## Print the full java invocation for running the jar by hand
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE)

# Two identical greedy runs off the jar must be byte-identical. MODEL is a reference, resolved as
# `jinfer pull` resolves it (the local store or the HuggingFace hub cache serve it offline, else it
# is downloaded), or a path to a GGUF.
MODEL ?= hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0
GOLDEN_RUN = $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE) --model $(MODEL) --prompt "List the planets." --temp 0 --max-output-tokens 64

test-golden: jar ## Determinism check: two greedy runs, byte-identical (MODEL=ref or path)
	@$(GOLDEN_RUN) > /tmp/jinfer-golden-1.txt 2> /tmp/jinfer-golden-1.err || { cat /tmp/jinfer-golden-1.err >&2; exit 1; }
	@$(GOLDEN_RUN) > /tmp/jinfer-golden-2.txt 2> /tmp/jinfer-golden-2.err || { cat /tmp/jinfer-golden-2.err >&2; exit 1; }
	@diff /tmp/jinfer-golden-1.txt /tmp/jinfer-golden-2.txt && echo "test-golden: deterministic"

help: ## Show this help
	@awk 'BEGIN { FS = " *## *" } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		NF >= 2 && $$1 ~ /^[a-zA-Z_-]+:/ { t = $$1; sub(/:.*/, "", t); \
			printf "  \033[36m%-16s\033[0m %s\n", t, $$2 }' $(MAKEFILE_LIST)

.PHONY: default help jar test native clean run run-jar-command test-golden
