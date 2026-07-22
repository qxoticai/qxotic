# jinfer build entry points - thin wrappers over the Maven build. The poms are the source of
# truth: jinfer-parent's pluginManagement owns the shared shade + native-image configuration,
# jinfer-server/-bench name their executable via the jinfer.image.* properties.

JAVA ?= java
JAR_FILE = jinfer.jar

ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif
NATIVE_FILE := jinfer$(EXE)

# Force-inline the hot Vector API helpers (see ./hotspot_compiler). Without the hint a weaker
# inliner (GraalVM CE / plain C2 under budget) leaves a FloatVector-returning helper out-of-line,
# so the vector escapes the call and is boxed (heap-allocated) instead of scalar-replaced into a
# SIMD register - measured ~45% slower prefill on GraalVM CE. A strong inliner (Oracle GraalVM)
# does it anyway. The file form keeps the (growing) list in one place.
INLINE_HINTS = -XX:CompileCommandFile=$(dir $(lastword $(MAKEFILE_LIST)))hotspot_compiler
JAVA_RUNTIME_OPTIONS = --enable-preview --add-modules jdk.incubator.vector,jdk.httpserver \
	--enable-native-access=ALL-UNNAMED $(INLINE_HINTS)

default: jar

compile:
	mvn -o compile

test-compile:
	mvn -o test-compile

# Unit tests + weights-free oracles; integration/bench/driver suites are tag-gated (see the
# surefire.excludedGroups comment in pom.xml for how to opt in).
test:
	mvn -o test

jar:
	mvn -o package -DskipTests
	cp jinfer-server/target/jinfer.jar $(JAR_FILE)

clean:
	mvn clean
	rm -f $(JAR_FILE) $(NATIVE_FILE)

# The full JVM flag set for running the jar by hand (Vector API, preview, the inline hints).
run-jar-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE)

# Two identical greedy runs off the jar must be byte-identical (set MODEL=path/to/model.gguf).
MODEL ?= ../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf

test-golden: jar
	@test -f $(MODEL) || { echo "test-golden: model not found ($(MODEL)), skipping"; exit 0; }
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --max-tokens 64 2>/dev/null > /tmp/jinfer-golden-1.txt
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --max-tokens 64 2>/dev/null > /tmp/jinfer-golden-2.txt
	@diff /tmp/jinfer-golden-1.txt /tmp/jinfer-golden-2.txt && echo "test-golden: deterministic"

# GraalVM Native Image, via the poms' -Pnative profile (single source of the image flags).
#   make native                            self-contained ./jinfer
#   PRELOAD_GGUF=model.gguf make native    embed the model, instant TTFT
NATIVE_IMAGE ?= $(if $(JAVA_HOME),$(JAVA_HOME)/bin/native-image,native-image)

check-native-image:
	@v=$$($(NATIVE_IMAGE) --version 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
	case "$$v" in \
	25.0.0|25.0.1|25.0.2) \
		echo "ERROR: native-image $$v is too old for the jinfer kernels (need >= 25.0.3)."; \
		echo "       e.g. JAVA_HOME=$$HOME/Downloads/graalvm-25.1.0-dev+9.1 make $(MAKECMDGOALS)"; \
		exit 1;; \
	esac

native: check-native-image
	mvn -o -Pnative -pl jinfer-server -am package -DskipTests -Djinfer.PreloadGGUF=$(PRELOAD_GGUF)
	cp jinfer-server/target/jinfer$(EXE) $(NATIVE_FILE)

.PHONY: default compile test-compile test jar clean run-jar-command test-golden check-native-image native
