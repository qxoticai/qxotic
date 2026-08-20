# Single root-reactor entry points. Always build from here: the subtrees are not
# dependency-closed, so `mvn -f jinfer` only works after the outer modules are installed.
# These targets run the full reactor from the root, which is the only build that works cold.

MAVEN ?= $(shell command -v mvnd >/dev/null 2>&1 && echo mvnd || echo mvn)

# Extra flags for every Maven invocation, e.g. `make test MAVEN_FLAGS=-o` for offline builds.
MAVEN_FLAGS ?=

default: test

# Build and test everything in the reactor (weights-free oracles; model-backed suites are
# tag-gated, see jinfer/pom.xml's surefire.excludedGroups).
test:
	$(MAVEN) $(MAVEN_FLAGS) test

# Compile everything without running tests.
compile:
	$(MAVEN) $(MAVEN_FLAGS) test-compile

# Build every artifact, skipping tests.
package:
	$(MAVEN) $(MAVEN_FLAGS) -DskipTests package

# Install every artifact into the local repository (the prerequisite for `mvn -f <subtree>`).
install:
	$(MAVEN) $(MAVEN_FLAGS) -DskipTests install

# GraalVM Native Image for the CLI. PRELOAD_GGUF is forwarded unchanged.
native:
	$(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/jinfer-cli -am clean package -DskipTests -Djinfer.preload=$(PRELOAD_GGUF)

# Demo apps are not in the default reactor; build them explicitly.
examples:
	$(MAVEN) $(MAVEN_FLAGS) -Pexamples package

# Apply Spotless across the reactor.
format:
	$(MAVEN) $(MAVEN_FLAGS) spotless:apply

.PHONY: default test compile package install native examples format
