# Single root-reactor entry points. Always build from here: the subtrees are not
# dependency-closed, so `mvn -f jinfer` only works after the outer modules are installed.
# These targets run the full reactor from the root, which is the only build that works cold.

MAVEN ?= mvn

default: test

# Build and test everything in the reactor (weights-free oracles; model-backed suites are
# tag-gated, see jinfer/pom.xml's surefire.excludedGroups).
test:
	$(MAVEN) -o test

# Compile everything without running tests.
compile:
	$(MAVEN) -o test-compile

# Build every artifact, skipping tests.
package:
	$(MAVEN) -o -DskipTests package

# Install every artifact into the local repository (the prerequisite for `mvn -f <subtree>`).
install:
	$(MAVEN) -o -DskipTests install

# GraalVM Native Image for the CLI. PRELOAD_GGUF is forwarded unchanged.
native:
	$(MAVEN) -o -Pnative -pl jinfer/jinfer-cli -am clean package -DskipTests

# Demo apps are not in the default reactor; build them explicitly.
examples:
	$(MAVEN) -o -Pexamples package

# Apply Spotless across the reactor.
format:
	$(MAVEN) -o spotless:apply

.PHONY: default test compile package install native examples format
