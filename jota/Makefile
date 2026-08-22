# jota developer conveniences - `make` (or `make help`) lists them. jota IS dependency-closed
# (nothing outside this tree but JUnit), so these targets drive this subtree's own reactor
# directly: cold-safe on a clean clone, no root build or `mvn install` needed. The root
# Makefile covers the whole reactor (`make -C .. test`, `make -C .. jota-test`, ...).

# Plain mvn by default: a long-lived mvnd daemon can serve a stale effective pom (observed:
# tests silently skipped after a pom edit until `mvnd --stop`). Opt in with MAVEN=mvnd.
MAVEN ?= mvn

# Extra flags for every Maven invocation, e.g. `make test MAVEN_FLAGS=-o` for offline builds.
MAVEN_FLAGS ?=

MVN = $(MAVEN) $(MAVEN_FLAGS) -f $(abspath $(dir $(lastword $(MAKEFILE_LIST))))/pom.xml

default: help

##@ Build

compile: ## Compile the subtree without running tests
	$(MVN) test-compile

package: ## Build every jota artifact, skipping tests
	$(MVN) -DskipTests package

##@ Test

test: ## Full suite: core, memory + tensor suite on the always-on Java backend (GPU: MAVEN_FLAGS=-Pall)
	$(MVN) test

##@ Tidy

clean: ## Wipe the subtree's build output
	$(MVN) clean

help: ## Show this help
	@awk 'BEGIN { FS = " *## *" } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		NF >= 2 && $$1 ~ /^[a-zA-Z_-]+:/ { t = $$1; sub(/:.*/, "", t); \
			printf "  \033[36m%-10s\033[0m %s\n", t, $$2 }' $(MAKEFILE_LIST)

.PHONY: default help compile package test clean
