# All Maven entry points for the repository - `make` (or `make help`) lists them. Maven runs
# at this root reactor or at a dependency-closed subtree's own: jinfer is NOT closed (it pulls
# gguf, json, toknroll, jota and jam), so only this reactor resolves it from a clean clone;
# jota IS closed, so its targets drive jota/pom.xml directly. jinfer/Makefile and
# jota/Makefile hold thin aliases into here plus their local helpers.

# Plain mvn by default: a long-lived mvnd daemon can serve a stale effective pom (observed:
# tests silently skipped after a pom edit until `mvnd --stop`). Opt in with MAVEN=mvnd.
MAVEN ?= mvn

# Extra flags for every Maven invocation, e.g. `make test MAVEN_FLAGS=-o` for offline builds.
MAVEN_FLAGS ?=

ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif

# The jinfer subtree, cold-safe: `jinfer` selects the aggregator and -amd follows parentage to
# every module beneath it, so a NEW module joins automatically; `jinfer/jinfer-cli` anchors -am
# so the sibling trees build too - -am does not traverse projects pulled in by -amd. A
# dependency outside the CLI's closure fails the build loudly instead of skipping silently.
JINFER = -pl jinfer,jinfer/jinfer-cli -amd -am

default: help

##@ Build

package: ## Build every artifact, skipping tests
	$(MAVEN) $(MAVEN_FLAGS) -DskipTests package

compile: ## Compile everything without running tests
	$(MAVEN) $(MAVEN_FLAGS) test-compile

install: ## Install every artifact into ~/.m2 (the prerequisite for `mvn -f <subtree>`)
	$(MAVEN) $(MAVEN_FLAGS) -DskipTests install

jar: jinfer-jar ## Alias for jinfer-jar

jinfer-jar: ## The jinfer CLI jar, copied to jinfer/jinfer.jar (incremental)
	$(MAVEN) $(MAVEN_FLAGS) -pl jinfer/jinfer-cli -am package -DskipTests
	cp jinfer/jinfer-cli/target/jinfer.jar jinfer/jinfer.jar

##@ Test

test: ## Build and test the whole reactor (model-backed suites are tag-gated, see jinfer/pom.xml)
	$(MAVEN) $(MAVEN_FLAGS) test

jinfer-test: ## Just the jinfer subtree's tests
	$(MAVEN) $(MAVEN_FLAGS) $(JINFER) test

jota-test: ## jota full suite (closed subtree): core, memory + the tensor suite on the Java backend
	$(MAVEN) $(MAVEN_FLAGS) -f jota/pom.xml test

jam-test: ## jam and its cross-backend parity suite (NativeJAM included when libjam loads)
	$(MAVEN) $(MAVEN_FLAGS) -pl jam/jam-vector -am verify

toknroll-fixtures: ## Download the enwik benchmark corpora into the cache (FIXTURES="enwik8" to fetch just one; ~350MB for both)
	$(MAVEN) $(MAVEN_FLAGS) -q -pl toknroll/toknroll-benchmarks -am install -DskipTests -Dspotless.check.skip=true
	$(MAVEN) $(MAVEN_FLAGS) -q -pl toknroll/toknroll-benchmarks exec:java -Dexec.mainClass=com.qxotic.toknroll.benchmarks.FetchCorpus -Dexec.classpathScope=test -Dexec.args="$(FIXTURES)"

##@ Native image (GraalVM)

NATIVE_IMAGE ?= $(if $(JAVA_HOME),$(JAVA_HOME)/bin/native-image,native-image)

native: ## jinfer CLI native image -> jinfer/jinfer; PRELOAD_GGUF=model.gguf embeds metadata
	@v=$$($(NATIVE_IMAGE) --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
	if [ -z "$$v" ]; then \
		echo "ERROR: native-image not found or version unparseable ($(NATIVE_IMAGE))"; exit 1; \
	fi; \
	major=$${v%%.*}; rest=$${v#*.}; minor=$${rest%%.*}; patch=$${v##*.}; \
	if [ "$$major" -lt 25 ] || { [ "$$major" -eq 25 ] && [ "$$minor" -eq 0 ] && [ "$$patch" -lt 3 ]; }; then \
		echo "ERROR: native-image $$v is too old for the jinfer kernels (need >= 25.0.3)."; \
		exit 1; \
	fi
	$(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/jinfer-cli -am package -DskipTests -Djinfer.preload=$(PRELOAD_GGUF)
	cp jinfer/jinfer-cli/target/jinfer$(EXE) jinfer/jinfer$(EXE)

##@ Tidy

format: ## Apply Spotless across the reactor
	$(MAVEN) $(MAVEN_FLAGS) spotless:apply

clean: ## Wipe the whole reactor's output
	$(MAVEN) $(MAVEN_FLAGS) clean
	rm -f jinfer/jinfer.jar jinfer/jinfer$(EXE)

jinfer-clean: ## Wipe jinfer plus the sibling output its -am closure built (same incrementality state)
	$(MAVEN) $(MAVEN_FLAGS) $(JINFER) clean
	rm -f jinfer/jinfer.jar jinfer/jinfer$(EXE)

jota-clean: ## Wipe just the jota subtree's output
	$(MAVEN) $(MAVEN_FLAGS) -f jota/pom.xml clean

##@ Release

release-canary: ## Prove the published shape works: install the release build into a throwaway repo, compile a BOM consumer against ONLY it
	MAVEN="$(MAVEN)" MAVEN_FLAGS="$(MAVEN_FLAGS)" ./release-canary.sh

jam-natives: ## Build, stage and stamp every shipped libjam (linux/windows x86-64 here, darwin-aarch64 on JAM_MAC=user@mac over ssh)
	jam/jam-native/scripts/natives.sh build

##@ Miscellaneous

examples: ## Build the demo apps (already in the default reactor; this target just limits the build to them)
	$(MAVEN) $(MAVEN_FLAGS) -pl examples -am package

help: ## Show this help
	@awk 'BEGIN { FS = " *## *" } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		NF >= 2 && $$1 ~ /^[a-zA-Z_-]+:/ { t = $$1; sub(/:.*/, "", t); \
			printf "  \033[36m%-14s\033[0m %s\n", t, $$2 }' $(MAKEFILE_LIST)
	@echo
	@echo '  Subtrees: make -C jinfer help (run, test-golden, ...) | make -C jota help'

.PHONY: default help package compile install jar jinfer-jar test jinfer-test jota-test \
	jam-test native format clean jinfer-clean jota-clean examples release-canary jam-natives
