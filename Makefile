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

jinfer-jar: ## The jinfer CLI jar -> bin/jinfer.jar (a clean build: shading has reused a stale module jar)
	$(MAVEN) $(MAVEN_FLAGS) -pl jinfer/jinfer-cli -am clean package -DskipTests
	mkdir -p bin && cp jinfer/jinfer-cli/target/jinfer.jar bin/jinfer.jar

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

# Which shipped binary: the module under jinfer/, and its executable - the module name with a
# -cli suffix dropped (jinfer-cli -> jinfer; jinfer-tts; jinfer-bench), as each pom's
# jinfer.image.name already says. A module's own Makefile passes NATIVE_MODULE, so
# `make -C jinfer/jinfer-tts native` builds that one; the default is the CLI.
NATIVE_MODULE ?= jinfer-cli
NATIVE_EXECUTABLE = $(patsubst %-cli,%,$(NATIVE_MODULE))

check-native-image: ## the GraalVM the images build with: native-image >= 25.0.4, GraalVM CE recommended (JAVA_HOME picks it)
	@v=$$($(NATIVE_IMAGE) --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
	if [ -z "$$v" ]; then \
		echo "ERROR: native-image not found or version unparseable ($(NATIVE_IMAGE))"; exit 1; \
	fi; \
	major=$${v%%.*}; rest=$${v#*.}; minor=$${rest%%.*}; patch=$${v##*.}; \
	if [ "$$major" -lt 25 ] || { [ "$$major" -eq 25 ] && [ "$$minor" -eq 0 ] && [ "$$patch" -lt 4 ]; }; then \
		echo "ERROR: native-image $$v is too old (need >= 25.0.4: the isolate parameters the C shims use)."; \
		exit 1; \
	fi; \
	vendor=$$($(NATIVE_IMAGE) --version 2>/dev/null | sed -n 2p | sed 's/^ *//'); \
	case "$$vendor" in *"Oracle GraalVM"*) \
		echo "note: $$vendor builds, but the binaries we ship are built with GraalVM CE (JAVA_HOME=/path/to/graalvm-ce)";; esac; \
	echo "native-image $$v ($$vendor)"

native: check-native-image ## GraalVM native image of the CLI for THIS machine (-march=native) -> bin/jinfer (a clean build, as jinfer-jar); NATIVE_MODULE=jinfer-tts|jinfer-bench for the others, or make -C jinfer/<module> native; PRELOAD_GGUF=model.gguf embeds metadata (CLI), MODELS=dir finds the Inflect lexicon (tts)
	$(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/$(NATIVE_MODULE) -am clean package -DskipTests -Djinfer.image.march=native -Djinfer.preload=$(PRELOAD_GGUF) $(if $(MODELS),-Djinfer.models=$(MODELS))
	mkdir -p bin && cp jinfer/$(NATIVE_MODULE)/target/$(NATIVE_EXECUTABLE)$(EXE) bin/$(NATIVE_EXECUTABLE)$(EXE)

libjinfer: ## libjinfer.so, jinfer's shared library with a C API (jinfer/jinfer-native); QUICK=1 for the dev loop
	$(MAKE) -C jinfer/jinfer-native libjinfer $(if $(QUICK),QUICK=1)

##@ Tidy

format: ## Apply Spotless across the reactor
	$(MAVEN) $(MAVEN_FLAGS) spotless:apply

clean: ## Wipe the whole reactor's output, bin/ included
	$(MAVEN) $(MAVEN_FLAGS) clean
	rm -rf bin

jinfer-clean: ## Wipe jinfer plus the sibling output its -am closure built (same incrementality state), bin/ included
	$(MAVEN) $(MAVEN_FLAGS) $(JINFER) clean
	rm -rf bin

jota-clean: ## Wipe just the jota subtree's output
	$(MAVEN) $(MAVEN_FLAGS) -f jota/pom.xml clean

##@ CI

test-fixtures: ## Fetch the tiktoken vocabularies, their golden truth, the enwik8 corpus and the Parakeet test speech the suites read (idempotent, ~109MB once)
	python3 toknroll/scripts/download_tiktoken_fixtures.py
	python3 toknroll/scripts/download_enwik8.py
	python3 jinfer/scripts/download_parakeet_fixtures.py

# Empty model caches: a test that reaches for a model fails instead of downloading one.
NO_MODELS := HF_HOME=$(CURDIR)/.ci-empty-hf-home JINFER_MODELS=$(CURDIR)/.ci-no-models

ci: test-fixtures ci-format ci-test ci-corpus ci-release ## What the pull-request CI runs, as one local sequence

ci-format: ## CI gate 1: formatting, including opt-in Maple and jinfer examples
	$(MAVEN) $(MAVEN_FLAGS) -B -Pmaple,examples spotless:check

ci-test: ## CI gate 2: model-free suite, including opt-in Maple and jinfer examples
	$(NO_MODELS) $(MAVEN) $(MAVEN_FLAGS) -B -Pmaple,examples test -Djinfer.test.noModels=true

ci-corpus: ## CI gate 3: the tests that read the enwik8 corpus (toknroll-core and what it builds on)
	$(NO_MODELS) $(MAVEN) $(MAVEN_FLAGS) -B -pl toknroll/toknroll-core -am test -Dgroups=corpus -Dsurefire.excludedGroups=

ci-release: ## CI gate 4: the release shape, unsigned, no natives
	$(MAVEN) $(MAVEN_FLAGS) -B clean -Prelease verify -DskipTests -Dgpg.skip=true -Djam.natives.check.skip=true

##@ Release

release-canary: ## Prove the published shape works: install the release build into a throwaway repo, compile a BOM consumer against ONLY it
	MAVEN="$(MAVEN)" MAVEN_FLAGS="$(MAVEN_FLAGS)" ./release-canary.sh

release-plan: ## What a release would publish, and whether the versions agree
	./release-plan.sh

release-deploy: ## Stage for Central what it does not hold yet: PROJECT=. for everything in one bundle, PROJECT=jinfer for one project (CHECK=--check to only look)
	MAVEN="$(MAVEN)" MAVEN_FLAGS="$(MAVEN_FLAGS)" ./release-deploy.sh $(PROJECT) $(CHECK)

jam-natives: ## Build, stage and stamp every shipped libjam (linux/windows x86-64 here, darwin-aarch64 on JAM_MAC=user@mac over ssh)
	jam/jam-native/scripts/natives.sh build

##@ Wheels

include hosts.mk
LEGS ?= linux-x86_64 linux-aarch64 macos-arm64 windows-x86_64
REF ?= HEAD

wheels: $(addprefix wheel-,$(LEGS)) ## Both wheels on every leg hosts.mk names, over ssh (REF=<tag> picks the commit) -> target/wheels/
	@ls -la target/wheels/*.whl | awk '{printf "%.1f MB %s\n", $$5/1e6, $$9}'

wheel-%: ## One leg: wheel-linux-x86_64, wheel-linux-aarch64, wheel-macos-arm64 or wheel-windows-x86_64
	HOST='$(HOST.$*)' SETUP='$(SETUP.$*)' ./wheels.sh $* $(REF)

wheels-native: ## Both wheels built and checked on THIS machine (what a leg runs; on Linux, inside the container)
	$(MAKE) -C toknroll/toknroll-native libtoknroll check-wheel
	$(MAKE) -C jinfer/jinfer-native libjinfer check-wheel

wheels-linux: ## wheels-native inside the manylinux container (docker): the glibc floor the wheels promise, whatever this host runs
	docker build -q -f manylinux.Dockerfile --build-arg ARCH=$(shell uname -m) -t qxotic-manylinux . >/dev/null
	docker run --rm -u "$$(id -u):$$(id -g)" -v /etc/passwd:/etc/passwd:ro -e HOME -v "$$HOME/.m2:$$HOME/.m2" -v "$$HOME/.cache:$$HOME/.cache" -v "$$PWD:/src" -w /src qxotic-manylinux make wheels-native

# The native Makefiles' release-tools venv (build, wheel, twine), shared by path.
TOOLS := target/pytools/$(shell python3 -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep="-")')
$(TOOLS)/bin/twine:
	rm -rf $(TOOLS) && python3 -m venv $(TOOLS) && $(TOOLS)/bin/pip install -q build wheel twine auditwheel

publish-wheels: $(TOOLS)/bin/twine ## twine check, then upload target/wheels/*.whl (TWINE_REPOSITORY=testpypi rehearses; credentials from ~/.pypirc)
	$(TOOLS)/bin/twine check --strict target/wheels/*.whl
	$(TOOLS)/bin/twine upload target/wheels/*.whl

##@ Miscellaneous

examples: ## Build the demo apps (already in the default reactor; this target just limits the build to them)
	$(MAVEN) $(MAVEN_FLAGS) -pl examples -am package

help: ## Show this help
	@awk 'BEGIN { FS = " *## *" } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		NF >= 2 && $$1 ~ /^[a-zA-Z_%-]+:/ { t = $$1; sub(/:.*/, "", t); \
			printf "  \033[36m%-18s\033[0m %s\n", t, $$2 }' $(MAKEFILE_LIST)
	@echo
	@echo '  Subtrees: make -C jinfer help (run, test-golden, ...) | make -C jota help | make -C jinfer/jinfer-tts native'

.PHONY: default help package compile install jar jinfer-jar test jinfer-test jota-test \
	jam-test check-native-image native libjinfer format clean jinfer-clean jota-clean examples release-canary jam-natives toknroll-fixtures test-fixtures ci ci-format ci-test ci-corpus ci-release release-plan release-deploy wheels wheels-native wheels-linux publish-wheels
