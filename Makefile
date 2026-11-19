# All Maven entry points for the repository - `make` (or `make help`) lists them. Maven runs
# at this root reactor or at a dependency-closed subtree's own: jinfer is NOT closed (it pulls
# gguf, json, toknroll, jota and jam), so only this reactor resolves it from a clean clone;
# jota IS closed, so its targets drive jota/pom.xml directly. jinfer/Makefile holds thin
# aliases into here plus its local helpers; jota/Makefile drives jota's own reactor, and the
# jota targets here delegate to it.

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
	$(MAKE) -C jota test

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

check-native-image: ## The GraalVM the images build with: native-image >= 25.0.4 (JAVA_HOME picks it); Oracle GraalVM adds PGO to libtoknroll
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
	echo "native-image $$v ($$vendor)"

native: check-native-image ## GraalVM native image of the CLI for THIS machine (-march=native) -> bin/jinfer (a clean build, as jinfer-jar); NATIVE_MODULE=jinfer-tts|jinfer-bench for the others, or make -C jinfer/<module> native; PRELOAD_GGUF=model.gguf embeds metadata (CLI), MODELS=dir finds the Inflect lexicon (tts)
	$(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/$(NATIVE_MODULE) -am clean package -DskipTests -Djinfer.image.march=native -Djinfer.preload=$(PRELOAD_GGUF) $(if $(MODELS),-Djinfer.models=$(MODELS))
	mkdir -p bin && cp jinfer/$(NATIVE_MODULE)/target/$(NATIVE_EXECUTABLE)$(EXE) bin/$(NATIVE_EXECUTABLE)$(EXE)

libjinfer: ## libjinfer.so, jinfer's shared library with a C API (jinfer/jinfer-native); QUICK=1 for the dev loop
	$(MAKE) -C jinfer/jinfer-native libjinfer

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
	$(MAKE) -C jota clean

##@ CI

test-fixtures: ## Fetch the tiktoken vocabularies, their golden truth, the enwik8 corpus, the Parakeet test speech and the jinfer test media the suites read (idempotent, ~112MB once)
	python3 toknroll/scripts/download_tiktoken_fixtures.py
	python3 toknroll/scripts/download_enwik8.py
	python3 jinfer/scripts/download_parakeet_fixtures.py
	python3 jinfer/scripts/download_media_fixtures.py

cache-models: ## Cache every model the native suites load, through the libraries built here: online, once per machine, the container too (cache-models-in-container)
	python3 jinfer/scripts/cache_native_models.py

cache-models-in-container:
	$(call in-container,cache-models)

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

release-canary: ## Prove the published shape works: install the release build into a throwaway repo, compile consumers with and without BOMs against ONLY it
	MAVEN="$(MAVEN)" MAVEN_FLAGS="$(MAVEN_FLAGS)" scripts/release-canary.sh

native-canary: ## Prove apps' native images need no configuration: a LangChain4j and a Spring Boot app, built with GraalVM against the installed jars, must answer
	MAVEN="$(MAVEN)" MAVEN_FLAGS="$(MAVEN_FLAGS)" scripts/native-canary.sh

release-plan: ## What a release would publish, and whether the versions agree
	scripts/release-plan.sh

release-deploy: ## Stage for Central what it does not hold yet: PROJECT=. for everything in one bundle, PROJECT=jinfer for one project (CHECK=--check to only look)
	MAVEN="$(MAVEN)" MAVEN_FLAGS="$(MAVEN_FLAGS)" scripts/release-deploy.sh $(PROJECT) $(CHECK)

jam-natives: ## Build, stage and stamp this host's shipped libjam (Linux x86-64: linux and windows x86-64; a Mac: darwin-aarch64, and linux-aarch64 in the container); natives.sh import adds the other host's
	jam/jam-native/scripts/natives.sh build
	$(if $(filter Darwin,$(shell uname -s)),$(call in-container,jam-natives))

##@ Wheels and archives

# Each machine builds its own wheels and release archives, from a clean export of REF (a tag, a commit, any
# tree-ish): nothing of the working tree reaches them, and the Linux container never meets the host's build
# directories. Windows, where make is not native, has scripts\wheels.ps1 and scripts\dists.ps1.
REF ?= HEAD
WHEELS_SRC := target/wheels-src
DISTS_SRC := target/dists-src

# The export of REF in $(1), where this Makefile's target $(2) runs.
define export-ref
rm -rf $(1) $(1).tar && mkdir -p $(1)
git archive -o $(1).tar $(REF) && tar -xf $(1).tar -C $(1)
$(MAKE) -C $(1) $(2)
endef

# The export builds the wheels with target $(1); each new wheel replaces the package's wheel for its platform
# in target/wheels/.
define wheels-from-export
$(call export-ref,$(WHEELS_SRC),$(1))
mkdir -p target/wheels
for w in $(WHEELS_SRC)/toknroll/toknroll-native/target/wheel/*.whl $(WHEELS_SRC)/jinfer/jinfer-native/target/wheel/*.whl; do \
  f=$${w##*/}; rm -f target/wheels/$${f%%-*}-*-$${f##*-} && cp $$w target/wheels/; done
@ls -l target/wheels/*.whl | awk '{printf "%.1f MB %s\n", $$5/1e6, $$9}'
endef

wheels: ## This machine's two wheels, built and checked from a clean export of REF (HEAD): Linux in the manylinux container, macOS natively -> target/wheels/
	$(call wheels-from-export,$(if $(filter Linux,$(shell uname -s)),wheels-in-container,wheels-here))

wheels-linux: ## The Linux wheels for this machine's architecture, in the manylinux container: on a Mac, Linux aarch64 (docker through Colima)
	$(call wheels-from-export,wheels-in-container)

wheels-here: # the two libraries, their checks (C, Python) and their checked wheels, in this tree, on this machine
	$(MAKE) -C toknroll/toknroll-native libtoknroll check check-wheel
	$(MAKE) -C jinfer/jinfer-native libjinfer check check-wheel

# A target of this Makefile in the manylinux container of this machine's architecture, which pins the glibc
# floor the wheels promise. It runs as the caller, with the caller's own passwd line (the JVM takes the home
# directory from it; a Mac's VM has another), and the Maven and model caches at their real paths. The image is
# built from the Dockerfile alone, with no build context: it copies nothing in.
define in-container
docker build -q --build-arg ARCH=$(shell uname -m | sed s/arm64/aarch64/) -t qxotic-manylinux - < scripts/manylinux.Dockerfile >/dev/null
mkdir -p target && printf '%s:x:%s:%s::%s:/bin/sh\n' "$$(id -un)" "$$(id -u)" "$$(id -g)" "$$HOME" > target/passwd
docker run --rm -u "$$(id -u):$$(id -g)" -v "$(CURDIR)/target/passwd:/etc/passwd:ro" -e HOME -e CTEST_ALLOW_SKIP \
  -v "$$HOME/.m2:$$HOME/.m2" -v "$$HOME/.cache:$$HOME/.cache" -v "$(CURDIR):/src" -w /src qxotic-manylinux make $(1)
endef

wheels-in-container:
	$(call in-container,wheels-here)

# The release archives, as the wheels: the two libraries for C and the jinfer command, built and checked by target
# $(1) in the export, Linux in the manylinux container for the glibc floor, then collected in target/dists/, one per
# library or command and platform: a new archive replaces the one of any version, a candidate's too, for its platform.
define dists-from-export
$(call export-ref,$(DISTS_SRC),$(1))
mkdir -p target/dists
for a in $(DISTS_SRC)/toknroll/toknroll-native/target/dist/*.tar.gz $(DISTS_SRC)/jinfer/jinfer-native/target/dist/*.tar.gz $(DISTS_SRC)/jinfer/jinfer-cli/target/dist/*.tar.gz; do \
  f=$${a##*/}; p=$$(printf %s "$$f" | sed -E 's/.*-((darwin|linux|windows)-)/\1/'); \
  rm -f target/dists/$${f%%-*}-*-$$p target/dists/$${f%%-*}-*-$$p.sha256 && cp $$a $$a.sha256 target/dists/; done
@ls -l target/dists/*.tar.gz target/dists/*.zip 2>/dev/null | awk '{printf "%.1f MB %s\n", $$5/1e6, $$9}'
endef

dists: ## This machine's three release archives, libtoknroll, libjinfer and the jinfer command, built and checked from a clean export of REF (HEAD): Linux in the manylinux container, macOS natively -> target/dists/
	$(call dists-from-export,$(if $(filter Linux,$(shell uname -s)),dists-in-container,dists-here))

dists-linux: ## The Linux archives for this machine's architecture, in the manylinux container: on a Mac, Linux aarch64
	$(call dists-from-export,dists-in-container)

dists-here: # the two libraries and the command, their checks, their archives and the archives' checks, in this tree, on this machine
	$(MAKE) -C toknroll/toknroll-native libtoknroll check dist check-dist
	$(MAKE) -C jinfer/jinfer-native libjinfer check dist check-dist
	$(MAKE) -C jinfer/jinfer-cli image check dist check-dist

dists-in-container:
	$(call in-container,dists-here)

# The release tools' venv: native.mk's, by the same path, so an in-tree build and publishing share it.
TOOLS := target/pytools/$(shell python3 -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep="-")')
$(TOOLS)/bin/twine:
	rm -rf $(TOOLS) && python3 -m venv $(TOOLS) && $(TOOLS)/bin/pip install -q twine auditwheel

publish-wheels: $(TOOLS)/bin/twine ## Upload target/wheels/ to PyPI after twine check, skipping wheels it holds, each built from REF (HEAD): each machine uploads its own (TWINE_REPOSITORY=testpypi rehearses; credentials from ~/.pypirc)
	@c=$$(git rev-parse '$(REF)^{commit}') && for w in target/wheels/*.whl; do n=$${w##*/}; \
	  unzip -p $$w $${n%%-*}/__init__.py | grep -q "built from $$c" || { echo "$$w is not built from $(REF), $$c"; exit 1; }; done
	$(TOOLS)/bin/twine check --strict target/wheels/*.whl
	$(TOOLS)/bin/twine upload --skip-existing target/wheels/*.whl

##@ Miscellaneous

examples: ## Build the demo apps (already in the default reactor; this target just limits the build to them)
	$(MAVEN) $(MAVEN_FLAGS) -pl examples -am package

help: ## Show this help
	@awk 'BEGIN { FS = " *## *" } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
		NF >= 2 && $$1 ~ /^[a-zA-Z_%-]+:/ { t = $$1; sub(/:.*/, "", t); \
			printf "  \033[36m%-18s\033[0m %s\n", t, $$2 }' $(MAKEFILE_LIST)
	@echo
	@echo '  Subtrees: make -C jinfer help (run, native, ...) | make -C jota help | make -C jinfer/jinfer-tts native'

.PHONY: dists dists-linux dists-here dists-in-container default help package compile install jar jinfer-jar test jinfer-test jota-test \
	jam-test check-native-image native libjinfer format clean jinfer-clean jota-clean examples release-canary jam-natives toknroll-fixtures test-fixtures cache-models cache-models-in-container ci ci-format ci-test ci-corpus ci-release release-plan release-deploy wheels wheels-linux wheels-here wheels-in-container publish-wheels
