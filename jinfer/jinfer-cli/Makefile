# The jinfer command as a GraalVM native image. `make native` is this machine's build, the root Makefile's
# recipe (-march=native, into the repository's bin/). What ships is `make image`: the image at the
# architecture's floor, native-image's default target spelled out since the pom's is compatibility (x86-64-v3,
# armv8.1-a: the libraries' floor), Oracle GraalVM's PGO build (GraalVM CE: -O3); the kernels' vector width follows
# the target through the libraries' own features, as in every image. `make dist` wraps it as the release
# archive, a prefix tree with bin/jinfer, this README, the license and the notice, and check-dist runs that.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
MAVEN ?= mvn
OS    := $(shell uname -s | tr A-Z a-z)
ARCH  := $(shell uname -m | sed -e 's/x86_64/x86-64/' -e 's/arm64/aarch64/')
PLATFORM := $(OS)-$(ARCH)
MARCH := $(if $(filter x86-64,$(ARCH)),x86-64-v3,armv8.1-a)
BIN   := target/jinfer
NATIVE_IMAGE ?= $(if $(JAVA_HOME),$(JAVA_HOME)/bin/native-image,native-image)
# Oracle GraalVM builds what ships optimized for a profile its instrumented image records first (PGO), which
# libjinfer's build reuses; GraalVM CE builds at the pom's -O3
PGO = $(shell $(NATIVE_IMAGE) --version 2>/dev/null | grep -q 'Oracle GraalVM' && echo yes)
# The profile, a list of ../scripts/pgo.txt's: one per turn, comma-separated as --pgo takes them
PROFILES := target/pgo/profiles
PGO_PROMPT := In two sentences, explain why a compiler that sees how a program actually runs can make it faster than one that only reads its source code.
# the check's model, in every build machine's cache (the test fixture); make keeps the spaces before a trailing comment, so none here
MODEL ?= LiquidAI/LFM2.5-350M-GGUF:Q8_0
# as native.mk: the libraries' floor, GraalVM 25's own darwin objects are built for it
MACOSX_DEPLOYMENT_TARGET ?= $(if $(filter darwin,$(OS)),14.0)
export MACOSX_DEPLOYMENT_TARGET
# the version is the image's own word, once it exists
VERSION = $(shell $(BIN) --version | awk '{print $$2}')
DIST = jinfer-$(VERSION)-$(PLATFORM)
# root's entries, as native.mk's tarballs: GNU tar (Linux, the container) or bsdtar (macOS)
TAR_ROOT := $(if $(shell tar --version 2>/dev/null | grep 'GNU tar'),--owner=0 --group=0,--uid 0 --gid 0) --numeric-owner
.DEFAULT_GOAL := help

help:
	@echo 'make native       GraalVM native image of jinfer for this machine (-march=native) -> $(ROOT)/bin/jinfer'
	@echo 'make image        what ships: the image at the architecture floor (-march=$(MARCH)) -> $(BIN)'
	@echo '                  Oracle GraalVM: PGO, from a profile recorded first (pgo-profile) and kept; CE: -O3'
	@echo 'make pgo-profile  record the profile again: ../scripts/pgo.txt'"'"'s turns on an instrumented image -> $(PROFILES)'
	@echo 'make check        the image answers --version, --help and one instruct turn on $(MODEL)'
	@echo 'make dist         the release tarball, a prefix tree: bin/jinfer, README, LICENSE, NOTICE -> target/dist/jinfer-<version>-$(PLATFORM).tar.gz'
	@echo 'make check-dist   unpack the tarball and run its jinfer as check does'

native:
	$(MAKE) -C $(ROOT) native NATIVE_MODULE=$(notdir $(CURDIR))

image: $(if $(PGO),$(PROFILES))   # no clean: the image is linked from the modules' classes, not the shaded jar the root's native target cleans for
	@$(MAKE) -s -C $(ROOT) check-native-image
	cd $(ROOT) && $(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/jinfer-cli -am package -DskipTests -Djinfer.image.march=$(MARCH) \
	  $(if $(PGO),-Djinfer.image.opt=--pgo=$$(cat $(abspath $(PROFILES))))

# The profile: each turn of ../scripts/pgo.txt on the instrumented image, a process and a profile each, a prompt and
# 4 tokens, as instrumented code decodes a 1B model at seconds a token. Recorded once per tree, since that takes
# minutes: after the code changed, make pgo-profile records it again (a stale one costs speed, never correctness).
# The models download on the first run, or come from the cache (make cache-models).
$(PROFILES):
	@$(MAKE) -s -C $(ROOT) check-native-image
	cd $(ROOT) && $(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/jinfer-cli -am package -DskipTests -Djinfer.image.march=$(MARCH) \
	  -Djinfer.image.opt=--pgo-instrument
	rm -rf target/pgo && mkdir -p target/pgo && mv $(BIN) target/pgo/train
	i=0; grep -vE '^(#|$$)' ../scripts/pgo.txt | while read -r backend temp model; do i=$$((i + 1)); \
	  echo "pgo $$i: $$model on jam-$$backend"; \
	  target/pgo/train -XX:ProfilesDumpFile=$(abspath target/pgo)/$$i.iprof -Djam.$$([ $$backend = native ] && echo vector || echo native).disabled=true \
	    -m $$model $$([ $$temp = default ] || echo --temp $$temp) --think off -n 4 instruct "$(PGO_PROMPT)" > /dev/null || exit 1; done
	ls $(abspath target/pgo)/*.iprof | paste -sd, - > $@

pgo-profile:
	rm -f $(PROFILES) && $(MAKE) $(PROFILES)

check:
	$(BIN) --version && $(BIN) --help > /dev/null
	$(BIN) -m $(MODEL) --temp 0 instruct "Reply with the single word ready." > /dev/null

$(BIN):
	@echo "$(BIN) is not built: make image" >&2; exit 1

dist: $(BIN)
	rm -rf target/dist/$(DIST) && mkdir -p target/dist/$(DIST)/bin
	cp $(BIN) target/dist/$(DIST)/bin/ && cp README.md ../jinfer-native/NOTICE $(ROOT)/LICENSE target/dist/$(DIST)/   # the notice: the same image, the same terms
	cd target/dist && rm -f $(DIST).tar.gz && tar $(TAR_ROOT) -czf $(DIST).tar.gz $(DIST) && sha256sum $(DIST).tar.gz > $(DIST).tar.gz.sha256
	@ls -la target/dist/$(DIST).tar.gz && cat target/dist/$(DIST).tar.gz.sha256

check-dist:
	rm -rf target/distenv && mkdir -p target/distenv && tar -xzf target/dist/$(DIST).tar.gz -C target/distenv
	$(MAKE) check BIN=$(abspath target/distenv/$(DIST)/bin/jinfer)
	@echo "the unpacked $(DIST) answers --version, --help and an instruct turn"

.PHONY: help native image pgo-profile check dist check-dist
