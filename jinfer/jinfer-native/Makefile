# libjinfer: jinfer as a shared library with the C API in include/jinfer.h. `make help` lists the targets.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
MAVEN ?= mvn
OPT   := $(if $(QUICK),-Ob,-O3)
OS    := $(shell uname -s | tr A-Z a-z)
ARCH  := $(shell uname -m | sed -e 's/x86_64/x86-64/' -e 's/arm64/aarch64/')
# The ISA level the image is built for; the kernels' vector width follows it (VectorWidth, at
# build time). x86-64-v3 (AVX2, 2013) is GraalVM's own default on x86-64, spelled out because the
# plugin cannot leave the flag out, and the floor: below it the kernels are not usable.
MARCH ?= $(if $(QUICK),native,$(if $(filter x86-64,$(ARCH)),x86-64-v3,compatibility))
PLATFORM := $(OS)-$(ARCH)
LIBFILE := $(if $(filter darwin,$(OS)),libjinfer.dylib,libjinfer.so)
LIB   := target/$(LIBFILE)
VERSION := $(shell grep -oE 'JINFER_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' include/jinfer.h | awk '{print $$2}' | paste -sd.)
DIST  := libjinfer-$(VERSION)-$(PLATFORM)
# The wheel is pure Python over the library it carries: py3-none, tagged with Python's own platform
# name, where a Linux wheel states the glibc floor read off the library, so it cannot drift from what was linked.
GLIBC = $(shell objdump -T $(LIB) | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1 | sed 's/GLIBC_//; s/\./_/')
WHEEL_PLAT = $(shell python3 -c 'import sysconfig; print(sysconfig.get_platform())' | sed 's/^linux-/manylinux_$(GLIBC)_/')
.DEFAULT_GOAL := help

help:
	@echo 'make libjinfer             -O3 -march=$(MARCH) (GraalVM'"'"'s default), what ships -> $(LIB); MARCH=x86-64-v4 for AVX-512'
	@echo 'make libjinfer QUICK=1     this machine, quick build (-Ob -march=native): runs, kernels unoptimized, never measure it'
	@echo 'make check                 header vs exports, C example, Python test'
	@echo 'make qa                    the corner cases: threads, exit, soak, offline, tools'
	@echo 'make python                stage $(LIB) into python/jinfer/lib/$(PLATFORM)/'
	@echo 'make wheel                 the wheel, tagged with the platform and glibc floor of $(LIB) -> target/wheel/'
	@echo 'make check-wheel           the wheel as a user gets it: twine check, a fresh venv, pip install, the Python test from outside the checkout'
	@echo 'make publish               twine upload to PyPI after check-wheel; TWINE_REPOSITORY=testpypi rehearses'
	@echo 'make dist                  release bundle: header, library, README, LICENSE, wheel -> target/dist/$(DIST).zip'

libjinfer:
	@$(MAKE) -s -C $(ROOT) check-native-image
	cd $(ROOT) && $(MAVEN) -Pnative -pl jinfer/jinfer-native -am package -DskipTests \
	  -Djinfer.image.opt=$(OPT) -Djinfer.image.march=$(MARCH)

TOKNROLL_INCLUDE := $(ROOT)/toknroll/toknroll-native/include

# Every function the two headers declare is exported, and nothing jinfer_* or toknroll_* beyond them is.
check-exports:
	@grep -hE '^(JINFER|TOKNROLL)_API' include/jinfer.h $(TOKNROLL_INCLUDE)/toknroll.h | grep -oE '(jinfer|toknroll)_[a-z_]+\(' | tr -d '(' | sort > target/header.syms
	@nm -D --defined-only $(LIB) | awk '{print $$3}' | grep -E '^(jinfer|toknroll)_' | sort > target/exported.syms
	@diff target/header.syms target/exported.syms && echo "exports match the headers: $$(wc -l < target/header.syms) functions"

target/chat: examples/c/chat.c include/jinfer.h $(TOKNROLL_INCLUDE)/toknroll.h $(LIB)
	cc -std=c99 -Wall -Wextra -Iinclude -I$(TOKNROLL_INCLUDE) -o $@ $< -L target -ljinfer -Wl,-rpath,$(abspath target)

check-version: ## the header, the wheel and the pom name the same release (__version__ is the library's own string)
	@h=$$(grep -oE 'JINFER_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' include/jinfer.h | awk '{printf "%s%s", (NR>1?".":""), $$2}'); \
	 w=$$(grep -oE '^version = "[^"]+"' python/pyproject.toml | cut -d'"' -f2); \
	 m=$$(grep -oE '<version>[^<]+</version>' pom.xml | head -1 | sed 's/<[^>]*>//g'); \
	 [ "$$h" = "$$w" ] && [ "$$h" = "$$m" ] && echo "version $$h everywhere" || { echo "version drift: header $$h, wheel $$w, pom $$m"; exit 1; }

check: check-version check-exports target/chat
	./target/chat
	python3 python/test_jinfer.py

qa:
	python3 python/qa.py

python: $(LIB)
	mkdir -p python/jinfer/lib/$(PLATFORM) && cp $(LIB) python/jinfer/lib/$(PLATFORM)/

# The release tools, one venv for both packages: build makes the wheel, wheel tags it with the
# platform, twine checks and uploads it, auditwheel audits the manylinux claim.
TOOLS := $(ROOT)/target/pytools
$(TOOLS)/bin/twine:
	python3 -m venv $(TOOLS) && $(TOOLS)/bin/pip install -q build wheel twine auditwheel

wheel: python $(TOOLS)/bin/twine
	cp $(ROOT)/LICENSE python/LICENSE
	rm -rf target/wheel python/build && $(TOOLS)/bin/python -m build --wheel --outdir target/wheel python
	$(TOOLS)/bin/wheel tags --remove --platform-tag $(WHEEL_PLAT) target/wheel/*.whl

# The wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv with the
# toknroll wheel it depends on, tested from outside the checkout.
check-wheel: wheel
	$(MAKE) -s -C $(ROOT)/toknroll/toknroll-native wheel
	$(TOOLS)/bin/twine check --strict target/wheel/*.whl
	$(if $(filter linux,$(OS)),$(TOOLS)/bin/auditwheel show target/wheel/*.whl)   # "unsupported ISA extensions" is the x86-64-v3 floor, known and documented
	rm -rf target/wheelenv && python3 -m venv target/wheelenv && target/wheelenv/bin/pip install -q target/wheel/*.whl $(ROOT)/toknroll/toknroll-native/target/wheel/*.whl
	cd target && JINFER_INSTALLED=1 wheelenv/bin/python ../python/test_jinfer.py
	@unzip -l target/wheel/*.whl | grep -q LICENSE && echo "the wheel carries the license"

# TWINE_REPOSITORY=testpypi rehearses; credentials come from ~/.pypirc or TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-...
publish: check-wheel
	$(TOOLS)/bin/twine upload target/wheel/*.whl

# What a GitHub release carries per platform. The wheel rides along so pip users need no second download.
dist: wheel
	rm -rf target/dist/$(DIST) && mkdir -p target/dist/$(DIST)/include target/dist/$(DIST)/lib/$(PLATFORM) target/dist/$(DIST)/python
	cp include/jinfer.h $(TOKNROLL_INCLUDE)/toknroll.h target/dist/$(DIST)/include/
	cp $(LIB) target/dist/$(DIST)/lib/$(PLATFORM)/
	cp README.md $(ROOT)/LICENSE target/dist/$(DIST)/
	cp target/wheel/*.whl target/dist/$(DIST)/python/
	cd target/dist && rm -f $(DIST).zip && zip -qr $(DIST).zip $(DIST) && sha256sum $(DIST).zip > $(DIST).zip.sha256
	@ls -la target/dist/$(DIST).zip && cat target/dist/$(DIST).zip.sha256

.PHONY: help libjinfer check check-version check-exports qa python wheel check-wheel publish dist
