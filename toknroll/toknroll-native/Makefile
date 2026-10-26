# libtoknroll: toknroll as a shared library with the C API in include/toknroll.h. `make help` lists the targets.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
MAVEN ?= mvn
OPT   := $(if $(QUICK),-Ob,-O2)
OS    := $(shell uname -s | tr A-Z a-z)
ARCH  := $(shell uname -m | sed -e 's/x86_64/x86-64/' -e 's/arm64/aarch64/')
MARCH ?= $(if $(QUICK),native,compatibility)
PLATFORM := $(OS)-$(ARCH)
LIBFILE := $(if $(filter darwin,$(OS)),libtoknroll.dylib,libtoknroll.so)
LIB   := target/$(LIBFILE)
STATIC := target/libtoknroll.a
SYSLIBS := -lz -ldl -lpthread -lm -lrt   # what the image needs from the system; the shared library carries these as NEEDED
# what `make check` tokenizes: a hub repository, metadata only
SOURCE ?= Qwen/Qwen3-0.6B
VERSION := $(shell grep -oE 'TOKNROLL_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' include/toknroll.h | awk '{print $$2}' | paste -sd.)
DIST  := libtoknroll-$(VERSION)-$(PLATFORM)
# The wheel is pure Python over the library it carries: py3-none, tagged with Python's own platform
# name, where a Linux wheel states the glibc floor read off the library, so it cannot drift from what was linked.
GLIBC = $(shell objdump -T $(LIB) | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1 | sed 's/GLIBC_//; s/\./_/')
WHEEL_PLAT = $(shell python3 -c 'import sysconfig; print(sysconfig.get_platform())' | sed 's/^linux-/manylinux_$(GLIBC)_/')
.DEFAULT_GOAL := help

help:
	@echo 'make libtoknroll           -O2 -march=$(MARCH), what ships -> $(LIB) and $(STATIC)'
	@echo 'make libtoknroll QUICK=1   this machine, quick build (-Ob -march=native)'
	@echo 'make check                 header vs exports, C example, Python test'
	@echo 'make python                stage $(LIB) into python/toknroll/lib/$(PLATFORM)/'
	@echo 'make wheel                 the wheel, tagged with the platform and glibc floor of $(LIB) -> target/wheel/'
	@echo 'make check-wheel           the wheel as a user gets it: twine check, a fresh venv, pip install, the suite from outside the tree'

libtoknroll:
	@$(MAKE) -s -C $(ROOT) check-native-image
	rm -f target/cc.log
	cd $(ROOT) && TOKNROLL_CC_LOG=$(abspath target/cc.log) $(MAVEN) -Pnative -pl toknroll/toknroll-native -am package -DskipTests \
	  -Dtoknroll.image.opt=$(OPT) -Dtoknroll.image.march=$(MARCH) $(MAVEN_ARGS)
	src/main/c/static.sh include/toknroll.h target/cc.log $(STATIC)

# Every function the header declares is exported, and nothing toknroll_* beyond the header is.
check-exports:
	@grep -E '^TOKNROLL_API' include/toknroll.h | grep -oE 'toknroll_[a-z_]+\(' | tr -d '(' | sort > target/header.syms
	@nm -D --defined-only $(LIB) | awk '{print $$3}' | grep -E '^toknroll_' | sort > target/exported.syms
	@diff target/header.syms target/exported.syms && echo "exports match the header: $$(wc -l < target/header.syms) functions"

target/tokenize: examples/c/tokenize.c include/toknroll.h $(LIB)
	cc -std=c99 -Wall -Wextra -Iinclude -o $@ $< -L target -ltoknroll -Wl,-rpath,$(abspath target)

# The static archive: the example linked against it, no shared library in sight, and only toknroll_* exported.
target/tokenize-static: examples/c/tokenize.c include/toknroll.h $(STATIC)
	cc -std=c99 -Wall -Wextra -Iinclude -o $@ $< $(STATIC) $(SYSLIBS)

check-static: target/tokenize-static
	@grep -E '^TOKNROLL_API' include/toknroll.h | grep -oE 'toknroll_[a-z_]+\(' | tr -d '(' | sort > target/header.syms
	@nm --defined-only -g $(STATIC) | awk 'NF == 3 {print $$3}' | sort > target/static.syms   # NF: the member header and blank lines nm prints for an archive
	@diff target/header.syms target/static.syms && echo "the static archive exports the header's $$(wc -l < target/header.syms) functions and nothing else"
	@! ldd target/tokenize-static | grep -q toknroll && echo "the static example needs no libtoknroll"
	./target/tokenize-static $(SOURCE) "Hello, Tok'n'Roll!"

check-version: ## the header, the wheel, the package and the pom name the same release
	@h=$$(grep -oE 'TOKNROLL_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' include/toknroll.h | awk '{printf "%s%s", (NR>1?".":""), $$2}'); \
	 w=$$(grep -oE '^version = "[^"]+"' python/pyproject.toml | cut -d'"' -f2); \
	 p=$$(grep -oE '^__version__ = "[^"]+"' python/toknroll/__init__.py | cut -d'"' -f2); \
	 m=$$(grep -oE '<version>[^<]+</version>' pom.xml | head -1 | sed 's/<[^>]*>//g'); \
	 [ "$$h" = "$$w" ] && [ "$$h" = "$$p" ] && [ "$$h" = "$$m" ] && echo "version $$h everywhere" || { echo "version drift: header $$h, wheel $$w, package $$p, pom $$m"; exit 1; }

check: check-version check-exports target/tokenize check-static
	./target/tokenize $(SOURCE) "Hello, Tok'n'Roll!"
	cd python && python3 -m unittest discover -s tests -t tests

python: $(LIB)
	mkdir -p python/toknroll/lib/$(PLATFORM) && cp $(LIB) python/toknroll/lib/$(PLATFORM)/

wheel: python
	cp $(ROOT)/LICENSE python/LICENSE
	rm -rf target/wheel python/build && cd python && python3 -m pip wheel . -w ../target/wheel --no-deps -q -C--build-option=--plat-name=$(WHEEL_PLAT)

# The wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv, tested from outside the checkout.
check-wheel: wheel
	rm -rf target/wheelenv && python3 -m venv target/wheelenv && target/wheelenv/bin/pip install -q twine target/wheel/*.whl
	target/wheelenv/bin/twine check --strict target/wheel/*.whl
	cd target && TOKNROLL_INSTALLED=1 TOKNROLL_TEST_GGUF=$(TOKNROLL_TEST_GGUF) wheelenv/bin/python -m unittest discover -s ../python/tests -t ../python/tests
	@unzip -l target/wheel/*.whl | grep -q LICENSE && echo "the wheel carries the license"

# What a GitHub release carries per platform: both libraries, the header, the wheel.
dist: wheel
	rm -rf target/dist/$(DIST) && mkdir -p target/dist/$(DIST)/include target/dist/$(DIST)/lib/$(PLATFORM) target/dist/$(DIST)/python
	cp include/toknroll.h target/dist/$(DIST)/include/
	cp $(LIB) $(STATIC) target/dist/$(DIST)/lib/$(PLATFORM)/
	cp README.md $(ROOT)/LICENSE target/dist/$(DIST)/
	cp target/wheel/*.whl target/dist/$(DIST)/python/
	cd target/dist && rm -f $(DIST).zip && zip -qr $(DIST).zip $(DIST) && sha256sum $(DIST).zip > $(DIST).zip.sha256
	@ls -la target/dist/$(DIST).zip && cat target/dist/$(DIST).zip.sha256

.PHONY: help libtoknroll check check-version check-exports check-static python wheel check-wheel dist
