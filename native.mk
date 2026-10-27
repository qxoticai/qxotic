# What the two native libraries' Makefiles share (toknroll/toknroll-native, jinfer/jinfer-native): the
# platform, the wheel's tag, the Python release tools, and the targets that read the same for both. The
# including Makefile sets NAME first (its library is lib$(NAME), its header include/$(NAME).h, its Python
# package python/$(NAME)) and defines lib$(NAME), check and its own help:: lines.
ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MAVEN ?= mvn
OS    := $(shell uname -s | tr A-Z a-z)
ARCH  := $(shell uname -m | sed -e 's/x86_64/x86-64/' -e 's/arm64/aarch64/')
PLATFORM := $(OS)-$(ARCH)
UPPER := $(shell echo $(NAME) | tr a-z A-Z)
LIB   := target/lib$(NAME).$(if $(filter darwin,$(OS)),dylib,so)
VERSION := $(shell grep -oE '$(UPPER)_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' include/$(NAME).h | awk '{print $$2}' | paste -sd. -)
DIST  := lib$(NAME)-$(VERSION)-$(PLATFORM)
# What the including Makefile may widen (no trailing comments here: make keeps the spaces before one):
# SYMS, the exported symbol prefixes the headers declare; HEADERS, the headers whose functions the library
# exports; DIST_LIBS, what the release bundle carries beside the headers; WHEEL_DEPS, other wheels the
# installed one needs; WHEEL_TEST, the tests run from target/ with wheelenv's python.
SYMS ?= $(NAME)
HEADERS ?= include/$(NAME).h
DIST_LIBS ?= $(LIB)
WHEEL_DEPS ?=
# the exported (defined, global) symbol names of a shared library or archive; Mach-O names lose their underscore
EXPORTS = $(if $(filter darwin,$(OS)),nm -gU $(1) | awk '{print $$3}' | sed 's/^_//',nm -D --defined-only $(1) | awk '{print $$3}')
# The wheel is pure Python over the library it carries: py3-none, tagged with the platform. The tag's
# floor is read off the library itself, the glibc version on Linux and the minimum macOS on a Mac, so it
# cannot drift from what was linked; the rest is Python's own platform name.
GLIBC = $(shell objdump -T $(LIB) | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1 | sed 's/GLIBC_//; s/\./_/')
MINOS = $(shell otool -l $(LIB) | awk '/LC_BUILD_VERSION/ {v=1} v && /minos/ {print $$2; exit}' | tr . _)
WHEEL_PLAT = $(shell python3 -c 'import sysconfig; print(sysconfig.get_platform())' | sed 's/^linux-/manylinux_$(GLIBC)_/; s/^macosx-[0-9.]*-/macosx_$(MINOS)_/')
# macOS: what the library and the wheel tag promise, 14.0 being what GraalVM 25's own darwin runtime objects
# are built for. Reaches the pom's cc through the environment and native-image's linker through -E; empty
# elsewhere, which compilers ignore. (No trailing comment: Apple's make would hand clang the spaces.)
MACOSX_DEPLOYMENT_TARGET ?= $(if $(filter darwin,$(OS)),14.0)
export MACOSX_DEPLOYMENT_TARGET
.DEFAULT_GOAL := help

help::
	@echo 'make python                stage $(LIB) into python/$(NAME)/lib/$(PLATFORM)/'
	@echo 'make wheel                 the wheel, tagged with the platform and glibc floor of $(LIB) -> target/wheel/'
	@echo 'make check-wheel           the wheel as a user gets it: twine check, a fresh venv, pip install, the tests from outside the tree'
	@echo 'make dist                  release bundle: headers, library, README, LICENSE, wheel -> target/dist/$(DIST).zip'
	@echo 'All platforms at once, and PyPI: make wheels, make publish-wheels at the root.'

check-version:   # the header, the wheel, the pom and (where it is a literal) the package name the same release
	@h=$$(grep -oE '$(UPPER)_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' include/$(NAME).h | awk '{printf "%s%s", (NR>1?".":""), $$2}'); \
	 w=$$(grep -oE '^version = "[^"]+"' python/pyproject.toml | cut -d'"' -f2); \
	 m=$$(grep -oE '<version>[^<]+</version>' pom.xml | head -1 | sed 's/<[^>]*>//g'); \
	 p=$$(grep -oE '^__version__ = "[^"]+"' python/$(NAME)/__init__.py | cut -d'"' -f2); \
	 [ "$$h" = "$$w" ] && [ "$$h" = "$$m" ] && [ "$$h" = "$${p:-$$h}" ] && echo "version $$h everywhere" || { echo "version drift: header $$h, wheel $$w, pom $$m, package $$p"; exit 1; }

# Every function the headers declare is exported, and nothing with the headers' prefixes beyond them is.
check-exports:
	@grep -hE '^[A-Z]+_API' $(HEADERS) | grep -oE '($(SYMS))_[a-z_]+\(' | tr -d '(' | sort > target/header.syms
	@$(call EXPORTS,$(LIB)) | grep -E '^($(SYMS))_' | sort > target/exported.syms
	@diff target/header.syms target/exported.syms && echo "exports match the headers: $$(wc -l < target/header.syms | tr -d " ") functions"

python: $(LIB)
	mkdir -p python/$(NAME)/lib/$(PLATFORM) && cp $(LIB) python/$(NAME)/lib/$(PLATFORM)/

# The release tools, one venv for both packages: build makes the wheel, wheel tags it with the platform,
# twine checks and uploads it, auditwheel audits the manylinux claim. One venv per Python, since a venv is
# bound to the interpreter that made it and a build container's differs from the host's.
TOOLS := $(ROOT)/target/pytools/$(shell python3 -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep="-")')
$(TOOLS)/bin/twine:
	rm -rf $(TOOLS) && python3 -m venv $(TOOLS) && $(TOOLS)/bin/pip install -q build wheel twine auditwheel

wheel: python $(TOOLS)/bin/twine
	cp $(ROOT)/LICENSE python/LICENSE
	rm -rf target/wheel python/build && $(TOOLS)/bin/python -m build --wheel --outdir target/wheel python
	$(TOOLS)/bin/wheel tags --remove --python-tag py3 --abi-tag none --platform-tag $(WHEEL_PLAT) target/wheel/*.whl   # ctypes over a C library: any Python 3

# The wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv with the wheels it
# depends on, tested from outside the checkout ($(WHEEL_TEST), run in target/ with wheelenv's python).
check-wheel: wheel
	$(TOOLS)/bin/twine check --strict target/wheel/*.whl
	$(if $(filter linux,$(OS)),$(TOOLS)/bin/auditwheel show target/wheel/*.whl)   # "unsupported ISA extensions" is jinfer's x86-64-v3 floor, known and documented
	rm -rf target/wheelenv && python3 -m venv target/wheelenv && target/wheelenv/bin/pip install -q target/wheel/*.whl $(WHEEL_DEPS)
	cd target && $(WHEEL_TEST)
	@unzip -l target/wheel/*.whl | grep -q LICENSE && echo "the wheel carries the license"

# What a release carries per platform: headers, libraries, the wheel, so pip users need no second download.
dist: wheel
	rm -rf target/dist/$(DIST) && mkdir -p target/dist/$(DIST)/include target/dist/$(DIST)/lib/$(PLATFORM) target/dist/$(DIST)/python
	cp $(HEADERS) target/dist/$(DIST)/include/
	cp $(DIST_LIBS) target/dist/$(DIST)/lib/$(PLATFORM)/
	cp README.md $(ROOT)/LICENSE target/dist/$(DIST)/
	cp target/wheel/*.whl target/dist/$(DIST)/python/
	cd target/dist && rm -f $(DIST).zip && zip -qr $(DIST).zip $(DIST) && sha256sum $(DIST).zip > $(DIST).zip.sha256
	@ls -la target/dist/$(DIST).zip && cat target/dist/$(DIST).zip.sha256

.PHONY: help check-version check-exports python wheel check-wheel dist
