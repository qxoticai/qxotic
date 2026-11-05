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
# The soname follows the version, never set by hand: the minor while the major is 0 (any minor may break),
# the major from 1.0. The build names the library by it (native-cc.sh) and links target/$(SONAME) to it, so
# a program linked against target/ runs from there.
MAJOR := $(word 1,$(subst ., ,$(VERSION)))
SOVERSION := $(if $(filter 0,$(MAJOR)),0.$(word 2,$(subst ., ,$(VERSION))),$(MAJOR))
SONAME := $(if $(filter darwin,$(OS)),lib$(NAME).$(SOVERSION).dylib,lib$(NAME).so.$(SOVERSION))
NATIVE_IMAGE ?= $(if $(JAVA_HOME),$(JAVA_HOME)/bin/native-image,native-image)
# native-image's options: the caller's (NATIVE_IMAGE_OPTIONS=-march=x86-64-v4 asks for another target than
# native-image's own default), then QUICK=1's: this machine
IMAGE_OPTIONS = $(strip $(NATIVE_IMAGE_OPTIONS) $(if $(QUICK),-march=native))
# What the including Makefile may widen (no trailing comments here: make keeps the spaces before one):
# SYMS, the prefixes of the functions the headers declare; HEADERS, the headers the release carries;
# EXPORT_HEADERS, those whose functions the library exports; EXTRA_EXPORTS, the exported names no header
# declares; DIST_LIBS, what the release bundle carries beside the headers; WHEEL_DEPS, other wheels the
# installed one needs; WHEEL_TEST, the tests run from target/ with wheelenv's python.
SYMS ?= $(NAME)
HEADERS ?= include/$(NAME).h
EXPORT_HEADERS ?= $(HEADERS)
EXTRA_EXPORTS ?=
DIST_LIBS ?= $(LIB)
WHEEL_DEPS ?=
# the exported (defined, global) symbol names of a shared library or archive; Mach-O names lose their underscore
EXPORTS = $(if $(filter darwin,$(OS)),nm -gU $(1) | awk '{print $$3}' | sed 's/^_//',nm -D --defined-only $(1) | awk '{print $$3}')
# The wheel is Python (ctypes) over the library it carries: py3-none, tagged with the platform (setup.py
# reads WHEEL_PLAT). The tag's floor is read off the library itself, the glibc version on Linux and the minimum
# macOS on a Mac, so it cannot drift from what was linked; the rest is Python's own platform name. On Linux
# the floor is 2.28 at least, the oldest glibc supported: the container's, which a library built there may
# need less than (2.17 was measured), never more.
GLIBC = $(shell { echo GLIBC_2.28; objdump -T $(LIB) | grep -oE 'GLIBC_[0-9.]+'; } | sort -Vu | tail -1 | sed 's/GLIBC_//; s/\./_/')
MINOS = $(shell otool -l $(LIB) | awk '/LC_BUILD_VERSION/ {v=1} v && /minos/ {print $$2; exit}' | tr . _)
FLOOR = $(if $(filter darwin,$(OS)),$(MINOS),$(GLIBC))
WHEEL_PLAT = $(shell python3 -c 'import sysconfig; print(sysconfig.get_platform())' | sed 's/^linux-/manylinux_$(FLOOR)_/; s/^macosx-[0-9.]*-/macosx_$(FLOOR)_/')
# macOS: what the library and the wheel tag promise, 14.0 being what GraalVM 25's own darwin runtime objects
# are built for. Reaches the pom's cc through the environment and native-image's linker through -E; empty
# elsewhere, which compilers ignore. (No trailing comment: Apple's make would hand clang the spaces.)
MACOSX_DEPLOYMENT_TARGET ?= $(if $(filter darwin,$(OS)),14.0)
export MACOSX_DEPLOYMENT_TARGET
.DEFAULT_GOAL := help
# Each target is a step of one pipeline, Maven parallel inside; under -j two would share the tools venv
.NOTPARALLEL:

$(LIB):
	@echo "$(LIB) is not built: make lib$(NAME)" >&2; exit 1

help::
	@echo 'make python                stage $(LIB) into python/$(NAME)/lib/$(PLATFORM)/'
	@echo 'make wheel                 the wheel, tagged with the platform and glibc floor of $(LIB) -> target/wheel/'
	@echo 'make check-wheel           the wheel as a user gets it: twine check, a fresh venv, pip install, the tests from outside the tree'
	@echo 'make dist                  release bundle: headers, library, README, LICENSE, wheel -> target/dist/$(DIST).zip'
	@echo 'This machine'"'"'s two wheels from a clean export, and PyPI: make wheels, make publish-wheels at the root.'

check-version:   # the header, the wheel, the pom and (where it is a literal) the package name the same release
	@h=$(VERSION); \
	 w=$$(grep -oE '^version = "[^"]+"' python/pyproject.toml | cut -d'"' -f2); \
	 m=$$(grep -oE '<version>[^<]+</version>' pom.xml | head -1 | sed 's/<[^>]*>//g'); \
	 p=$$(grep -oE '^__version__ = "[^"]+"' python/$(NAME)/__init__.py | cut -d'"' -f2); \
	 [ "$$h" = "$$w" ] && [ "$$h" = "$$m" ] && [ "$$h" = "$${p:-$$h}" ] && echo "version $$h everywhere" || { echo "version drift: header $$h, wheel $$w, pom $$m, package $$p"; exit 1; }

# The library exports every function the headers declare, the EXTRA_EXPORTS, and nothing else: none of the
# runtime behind it (graal_*, the entry points), whose names would collide with another library's.
check-exports:
	@grep -hE '^[A-Z]+_API' $(EXPORT_HEADERS) | grep -oE '($(SYMS))_[a-z0-9_]+\(' | tr -d '(' | sort > target/header.syms
	@{ cat target/header.syms; printf '%s\n' $(EXTRA_EXPORTS); } | grep . | sort > target/expected.syms
	@$(call EXPORTS,$(LIB)) | sort > target/exported.syms
	@diff target/expected.syms target/exported.syms && echo "exports match the headers: $$(wc -l < target/header.syms | tr -d " ") functions$(if $(EXTRA_EXPORTS), and $(words $(EXTRA_EXPORTS)) more)"

# The package carries this platform's library alone: a wheel holds what its tag names, whatever else this tree built
python: $(LIB)
	rm -rf python/$(NAME)/lib && mkdir -p python/$(NAME)/lib/$(PLATFORM) && cp $(LIB) python/$(NAME)/lib/$(PLATFORM)/
	cp $(ROOT)/LICENSE python/LICENSE   # pyproject's license-files: a pip install of python/ carries it, as the wheel does

# The release tools, one venv for both packages (the root's publish-wheels makes the same one, by the same
# path, to upload with): pip builds the wheel, twine
# checks and uploads it, auditwheel audits the manylinux claim. One venv per Python, since a venv is bound to
# the interpreter that made it and a build container's differs from the host's.
TOOLS := $(ROOT)/target/pytools/$(shell python3 -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep="-")')
$(TOOLS)/bin/twine:
	rm -rf $(TOOLS) && python3 -m venv $(TOOLS) && $(TOOLS)/bin/pip install -q twine auditwheel

wheel: python $(TOOLS)/bin/twine
	rm -rf target/wheel python/build && WHEEL_PLAT=$(WHEEL_PLAT) $(TOOLS)/bin/pip wheel -q --no-deps -w target/wheel ./python

# The wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv with the wheels it
# depends on, tested from outside the checkout ($(WHEEL_TEST), run in target/ with wheelenv's python).
check-wheel: wheel
	$(TOOLS)/bin/twine check --strict target/wheel/*.whl
	$(if $(filter linux,$(OS)),$(TOOLS)/bin/auditwheel show target/wheel/*.whl)   # "unsupported ISA extensions": native-image's default target, x86-64-v3, the libraries' floor
	rm -rf target/wheelenv && python3 -m venv target/wheelenv && target/wheelenv/bin/pip install -q target/wheel/*.whl $(WHEEL_DEPS)
	cd target && $(WHEEL_TEST)
	@python3 -m zipfile -l target/wheel/*.whl | grep -q LICENSE && echo "the wheel carries the license"

# What a release carries per platform: headers, libraries, the wheel, so pip users need no second download.
dist: wheel
	rm -rf target/dist/$(DIST) && mkdir -p target/dist/$(DIST)/include target/dist/$(DIST)/lib/$(PLATFORM) target/dist/$(DIST)/python
	cp $(HEADERS) target/dist/$(DIST)/include/
	cp $(DIST_LIBS) target/dist/$(DIST)/lib/$(PLATFORM)/
	cd target/dist/$(DIST)/lib/$(PLATFORM) && mv $(notdir $(LIB)) $(SONAME) && ln -s $(SONAME) $(notdir $(LIB))   # the file by its soname, the name a linker looks for beside it
	cp README.md $(ROOT)/LICENSE target/dist/$(DIST)/
	cp target/wheel/*.whl target/dist/$(DIST)/python/
	cd target/dist && rm -f $(DIST).zip && zip -qry $(DIST).zip $(DIST) && sha256sum $(DIST).zip > $(DIST).zip.sha256
	@ls -la target/dist/$(DIST).zip && cat target/dist/$(DIST).zip.sha256

.PHONY: help check-version check-exports python wheel check-wheel dist
