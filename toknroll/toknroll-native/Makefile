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
SYSLIBS := -lz -ldl -lpthread -lm $(if $(filter darwin,$(OS)),-framework Foundation,-lrt)   # what the image needs from the system; the shared library carries these as NEEDED
# the exported (defined, global) symbol names of a shared library or archive; Mach-O names lose their underscore
EXPORTS = $(if $(filter darwin,$(OS)),nm -gU $(1) | awk '{print $$3}' | sed 's/^_//',nm -D --defined-only $(1) | awk '{print $$3}')
# what `make check` tokenizes: a hub repository, metadata only
SOURCE ?= Qwen/Qwen3-0.6B
VERSION := $(shell grep -oE 'TOKNROLL_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' include/toknroll.h | awk '{print $$2}' | paste -sd. -)
DIST  := libtoknroll-$(VERSION)-$(PLATFORM)
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

help:
	@echo 'make libtoknroll           -O2 -march=$(MARCH), what ships -> $(LIB) and $(STATIC)'
	@echo 'make libtoknroll QUICK=1   this machine, quick build (-Ob -march=native)'
	@echo 'make check                 header vs exports, C example, Python test'
	@echo 'make python                stage $(LIB) into python/toknroll/lib/$(PLATFORM)/'
	@echo 'make wheel                 the wheel, tagged with the platform and glibc floor of $(LIB) -> target/wheel/'
	@echo 'make check-wheel           the wheel as a user gets it: twine check, a fresh venv, pip install, the suite from outside the tree'
	@echo 'make publish               twine upload to PyPI after check-wheel; TWINE_REPOSITORY=testpypi rehearses'

libtoknroll:
	@$(MAKE) -s -C $(ROOT) check-native-image
	rm -f target/cc.log
	cd $(ROOT) && TOKNROLL_CC_LOG=$(abspath target/cc.log) $(MAVEN) -Pnative -pl toknroll/toknroll-native -am package -DskipTests \
	  -Dtoknroll.image.opt=$(OPT) -Dtoknroll.image.march=$(MARCH) $(MAVEN_ARGS)
	src/main/c/static.sh include/toknroll.h target/cc.log $(STATIC)

# Every function the header declares is exported, and nothing toknroll_* beyond the header is.
check-exports:
	@grep -E '^TOKNROLL_API' include/toknroll.h | grep -oE 'toknroll_[a-z_]+\(' | tr -d '(' | sort > target/header.syms
	@$(call EXPORTS,$(LIB)) | grep -E '^toknroll_' | sort > target/exported.syms
	@diff target/header.syms target/exported.syms && echo "exports match the header: $$(wc -l < target/header.syms) functions"

target/tokenize: examples/c/tokenize.c include/toknroll.h $(LIB)
	cc -std=c99 -Wall -Wextra -Iinclude -o $@ $< -L target -ltoknroll -Wl,-rpath,$(abspath target)

# The static archive: the example linked against it, no shared library in sight, and only toknroll_* exported.
target/tokenize-static: examples/c/tokenize.c include/toknroll.h $(STATIC)
	cc -std=c99 -Wall -Wextra -Iinclude -o $@ $< $(STATIC) $(SYSLIBS)

check-static: target/tokenize-static
	@grep -E '^TOKNROLL_API' include/toknroll.h | grep -oE 'toknroll_[a-z_]+\(' | tr -d '(' | sort > target/header.syms
	@nm --defined-only -g $(STATIC) | awk 'NF == 3 {print $$3}' | sed 's/^_//' | sort > target/static.syms   # NF: the member header and blank lines nm prints for an archive
	@diff target/header.syms target/static.syms && echo "the static archive exports the header's $$(wc -l < target/header.syms) functions and nothing else"
	@! $(if $(filter darwin,$(OS)),otool -L,ldd) target/tokenize-static | grep -q toknroll && echo "the static example needs no libtoknroll"
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

# The release tools, one venv for both packages: build makes the wheel, wheel tags it with the platform,
# twine checks and uploads it, auditwheel audits the manylinux claim. One venv per Python, since a venv is
# bound to the interpreter that made it and a build container's differs from the host's.
TOOLS := $(ROOT)/target/pytools/$(shell python3 -c 'import sys, sysconfig; print(sysconfig.get_platform(), *sys.version_info[:2], sep="-")')
$(TOOLS)/bin/twine:
	rm -rf $(TOOLS) && python3 -m venv $(TOOLS) && $(TOOLS)/bin/pip install -q build wheel twine auditwheel

wheel: python $(TOOLS)/bin/twine
	cp $(ROOT)/LICENSE python/LICENSE
	rm -rf target/wheel python/build && $(TOOLS)/bin/python -m build --wheel --outdir target/wheel python
	$(TOOLS)/bin/wheel tags --remove --platform-tag $(WHEEL_PLAT) target/wheel/*.whl

# The wheel as a user gets it: metadata as PyPI renders it, installed into a fresh venv, tested from outside the checkout.
check-wheel: wheel
	$(TOOLS)/bin/twine check --strict target/wheel/*.whl
	$(if $(filter linux,$(OS)),$(TOOLS)/bin/auditwheel show target/wheel/*.whl)
	rm -rf target/wheelenv && python3 -m venv target/wheelenv && target/wheelenv/bin/pip install -q target/wheel/*.whl
	cd target && TOKNROLL_INSTALLED=1 TOKNROLL_TEST_GGUF=$(TOKNROLL_TEST_GGUF) wheelenv/bin/python -m unittest discover -s ../python/tests -t ../python/tests
	@unzip -l target/wheel/*.whl | grep -q LICENSE && echo "the wheel carries the license"

# TWINE_REPOSITORY=testpypi rehearses; credentials come from ~/.pypirc or TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-...
publish: check-wheel
	$(TOOLS)/bin/twine upload target/wheel/*.whl

# What a GitHub release carries per platform: both libraries, the header, the wheel.
dist: wheel
	rm -rf target/dist/$(DIST) && mkdir -p target/dist/$(DIST)/include target/dist/$(DIST)/lib/$(PLATFORM) target/dist/$(DIST)/python
	cp include/toknroll.h target/dist/$(DIST)/include/
	cp $(LIB) $(STATIC) target/dist/$(DIST)/lib/$(PLATFORM)/
	cp README.md $(ROOT)/LICENSE target/dist/$(DIST)/
	cp target/wheel/*.whl target/dist/$(DIST)/python/
	cd target/dist && rm -f $(DIST).zip && zip -qr $(DIST).zip $(DIST) && sha256sum $(DIST).zip > $(DIST).zip.sha256
	@ls -la target/dist/$(DIST).zip && cat target/dist/$(DIST).zip.sha256

.PHONY: help libtoknroll check check-version check-exports check-static python wheel check-wheel publish dist
