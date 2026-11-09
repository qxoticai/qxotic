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
# The soname follows the version, never set by hand: the whole version while the major is 0, as ABI
# compatibility is not guaranteed between any two 0.x releases (a program loads the release it was built
# against and no other), the major from 1.0. The build names the library by it (native-cc.sh) and links
# target/$(SONAME) to it, so a program linked against target/ runs from there.
MAJOR := $(word 1,$(subst ., ,$(VERSION)))
SOVERSION := $(if $(filter 0,$(MAJOR)),$(VERSION),$(MAJOR))
SONAME := $(if $(filter darwin,$(OS)),lib$(NAME).$(SOVERSION).dylib,lib$(NAME).so.$(SOVERSION))
NATIVE_IMAGE ?= $(if $(JAVA_HOME),$(JAVA_HOME)/bin/native-image,native-image)
# native-image's options: the caller's (NATIVE_IMAGE_OPTIONS=-march=x86-64-v4 asks for another target than
# native-image's own default), then QUICK=1's: this machine
IMAGE_OPTIONS = $(strip $(NATIVE_IMAGE_OPTIONS) $(if $(QUICK),-march=native))
# What the including Makefile may widen (no trailing comments here: make keeps the spaces before one):
# SYMS, the prefixes of the functions the headers declare; HEADERS, the headers the release carries;
# DESCRIPTION, the pkg-config file's line; WHEEL_EXAMPLE and WHEEL_EXAMPLE_ARGS, the C example built against
# the installed wheel and how it is run;
# EXPORT_HEADERS, those whose functions the library exports; EXTRA_EXPORTS, the exported names no header
# declares; DIST_LIBS, what the release bundle carries beside the headers; WHEEL_DEPS, other wheels the
# installed one needs; WHEEL_TEST, the tests run from target/ with wheelenv's python.
SYMS ?= $(NAME)
HEADERS ?= include/$(NAME).h
DESCRIPTION ?= lib$(NAME)
WHEEL_EXAMPLE ?= examples/c/$(NAME).c
WHEEL_EXAMPLE_ARGS ?=
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
	@echo 'make python                stage $(LIB), the headers and a pkg-config file into python/$(NAME)/'
	@echo 'make wheel                 the wheel, tagged with the platform and glibc floor of $(LIB) -> target/wheel/'
	@echo 'make check-wheel           the wheel as a user gets it: twine check, a fresh venv, pip install, the tests from outside the tree'
	@echo 'make dist                  the release tarball, a prefix tree: include/, lib/ with the library, its soname and .pc, README, NOTICE, LICENSE -> target/dist/$(DIST).tar.gz'
	@echo 'make check-dist            the tarball unpacked to a prefix of its own, the C example built with pkg-config alone'
	@echo 'make publish-dist TAG=<tag>   the tarball and its checksum as assets of that GitHub release (gh)'
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

# The package carries this platform's library alone, under its soname, with the headers and a pkg-config file
# beside it ($(NAME)-config says where): a wheel holds what its tag names, whatever else this tree built,
# and runs on the tag's whole platform, so never a library built for this machine alone (QUICK=1)
python: $(LIB)
	@if readelf -n $(LIB) 2>/dev/null | grep -q 'x86-64-v4'; then \
	    echo "$(LIB) needs x86-64-v4 (QUICK=1?): a package needs the x86-64-v3 floor, rebuild without QUICK"; exit 1; fi
	rm -rf python/$(NAME)/lib python/$(NAME)/include && mkdir -p python/$(NAME)/lib/$(PLATFORM)/pkgconfig python/$(NAME)/include
	cp $(LIB) python/$(NAME)/lib/$(PLATFORM)/$(SONAME) && cp $(HEADERS) python/$(NAME)/include/   # by its soname: a C program linked against it loads it by that name
	@printf '%s\n' 'prefix=$${pcfiledir}/../../..' 'includedir=$${prefix}/include' 'libdir=$${pcfiledir}/..' '' \
	    'Name: $(NAME)' 'Description: $(DESCRIPTION)' 'Version: $(VERSION)' \
	    'Cflags: -I$${includedir}' 'Libs: $${libdir}/$(SONAME) -Wl,-rpath,$${libdir}' > python/$(NAME)/lib/$(PLATFORM)/pkgconfig/$(NAME).pc
	cp $(ROOT)/LICENSE NOTICE python/   # pyproject's license-files: a pip install of python/ carries them, as the wheel does

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
	@python3 -m zipfile -l target/wheel/*.whl | grep -q LICENSE && python3 -m zipfile -l target/wheel/*.whl | grep -q NOTICE && echo "the wheel carries the license and the notice"
	@CC="$(CC)" python3 -c 'import os, shlex, subprocess, sys; flags = subprocess.check_output([sys.argv[1], "--cflags", "--libs"], text=True).splitlines(); subprocess.run(shlex.split(os.environ["CC"]) + ["-std=c99", "-o", "target/wheel-example", sys.argv[2], *flags], check=True)' \
	 "$(abspath target/wheelenv/bin/$(NAME)-config)" "$(WHEEL_EXAMPLE)"
	./target/wheel-example $(WHEEL_EXAMPLE_ARGS) > /dev/null
	@echo "the C example builds and runs against the installed wheel: $(NAME)-config --cflags --libs"
	@if command -v pkg-config > /dev/null; then \
	 pc=$$(target/wheelenv/bin/$(NAME)-config --pkgconfigdir) && PKG_CONFIG_PATH="$$pc" pkg-config --exists $(NAME); \
	 else echo "no pkg-config here: the .pc file not checked"; fi

# What a release carries per platform, for C and anyone without Python (the wheel is its own delivery): a
# prefix tree, include/, lib/ with the library under its soname, the linker's name linked to it, the static
# archive where there is one, and lib/pkgconfig/$(NAME).pc. Unpacked into /usr/local pkg-config finds it with
# no setup; elsewhere PKG_CONFIG_PATH=<prefix>/lib/pkgconfig. The platform is in the tarball's name, not inside it.
dist: $(LIB)
	rm -rf target/dist/$(DIST) && mkdir -p target/dist/$(DIST)/include target/dist/$(DIST)/lib/pkgconfig
	cp $(HEADERS) target/dist/$(DIST)/include/
	cp $(DIST_LIBS) target/dist/$(DIST)/lib/
	cd target/dist/$(DIST)/lib && mv $(notdir $(LIB)) $(SONAME) && ln -s $(SONAME) $(notdir $(LIB))
	@printf '%s\n' 'prefix=$${pcfiledir}/../..' 'includedir=$${prefix}/include' 'libdir=$${prefix}/lib' '' \
	    'Name: $(NAME)' 'Description: $(DESCRIPTION)' 'Version: $(VERSION)' \
	    'Cflags: -I$${includedir}' 'Libs: -L$${libdir} -l$(NAME) -Wl,-rpath,$${libdir}' > target/dist/$(DIST)/lib/pkgconfig/$(NAME).pc
	cp README.md NOTICE $(ROOT)/LICENSE target/dist/$(DIST)/
	cd target/dist && rm -f $(DIST).tar.gz && tar -czf $(DIST).tar.gz $(DIST) && sha256sum $(DIST).tar.gz > $(DIST).tar.gz.sha256   # a tarball: the Unix form, symlinks and modes kept; Windows' is a zip
	@ls -la target/dist/$(DIST).tar.gz && cat target/dist/$(DIST).tar.gz.sha256

# The tarball as a user gets it: a prefix with spaces, the C example built with pkg-config alone and run.
check-dist: dist
	rm -rf "target/dist env" && mkdir -p "target/dist env" && cd "target/dist env" && tar -xzf ../dist/$(DIST).tar.gz
	@if ! command -v pkg-config > /dev/null; then echo "no pkg-config here: the tarball not checked"; exit 0; fi; \
	 CC="$(CC)" PKG_CONFIG_PATH="$(CURDIR)/target/dist env/$(DIST)/lib/pkgconfig" \
	 python3 -c 'import os, shlex, subprocess, sys; flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", sys.argv[1]], text=True)); subprocess.run(shlex.split(os.environ["CC"]) + ["-std=c99", "-o", "target/dist-example", sys.argv[2], *flags], check=True)' "$(NAME)" "$(WHEEL_EXAMPLE)" && \
	 ./target/dist-example $(WHEEL_EXAMPLE_ARGS) > target/dist-example.out 2>&1 && ! grep -q "did not load" target/dist-example.out && \
	 echo "the C example builds and runs against the unpacked $(DIST) with pkg-config alone"

# The tarball to the release on GitHub, as an asset of the tag that names this version: make publish-dist TAG=<tag>
publish-dist:
	@tag="$(TAG)"; [ -n "$$tag" ] || { echo "make publish-dist TAG=<tag>, the release's tag"; exit 1; }; \
	 case "$$tag" in *v$(VERSION)) ;; *) echo "tag $$tag does not name version $(VERSION), the tarball's"; exit 1;; esac; \
	 [ -f target/dist/$(DIST).tar.gz ] || { echo "no target/dist/$(DIST).tar.gz: make dist"; exit 1; }; \
	 (cd target/dist && sha256sum -c $(DIST).tar.gz.sha256 --quiet --status) || { echo "the tarball does not match its checksum"; exit 1; }; \
	 gh release upload "$$tag" target/dist/$(DIST).tar.gz target/dist/$(DIST).tar.gz.sha256 --clobber && echo "uploaded $(DIST).tar.gz to $$tag"

.PHONY: help check-version check-exports python wheel check-wheel dist check-dist publish-dist
