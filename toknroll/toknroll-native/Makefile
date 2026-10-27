# libtoknroll: toknroll as a shared library with the C API in include/toknroll.h. `make help` lists the targets.
NAME  := toknroll
OPT   := $(if $(QUICK),-Ob,-O2)
MARCH ?= $(if $(QUICK),native,compatibility)
STATIC := target/libtoknroll.a
DIST_LIBS = $(LIB) $(STATIC)
SYSLIBS := -lz -ldl -lpthread -lm $(if $(filter darwin,$(OS)),-framework Foundation,-lrt)   # what the image needs from the system; the shared library carries these as NEEDED
WHEEL_TEST = TOKNROLL_INSTALLED=1 TOKNROLL_TEST_GGUF=$(TOKNROLL_TEST_GGUF) wheelenv/bin/python -m unittest discover -s ../python/tests -t ../python/tests
SOURCE ?= Qwen/Qwen3-0.6B   # what `make check` tokenizes: a hub repository, metadata only

help::
	@echo 'make libtoknroll           -O2 -march=$(MARCH), what ships -> $(LIB) and $(STATIC)'
	@echo 'make libtoknroll QUICK=1   this machine, quick build (-Ob -march=native)'
	@echo 'make check                 header vs exports, C examples (shared and static), Python tests'
include ../../native.mk

libtoknroll:
	@$(MAKE) -s -C $(ROOT) check-native-image
	rm -f target/cc.log
	cd $(ROOT) && NATIVE_CC_LOG=$(abspath target/cc.log) $(MAVEN) $(MAVEN_FLAGS) -Pnative -pl toknroll/toknroll-native -am package -DskipTests \
	  -Dtoknroll.image.opt=$(OPT) -Dtoknroll.image.march=$(MARCH)
	src/main/c/static.sh include/toknroll.h target/cc.log $(STATIC)

target/tokenize: examples/c/tokenize.c include/toknroll.h $(LIB)
	cc -std=c99 -Wall -Wextra -Wpedantic -Iinclude -o $@ $< -L target -ltoknroll -Wl,-rpath,$(abspath target)

# The static archive: the example linked against it, no shared library in sight, and only toknroll_* exported.
target/tokenize-static: examples/c/tokenize.c include/toknroll.h $(STATIC)
	cc -std=c99 -Wall -Wextra -Wpedantic -Iinclude -o $@ $< $(STATIC) $(SYSLIBS)

check-static: target/tokenize-static
	@nm --defined-only -g $(STATIC) | awk 'NF == 3 {print $$3}' | sed 's/^_//' | sort > target/static.syms   # NF: the member header and blank lines nm prints for an archive
	@diff target/header.syms target/static.syms && echo "the static archive exports the header's $$(wc -l < target/header.syms | tr -d " ") functions and nothing else"
	@! $(if $(filter darwin,$(OS)),otool -L,ldd) target/tokenize-static | grep -q toknroll && echo "the static example needs no libtoknroll"
	./target/tokenize-static $(SOURCE) "Hello, Tok'n'Roll!"

check: check-version check-exports target/tokenize check-static
	./target/tokenize $(SOURCE) "Hello, Tok'n'Roll!"
	cd python && python3 -m unittest discover -s tests -t tests

.PHONY: libtoknroll check check-static
