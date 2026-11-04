# libtoknroll: toknroll as a shared library with the C API in include/toknroll.h. `make help` lists the targets.
NAME  := toknroll
# The optimization: the pom's (-O2) unless QUICK=1 or a PGO profile asks for another (Oracle GraalVM compiles a PGO build at -O3)
OPT    = $(if $(QUICK),-Ob,$(if $(PGO),--pgo=$(abspath $(PROFILE))))
STATIC := target/libtoknroll.a
DIST_LIBS = $(LIB) $(STATIC)
# What the image needs from the system: the shared library carries these as NEEDED, a static consumer links them
SYSLIBS = -lz -ldl -lpthread -lm $(if $(filter darwin,$(OS)),-framework Foundation,-lrt)
WHEEL_TEST = TOKNROLL_INSTALLED=1 TOKNROLL_TEST_GGUF=$(TOKNROLL_TEST_GGUF) wheelenv/bin/python -m unittest discover -s ../python/tests -t ../python/tests
# What `make check` tokenizes: a hub repository, metadata only
SOURCE ?= Qwen/Qwen3-0.6B

help::
	@echo 'make libtoknroll           what ships, native-image'"'"'s default target -> $(LIB) and $(STATIC)'
	@echo '                           Oracle GraalVM: PGO at -O3, from a profile recorded first (pgo-profile); CE: -O2'
	@echo 'make libtoknroll QUICK=1   this machine, quick build (-Ob -march=native)'
	@echo 'make check                 header vs exports, C tests and examples (shared and static), Python tests'
include ../../native.mk
# Oracle GraalVM builds what ships optimized for a profile it records first (PGO); GraalVM CE builds without
# one, and QUICK=1 skips it. The collector is the serial one: NATIVE_IMAGE_OPTIONS=--gc=G1 (Oracle, Linux) runs
# up to 70% faster on many cores, but a forked child that allocates hangs, its parent's GC threads gone.
PGO = $(if $(QUICK),,$(shell $(NATIVE_IMAGE) --version 2>/dev/null | grep -q 'Oracle GraalVM' && echo yes))
PROFILE := target/pgo/default.iprof
# The profile's workload: enwik8, where the tests find it, and PgoTraining's tokenizers (PGO_SOURCES, if set)
CORPUS = $(shell python3 -c "import runpy; print(runpy.run_path('$(ROOT)/toknroll/scripts/download_enwik8.py', run_name='lib')['default_corpus_dir']() / 'enwik8')")

libtoknroll:
	@$(MAKE) -s -C $(ROOT) check-native-image
	$(if $(PGO),$(MAKE) pgo-profile)
	rm -f target/cc.log
	cd $(ROOT) && NATIVE_IMAGE_OPTIONS="$(IMAGE_OPTIONS)" \
	  NATIVE_CC_LOG=$(abspath target/cc.log) NATIVE_SONAME=$(SONAME) $(MAVEN) $(MAVEN_FLAGS) -Pnative -pl toknroll/toknroll-native -am package -DskipTests \
	  $(if $(OPT),-Dtoknroll.image.opt=$(OPT))
	ln -sf $(notdir $(LIB)) target/$(SONAME)
	src/main/c/static.sh include/toknroll.h target/cc.log $(STATIC)

# The profile, recorded by an instrumented build of PgoTraining with the library's own feature and options,
# rebuilt every time so it follows the code. The reactor's classes, not the installed jars; the runtime
# classpath, the builder's own jars (provided) being the builder's to bring.
pgo-profile:
	@[ -f "$(CORPUS)" ] || python3 $(ROOT)/toknroll/scripts/download_enwik8.py
	mkdir -p target/pgo && cd $(ROOT) && $(MAVEN) $(MAVEN_FLAGS) -q -pl toknroll/toknroll-native -am test-compile \
	  dependency:build-classpath -DincludeScope=runtime -Dmdep.outputFile=$(abspath target/pgo/classpath)
	cc -std=c11 -O2 -fPIC -Wall -Iinclude -c src/main/c/toknroll_isolate.c -o target/pgo/toknroll_isolate.o   # Workers' threads
	cd target/pgo && rm -f default.iprof && NATIVE_IMAGE_OPTIONS="$(IMAGE_OPTIONS)" $(NATIVE_IMAGE) --pgo-instrument \
	  -H:NativeLinkerOption=$(abspath target/pgo/toknroll_isolate.o) \
	  -cp $(abspath target/classes):$(abspath target/test-classes):$$(cat classpath) com.qxotic.toknroll.libtoknroll.PgoTraining -o train
	cd target/pgo && ./train "$(CORPUS)" $(PGO_SOURCES)

target/tokenize: examples/c/tokenize.c include/toknroll.h $(LIB)
	cc -std=c99 -Wall -Wextra -Wpedantic -Iinclude -o $@ $< -L target -ltoknroll -Wl,-rpath,$(abspath target)

# The static archive: the example linked against it, no shared library in sight, and only toknroll_* exported.
target/tokenize-static: examples/c/tokenize.c include/toknroll.h $(STATIC)
	cc -std=c99 -Wall -Wextra -Wpedantic -DTOKNROLL_STATIC -Iinclude -o $@ $< $(STATIC) $(SYSLIBS)

# The C tests (tests/c): every function's results, error codes and special cases, each scenario in a
# process of its own; the header compiled as C++ too.
TEST_DEPS = tests/c/toknroll_test.c tests/c/test.h tests/c/fixtures.h include/toknroll.h
CFLAGS_TEST = -std=c99 -Wall -Wextra -Wpedantic -Werror -Iinclude
LINK_TEST = -L target -ltoknroll -Wl,-rpath,$(abspath target) -lpthread
target/toknroll_test: $(TEST_DEPS) $(LIB)
	cc $(CFLAGS_TEST) -o $@ $< $(LINK_TEST)
target/toknroll_test-static: $(TEST_DEPS) $(STATIC)
	cc $(CFLAGS_TEST) -DTOKNROLL_STATIC -o $@ $< $(STATIC) $(SYSLIBS)
target/header_test: tests/c/header_test.cpp include/toknroll.h $(LIB)
	c++ -std=c++11 -Wall -Wextra -Wpedantic -Werror -Iinclude -o $@ $< $(LINK_TEST)

# not linked against the library: it opens it, by path, to close it again (dlclose)
target/unload_test: tests/c/unload_test.c tests/c/fixtures.h
	cc $(CFLAGS_TEST) -o $@ $< $(if $(filter linux,$(OS)),-ldl)

check-c: target/toknroll_test target/header_test target/unload_test
	./target/toknroll_test
	./target/header_test
	./target/unload_test $(abspath $(LIB))

check-static: check-exports target/tokenize-static target/toknroll_test-static
	@# NF: the member header and blank lines nm prints for an archive. u: GNU unique, the C++ statics of
	@# Oracle's G1 (svm_gc::), which objcopy cannot make local; they clash only with a second G1 image linked in
	@nm --defined-only -g $(STATIC) | awk 'NF == 3 && $$2 != "u" {print $$3}' | sed 's/^_//' | sort > target/static.syms
	@diff target/header.syms target/static.syms && echo "the static archive exports the header's $$(wc -l < target/header.syms | tr -d " ") functions and nothing else"
	@! $(if $(filter darwin,$(OS)),otool -L,ldd) target/tokenize-static | grep -q toknroll && echo "the static example needs no libtoknroll"
	./target/tokenize-static $(SOURCE) "Hello, Tok'n'Roll!"
	./target/toknroll_test-static

check: check-version check-exports check-c target/tokenize check-static
	./target/tokenize $(SOURCE) "Hello, Tok'n'Roll!"
	cd python && python3 -m unittest discover -s tests -t tests

.PHONY: libtoknroll pgo-profile check check-c check-static
