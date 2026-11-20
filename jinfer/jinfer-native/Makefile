# libjinfer: jinfer as a shared library with the C API in include/jinfer.h. `make help` lists the targets.
NAME  := jinfer
# The optimization: the pom's (-O3) unless QUICK=1 or a PGO profile asks for another. Oracle GraalVM builds what
# ships from the profile the instrumented CLI records (jinfer-cli's pgo-profile), the CLI's image's own; GraalVM CE
# builds without one.
CLI   := ../jinfer-cli
PGO    = $(if $(filter 1,$(QUICK)),,$(shell $(NATIVE_IMAGE) --version 2>/dev/null | grep -q 'Oracle GraalVM' && echo yes))
OPT    = $(if $(filter 1,$(QUICK)),-Ob,$(if $(PGO),--pgo=$$(cat $(abspath $(CLI))/target/pgo/profiles)))
# The ISA level is native-image's own default, x86-64-v3 (AVX2, 2013) on x86-64, the kernels' floor; their
# vector width follows it (VectorWidth, at build time). NATIVE_IMAGE_OPTIONS=-march=<target> asks for another.
TOKNROLL := ../../toknroll/toknroll-native
# The library carries toknroll's C face too, and libjam's, linked in whole: jam's Java side finds jam.h's
# functions and its JNI entry points by name in the image. Exported, not offered: the release carries no jam.h.
HEADERS := include/jinfer.h $(TOKNROLL)/include/toknroll.h
EXPORT_HEADERS := $(HEADERS) ../../jam/jam-native/include/jam.h
SYMS := jinfer|toknroll|jam
EXTRA_EXPORTS := JNI_OnLoad_jam Java_com_qxotic_jam_libjam_NativeJAM_newJni Java_com_qxotic_jam_libjam_NativeJAM_mmJni Java_com_qxotic_jam_libjam_NativeJAM_freeJni
WHEEL_DEPS = $(TOKNROLL)/target/wheel/*.whl
WHEEL_TEST = wheelenv/bin/python ../python/test_jinfer.py
DESCRIPTION = jinfer, local LLM inference with a C API
WHEEL_EXAMPLE = examples/c/chat.c

help::
	@echo 'make libjinfer             native-image'"'"'s default target, what ships -> $(LIB); NATIVE_IMAGE_OPTIONS=-march=x86-64-v4 for AVX-512'
	@echo '                           Oracle GraalVM: PGO, from the CLI'"'"'s profile (make -C $(CLI) pgo-profile); CE: -O3'
	@echo 'make libjinfer QUICK=1     this machine, quick build (-Ob -march=native): runs, kernels unoptimized, never measure it'
	@echo 'make check                 headers vs exports, C tests and example, Python test'
	@echo 'make qa                    the corner cases: threads, exit, soak, offline, tools'
include ../../scripts/native.mk

libjinfer:
	@$(MAKE) -s -C $(ROOT) check-native-image
	$(if $(PGO),$(MAKE) -C $(CLI) target/pgo/profiles)
	cd $(ROOT) && NATIVE_IMAGE_OPTIONS="$(IMAGE_OPTIONS)" \
	  NATIVE_SONAME=$(SONAME) $(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/jinfer-native -am package -DskipTests \
	  $(if $(OPT),-Djinfer.image.opt=$(OPT)) -Dtoknroll.image.skip=true
	ln -sf $(notdir $(LIB)) target/$(SONAME)

target/chat: examples/c/chat.c $(HEADERS) $(LIB)
	$(CC) -std=c99 -Wall -Wextra -Wpedantic -Iinclude -I$(TOKNROLL)/include -o $@ $< -L target -ljinfer -Wl,-rpath,$(abspath target)

# The C tests (tests/c): jinfer.h's functions on the cached models, and toknroll's tests again, against
# the toknroll functions libjinfer carries (TEST_LIBJINFER); both headers compiled as C++ too.
CFLAGS_TEST = -std=c99 -Wall -Wextra -Wpedantic -Werror -Iinclude -I$(TOKNROLL)/include -I$(TOKNROLL)/tests/c
LINK_TEST = -L target -ljinfer -Wl,-rpath,$(abspath target) -lpthread $(if $(filter linux,$(OS)),-lutil)   # forkpty
target/jinfer_test: tests/c/jinfer_test.c $(TOKNROLL)/tests/c/test.h $(HEADERS) $(LIB)
	$(CC) $(CFLAGS_TEST) -o $@ $< $(LINK_TEST)
target/toknroll_test: $(TOKNROLL)/tests/c/toknroll_test.c $(TOKNROLL)/tests/c/test.h $(TOKNROLL)/tests/c/fixtures.h $(TOKNROLL)/tests/c/progress.h $(HEADERS) $(LIB)
	$(CC) $(CFLAGS_TEST) -DTEST_LIBJINFER -o $@ $< $(LINK_TEST)
target/header_test: tests/c/header_test.cpp $(HEADERS) $(LIB)
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Werror -Iinclude -I$(TOKNROLL)/include -o $@ $< $(LINK_TEST)

check-c: target/jinfer_test target/toknroll_test target/header_test
	@if command -v clang >/dev/null; then clang $(CFLAGS_TEST) -fsyntax-only tests/c/jinfer_test.c && clang $(CFLAGS_TEST) -DTEST_LIBJINFER -fsyntax-only $(TOKNROLL)/tests/c/toknroll_test.c; fi   # macOS's compiler warns where gcc does not
	./target/jinfer_test
	./target/toknroll_test
	./target/header_test

check: check-version check-pin check-exports check-c target/chat
	./target/chat
	python3 python/test_jinfer.py

# The wheel pins the toknroll it embeds exactly: toknroll's Python layer is bound to this library, so a
# newer toknroll wheel calling a function this libjinfer lacks must be refused by pip, not found at import.
check-pin:
	@p=$$(grep -oE 'toknroll==[0-9a-z.]+' python/pyproject.toml | cut -d= -f3); \
	 t=$$(sed -n 's/^#define TOKNROLL_VERSION "\(.*\)"/\1/p' $(TOKNROLL)/include/toknroll.h | tr -d -); \
	 [ "$$p" = "$$t" ] && echo "the wheel pins toknroll==$$t, the one inside" || { echo "pin drift: the wheel says toknroll==$$p, libjinfer carries $$t"; exit 1; }

qa:
	python3 python/qa.py

# The installed wheel needs toknroll's, built from this tree
check-wheel: toknroll-wheel
toknroll-wheel:
	$(MAKE) -s -C $(TOKNROLL) wheel

.PHONY: libjinfer check check-c check-pin qa toknroll-wheel
