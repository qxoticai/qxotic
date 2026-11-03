# libjinfer: jinfer as a shared library with the C API in include/jinfer.h. `make help` lists the targets.
NAME  := jinfer
OPT   := $(if $(QUICK),-Ob,-O3)
# The ISA level is native-image's own default, x86-64-v3 (AVX2, 2013) on x86-64, the kernels' floor; their
# vector width follows it (VectorWidth, at build time). MARCH=<target> asks for another, QUICK=1 for this machine.
MARCH ?= $(if $(QUICK),native)
TOKNROLL := ../../toknroll/toknroll-native
HEADERS := include/jinfer.h $(TOKNROLL)/include/toknroll.h   # the library carries toknroll's C face too
# and libjam's, linked in whole: jam's Java side finds jam.h's functions and its JNI entry points by name
# in the image. Exported, not offered: the release carries no jam.h.
EXPORT_HEADERS := $(HEADERS) ../../jam/jam-native/include/jam.h
SYMS := jinfer|toknroll|jam
EXTRA_EXPORTS := JNI_OnLoad_jam Java_com_qxotic_jam_libjam_NativeJAM_newJni Java_com_qxotic_jam_libjam_NativeJAM_mmJni Java_com_qxotic_jam_libjam_NativeJAM_freeJni
WHEEL_DEPS = $(TOKNROLL)/target/wheel/*.whl
WHEEL_TEST = JINFER_INSTALLED=1 wheelenv/bin/python ../python/test_jinfer.py

help::
	@echo 'make libjinfer             -O3, native-image'"'"'s default target, what ships -> $(LIB); MARCH=x86-64-v4 for AVX-512'
	@echo 'make libjinfer QUICK=1     this machine, quick build (-Ob -march=native): runs, kernels unoptimized, never measure it'
	@echo 'make check                 headers vs exports, C example, Python test'
	@echo 'make qa                    the corner cases: threads, exit, soak, offline, tools'
include ../../native.mk

libjinfer:
	@$(MAKE) -s -C $(ROOT) check-native-image
	cd $(ROOT) && NATIVE_IMAGE_OPTIONS="$(strip $(NATIVE_IMAGE_OPTIONS) $(if $(MARCH),-march=$(MARCH)))" \
	  NATIVE_SONAME=$(SONAME) $(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/jinfer-native -am package -DskipTests \
	  -Djinfer.image.opt=$(OPT) -Dtoknroll.image.skip=true
	ln -sf $(notdir $(LIB)) target/$(SONAME)

target/chat: examples/c/chat.c $(HEADERS) $(LIB)
	cc -std=c99 -Wall -Wextra -Wpedantic -Iinclude -I$(TOKNROLL)/include -o $@ $< -L target -ljinfer -Wl,-rpath,$(abspath target)

check: check-version check-pin check-exports target/chat
	./target/chat
	python3 python/test_jinfer.py

# The wheel pins the toknroll it embeds exactly: toknroll's Python layer is bound to this library, so a
# newer toknroll wheel calling a function this libjinfer lacks must be refused by pip, not found at import.
check-pin:
	@p=$$(grep -oE 'toknroll==[0-9.]+' python/pyproject.toml | cut -d= -f3); \
	 t=$$(grep -oE 'TOKNROLL_VERSION_(MAJOR|MINOR|PATCH) [0-9]+' $(TOKNROLL)/include/toknroll.h | awk '{printf "%s%s", (NR>1?".":""), $$2}'); \
	 [ "$$p" = "$$t" ] && echo "the wheel pins toknroll==$$t, the one inside" || { echo "pin drift: the wheel says toknroll==$$p, libjinfer carries $$t"; exit 1; }

qa:
	python3 python/qa.py

check-wheel: toknroll-wheel   # the installed wheel needs toknroll's, built from this tree
toknroll-wheel:
	$(MAKE) -s -C $(TOKNROLL) wheel

.PHONY: libjinfer check check-pin qa toknroll-wheel
