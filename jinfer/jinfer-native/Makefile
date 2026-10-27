# libjinfer: jinfer as a shared library with the C API in include/jinfer.h. `make help` lists the targets.
NAME  := jinfer
OPT   := $(if $(QUICK),-Ob,-O3)
# The ISA level the image is built for; the kernels' vector width follows it (VectorWidth, at
# build time). x86-64-v3 (AVX2, 2013) and armv8.1-a are GraalVM's own defaults, spelled out because
# the plugin cannot leave the flag out; x86-64-v3 is the floor there, below it the kernels are not usable.
MARCH ?= $(if $(QUICK),native,$(if $(filter x86-64,$(ARCH)),x86-64-v3,armv8.1-a))
TOKNROLL := ../../toknroll/toknroll-native
HEADERS := include/jinfer.h $(TOKNROLL)/include/toknroll.h   # the library carries toknroll's C face too
SYMS := jinfer|toknroll
WHEEL_DEPS = $(TOKNROLL)/target/wheel/*.whl
WHEEL_TEST = JINFER_INSTALLED=1 wheelenv/bin/python ../python/test_jinfer.py

help::
	@echo 'make libjinfer             -O3 -march=$(MARCH) (GraalVM'"'"'s default), what ships -> $(LIB); MARCH=x86-64-v4 for AVX-512'
	@echo 'make libjinfer QUICK=1     this machine, quick build (-Ob -march=native): runs, kernels unoptimized, never measure it'
	@echo 'make check                 headers vs exports, C example, Python test'
	@echo 'make qa                    the corner cases: threads, exit, soak, offline, tools'
include ../../native.mk

libjinfer:
	@$(MAKE) -s -C $(ROOT) check-native-image
	cd $(ROOT) && $(MAVEN) $(MAVEN_FLAGS) -Pnative -pl jinfer/jinfer-native -am package -DskipTests \
	  -Djinfer.image.opt=$(OPT) -Djinfer.image.march=$(MARCH)

target/chat: examples/c/chat.c $(HEADERS) $(LIB)
	cc -std=c99 -Wall -Wextra -Iinclude -I$(TOKNROLL)/include -o $@ $< -L target -ljinfer -Wl,-rpath,$(abspath target)

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
