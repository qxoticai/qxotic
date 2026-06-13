ifdef JAVA_HOME
	JAVAC ?= ${JAVA_HOME}/bin/javac
	JAVA ?= ${JAVA_HOME}/bin/java
	JAR ?= ${JAVA_HOME}/bin/jar
	NATIVE_IMAGE ?= ${JAVA_HOME}/bin/native-image
endif

JAVAC ?= javac
JAVA ?= java
JAR ?= jar
NATIVE_IMAGE ?= native-image

JAVA_MAJOR_VERSION := $(shell $(JAVA) -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)

JAVA_COMPILE_OPTIONS = --enable-preview -source $(JAVA_MAJOR_VERSION) -g --add-modules jdk.incubator.vector,jdk.httpserver
JAVA_RUNTIME_OPTIONS = --enable-preview --add-modules jdk.incubator.vector,jdk.httpserver --enable-native-access=ALL-UNNAMED

ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif

# Define the executable name
NATIVE_FILE := lfm25$(EXE)

JAVA_MAIN_CLASS = com.llama4j.LFM25
JAR_FILE = lfm25.jar

JAVA_SOURCES = $(wildcard *.java)
JAVA_CLASSES = target/classes/.compiled

# External dependencies (fetched from Maven Central into libs/; also declared as jbang //DEPS)
DEPS = libs/gguf-0.1.0.jar libs/toknroll-core-0.1.0.jar libs/toknroll-gguf-0.1.0.jar libs/json-0.1.0.jar
DEPS_CLASSPATH = libs/gguf-0.1.0.jar:libs/toknroll-core-0.1.0.jar:libs/toknroll-gguf-0.1.0.jar:libs/json-0.1.0.jar

libs/%.jar:
	mkdir -p libs
	curl -fsSL -o $@ https://repo1.maven.org/maven2/com/qxotic/$(shell echo $* | sed 's/-[0-9.]*$$//')/$(shell echo $* | grep -oE '[0-9.]+$$')/$*.jar

# Bundle all classes in a jar (Class-Path points at the deps next to the jar)
$(JAR_FILE): $(JAVA_CLASSES) LICENSE
	printf 'Main-Class: $(JAVA_MAIN_CLASS)\nClass-Path: $(DEPS)\n' > target/MANIFEST.MF
	$(JAR) -cvfm $(JAR_FILE) target/MANIFEST.MF LICENSE -C target/classes .

jar: $(JAR_FILE)

# Compile the Java source files
compile: $(JAVA_CLASSES)

# Prints the command to run the Java main class
run-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -cp target/classes:$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS)

# Prints the command to run the $(JAR_FILE)
run-jar-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE)

# Clean the target directory
clean:
	rm -rf ./target
	rm $(JAR_FILE) $(NATIVE_FILE)

# Compile the Java source files (single pass: the classes reference each other)
target/classes/.compiled: $(JAVA_SOURCES) $(DEPS) | target/classes
	$(JAVAC) $(JAVA_COMPILE_OPTIONS) -cp $(DEPS_CLASSPATH) -d target/classes $(JAVA_SOURCES)
	@touch $@

# Create the target directory
target/classes:
	mkdir -p target/classes

# Kernel parity tests (synthetic tensors, no model file needed) + tokenizer parity (model-gated)
target/test-classes/com/llama4j/KernelParityTest.class: tests/*.java $(JAVA_CLASSES)
	$(JAVAC) $(JAVA_COMPILE_OPTIONS) -cp target/classes:$(DEPS_CLASSPATH) -d target/test-classes tests/*.java

test: target/test-classes/com/llama4j/KernelParityTest.class
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
		-cp target/classes:target/test-classes:$(DEPS_CLASSPATH) com.llama4j.KernelParityTest
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -cp target/classes:target/test-classes:$(DEPS_CLASSPATH) com.llama4j.TokenizerParityTest $(MODEL)

# End-to-end server test (in-process, ephemeral port); model-gated like the others.
test-server: target/test-classes/com/llama4j/KernelParityTest.class
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
		-cp target/classes:target/test-classes:$(DEPS_CLASSPATH) com.llama4j.ServerIntegrationTest $(MODEL)

# Greedy-determinism check: same runtime, same input => byte-identical output. Skips without a
# model. (Cross-runtime bit-identity is NOT an invariant: reduceLanes order differs, see FIXES.md.)
MODEL ?= ../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf
test-golden: jar
	@test -f $(MODEL) || { echo "test-golden: model not found ($(MODEL)), skipping"; exit 0; }
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 -jar $(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --max-tokens 64 2>/dev/null > target/golden-1.txt
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 -jar $(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --max-tokens 64 2>/dev/null > target/golden-2.txt
	@diff target/golden-1.txt target/golden-2.txt && echo "test-golden: deterministic"

# GraalVM >= 25.0.3 required for native images: 25.0.2 has slow MemorySegment scalar access
# and Vector API miscompiles (see FIXES.md). Set JAVA_HOME to the patched dev build.
check-native-image:
	@v=$$($(NATIVE_IMAGE) --version 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
	case "$$v" in \
	25.0.0|25.0.1|25.0.2) \
		echo "ERROR: native-image $$v is too old for the LFM25 kernels (need >= 25.0.3)."; \
		echo "       e.g. JAVA_HOME=$$HOME/Downloads/graalvm-25.1.0-dev+9.1 make $(MAKECMDGOALS)"; \
		exit 1;; \
	esac

# Flags shared by every native-image target (-O3 added where PGO does not forbid it).
NATIVE_IMAGE_FLAGS = -H:+UnlockExperimentalVMOptions -H:+VectorAPISupport -H:+ForeignAPISupport \
	-march=native --enable-preview --add-modules jdk.incubator.vector,jdk.httpserver \
	--enable-native-access=ALL-UNNAMED -J--enable-native-access=ALL-UNNAMED \
	--initialize-at-build-time='com.llama4j.AOT,com.llama4j.FloatTensor,com.llama4j.,com.qxotic.' \
	--initialize-at-run-time=com.llama4j.RuntimeFlags,com.llama4j.ThreadAffinity \
	-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
	-Dllama.PreloadGGUF=$(PRELOAD_GGUF)

$(NATIVE_FILE): check-native-image jar
	$(NATIVE_IMAGE) -O3 $(NATIVE_IMAGE_FLAGS) -cp $(JAR_FILE):$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS) -o $(NATIVE_FILE)

compile: target/classes
default: jar
native: $(NATIVE_FILE)

# PGO build (243 -> ~262 tok/s prefill): instrument, run a representative workload to produce
# default.iprof, then rebuild with the profile (instrumented runs are unoptimized => slow).
#   make native-pgo-instrument
#   ./lfm25.pgo --model <model.gguf> --prompt "$(cat prompt.txt)" --max-tokens 600
#   make native-pgo
# 3x2 everywhere on Graal: 4x4 only wins (+4%, 272) when PGO data rescues its register allocation
# and collapses without it (202) — not worth the coupling. -Dllama.Q8_0GemmTile=4x4 to experiment.
native-pgo-instrument: check-native-image jar
	$(NATIVE_IMAGE) --pgo-instrument $(NATIVE_IMAGE_FLAGS) -cp $(JAR_FILE):$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS) -o lfm25.pgo

native-pgo: check-native-image jar default.iprof
	$(NATIVE_IMAGE) --pgo=default.iprof $(NATIVE_IMAGE_FLAGS) -cp $(JAR_FILE):$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS) -o $(NATIVE_FILE)

# Native GEMM library (opt-in via -Dllama.nativeGemmLib=$(PWD)/$(NATIVE_LIB))
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Darwin)
    JAVA_HOME_DETECTED := $(shell /usr/libexec/java_home 2>/dev/null || dirname $$(dirname $$(readlink $$(which $(JAVA)))))
    JNI_PLATFORM_INCLUDE := darwin
    NATIVE_LIB := liblfm25jni.dylib
    NATIVE_SHARED_FLAGS := -dynamiclib
else
    JAVA_HOME_DETECTED := $(shell dirname $$(dirname $$(readlink -f $$(which $(JAVA)))))
    JNI_PLATFORM_INCLUDE := linux
    NATIVE_LIB := liblfm25jni.so
    NATIVE_SHARED_FLAGS := -shared
endif

$(NATIVE_LIB): lfm25jni.c
	gcc -O3 -march=native $(NATIVE_SHARED_FLAGS) -fPIC -pthread \
		-I$(JAVA_HOME_DETECTED)/include -I$(JAVA_HOME_DETECTED)/include/$(JNI_PLATFORM_INCLUDE) \
		lfm25jni.c -o $(NATIVE_LIB)

libnative: $(NATIVE_LIB)

# Native image with the AVX-512 GEMM statically linked (single self-contained binary).
# Uses unexported SVM internals via buildtools/LFM25StaticGemmFeature (oracle/graal#3359).
liblfm25jni.a: lfm25jni.c
	gcc -O3 -march=native -c -fPIC -pthread \
		-I$(JAVA_HOME_DETECTED)/include -I$(JAVA_HOME_DETECTED)/include/linux \
		lfm25jni.c -o lfm25jni.o
	ar rcs liblfm25jni.a lfm25jni.o

target/buildtools/LFM25StaticGemmFeature.class: buildtools/LFM25StaticGemmFeature.java
	$(JAVAC) --add-modules org.graalvm.nativeimage \
		-cp $(JAVA_HOME_DETECTED)/lib/svm/builder/svm.jar \
		-d target/buildtools buildtools/LFM25StaticGemmFeature.java

native-static-gemm: check-native-image jar liblfm25jni.a target/buildtools/LFM25StaticGemmFeature.class
	$(NATIVE_IMAGE) -O3 $(NATIVE_IMAGE_FLAGS) \
		--features=LFM25StaticGemmFeature \
		-H:CLibraryPath=$(CURDIR) \
		-J--add-exports=org.graalvm.nativeimage.builder/com.oracle.svm.hosted=ALL-UNNAMED \
		-J--add-exports=org.graalvm.nativeimage.builder/com.oracle.svm.hosted.c=ALL-UNNAMED \
		-J--add-exports=org.graalvm.nativeimage.builder/com.oracle.svm.core.jdk=ALL-UNNAMED \
		-Dllama.staticGemm=true \
		-cp $(JAR_FILE):$(DEPS_CLASSPATH):target/buildtools \
		$(JAVA_MAIN_CLASS) \
		-o $(NATIVE_FILE)

# ARM64 cross-compile target (binary cannot run on x86, but validates ARM NEON code compiles).
# Requires: apt install gcc-aarch64-linux-gnu
ARM64_CROSS_CC  ?= aarch64-linux-gnu-gcc
arm64-so: lfm25jni.c
	$(ARM64_CROSS_CC) -O3 -shared -fPIC -pthread \
		-I$(JAVA_HOME_DETECTED)/include -I$(JAVA_HOME_DETECTED)/include/linux \
		lfm25jni.c -o liblfm25jni-arm64.so
	@echo "ARM64 shared library built (cannot be loaded on x86)."

# Metal compute library (macOS / Apple Silicon only).
# Requires: Xcode Command Line Tools (xcrun).
gemm.metallib: gemm.metal
	xcrun -sdk macosx metal -c gemm.metal -o gemm.air
	xcrun -sdk macosx metallib gemm.air -o gemm.metallib
	rm -f gemm.air

metal-lib: gemm.metallib
	@echo "Metal library built."

.PHONY: check-native-image compile clean jar test test-server test-golden native native-pgo native-pgo-instrument native-static-gemm libnative metal-lib arm64-so run-command run-jar-command
.SUFFIXES: .java .class .jar
