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

# Force-inline the f16->f32 decode at every vector kernel site. Returning a FloatVector across a
# method boundary that the JIT declines to inline makes the vector ESCAPE and box on the heap
# (~15x slower). The rolling/decode attention uses Llama::f16DecodeVector; the block-tiled flash
# prefill (FlashAttention.qkTile/pvTile) reads the f16 KV cache through loadF16->f16ToF32Vector and
# needs the same guarantee, or it boxes and falls behind the rolling fallback.
INLINE_HINTS = -XX:CompileCommand=inline,com.qxotic.jinfer.Llama::f16DecodeVector \
	-XX:CompileCommand=inline,com.qxotic.jinfer.F16FloatTensor::f16ToF32Vector \
	-XX:CompileCommand=inline,com.qxotic.jinfer.FlashAttention::loadF16 \
	-XX:CompileCommand=inline,com.qxotic.jinfer.FlashAttention::loadF32
GEMM_INLINE_HINT = $(INLINE_HINTS)
JAVA_RUNTIME_OPTIONS = --enable-preview --add-modules jdk.incubator.vector,jdk.httpserver \
	--enable-native-access=ALL-UNNAMED $(INLINE_HINTS)

ifeq ($(OS),Windows_NT)
    EXE := .exe
else
    EXE :=
endif

NATIVE_FILE := jinfer$(EXE)
JAVA_MAIN_CLASS = com.qxotic.jinfer.LFM25
JAR_FILE = jinfer.jar

# External dependencies (fetched from Maven Central into libs/)
DEPS = libs/gguf-0.1.0.jar libs/toknroll-core-0.1.0.jar libs/toknroll-gguf-0.1.0.jar libs/json-0.1.0.jar
DEPS_CLASSPATH = libs/gguf-0.1.0.jar:libs/toknroll-core-0.1.0.jar:libs/toknroll-gguf-0.1.0.jar:libs/json-0.1.0.jar

# Multi-module classpath (Maven-compiled)
MM_CP = jinfer-core/target/classes:jinfer-jinja/target/classes:jinfer-models/target/classes:jinfer-server/target/classes

libs/%.jar:
	mkdir -p libs
	curl -fsSL -o $@ https://repo1.maven.org/maven2/com/qxotic/$(shell echo $* | sed 's/-[0-9.]*$$//')/$(shell echo $* | grep -oE '[0-9.]+$$')/$*.jar

# Build via Maven
compile:
	mvn -o compile

jar:
	mvn -o package -DskipTests
	@test -f jinfer-server/target/jinfer.jar && cp jinfer-server/target/jinfer.jar $(JAR_FILE) || true

test-compile:
	mvn -o test-compile

clean:
	mvn clean
	rm -f $(JAR_FILE) $(NATIVE_FILE)

# Run commands
run-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -cp $(MM_CP):$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS)

run-jar-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE)

# Tests (model-gated: set MODEL=path/to/model.gguf)
MODEL ?= ../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf

test: test-compile
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
		-cp $(MM_CP):jinfer-core/target/test-classes:$(DEPS_CLASSPATH) com.qxotic.jinfer.KernelParityTest
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) \
		-cp $(MM_CP):jinfer-core/target/test-classes:$(DEPS_CLASSPATH) com.qxotic.jinfer.GrammarTest
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) \
		-cp $(MM_CP):jinfer-jinja/target/test-classes:$(DEPS_CLASSPATH) com.qxotic.jinfer.jinja.JinjaRendererTest
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) --enable-native-access=ALL-UNNAMED \
		-cp $(MM_CP):jinfer-core/target/test-classes:$(DEPS_CLASSPATH) com.qxotic.jinfer.CacheStoreTest
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) \
		-cp $(MM_CP):jinfer-models/target/test-classes:$(DEPS_CLASSPATH) com.qxotic.jinfer.TokenizerParityTest $(MODEL)

test-server: test-compile
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
		-cp $(MM_CP):jinfer-server/target/test-classes:$(DEPS_CLASSPATH) com.qxotic.jinfer.ServerIntegrationTest $(MODEL)
	$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
		-Djinfer.serverMaxTokens=24 -Djinfer.serverRequestTimeout=0 \
		-cp $(MM_CP):jinfer-server/target/test-classes:$(DEPS_CLASSPATH) com.qxotic.jinfer.ServerIntegrationTest $(MODEL) ceiling

test-golden: jar
	@test -f $(MODEL) || { echo "test-golden: model not found ($(MODEL)), skipping"; exit 0; }
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 -jar $(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --max-tokens 64 2>/dev/null > /tmp/jinfer-golden-1.txt
	@$(JAVA) $(JAVA_RUNTIME_OPTIONS) -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 -jar $(JAR_FILE) \
		--model $(MODEL) --prompt "List the planets." --max-tokens 64 2>/dev/null > /tmp/jinfer-golden-2.txt
	@diff /tmp/jinfer-golden-1.txt /tmp/jinfer-golden-2.txt && echo "test-golden: deterministic"

# GraalVM Native Image
check-native-image:
	@v=$$($(NATIVE_IMAGE) --version 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1); \
	case "$$v" in \
	25.0.0|25.0.1|25.0.2) \
		echo "ERROR: native-image $$v is too old for the jinfer kernels (need >= 25.0.3)."; \
		echo "       e.g. JAVA_HOME=$$HOME/Downloads/graalvm-25.1.0-dev+9.1 make $(MAKECMDGOALS)"; \
		exit 1;; \
	esac

NATIVE_IMAGE_FLAGS = -H:+UnlockExperimentalVMOptions -H:+VectorAPISupport -H:+ForeignAPISupport \
	-march=native --enable-preview --add-modules jdk.incubator.vector,jdk.httpserver \
	--enable-native-access=ALL-UNNAMED -J--enable-native-access=ALL-UNNAMED \
	--initialize-at-build-time='com.qxotic.' \
	--initialize-at-run-time=com.qxotic.jinfer.RuntimeFlags,com.qxotic.jinfer.Parallel \
	-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 \
	-Djinfer.PreloadGGUF=$(PRELOAD_GGUF)

$(NATIVE_FILE): check-native-image jar
	$(NATIVE_IMAGE) -O3 $(NATIVE_IMAGE_FLAGS) -cp $(JAR_FILE):$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS) -o $(NATIVE_FILE)

default: jar
native: $(NATIVE_FILE)

native-pgo-instrument: check-native-image jar
	$(NATIVE_IMAGE) --pgo-instrument $(NATIVE_IMAGE_FLAGS) -cp $(JAR_FILE):$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS) -o jinfer.pgo

native-pgo: check-native-image jar default.iprof
	$(NATIVE_IMAGE) --pgo=default.iprof $(NATIVE_IMAGE_FLAGS) -cp $(JAR_FILE):$(DEPS_CLASSPATH) $(JAVA_MAIN_CLASS) -o $(NATIVE_FILE)

# Native GEMM library
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Darwin)
    JAVA_HOME_DETECTED := $(shell /usr/libexec/java_home 2>/dev/null || dirname $$(dirname $$(readlink $$(which $(JAVA)))))
    JNI_PLATFORM_INCLUDE := darwin
    NATIVE_LIB := libjinferjni.dylib
    NATIVE_SHARED_FLAGS := -dynamiclib
else
    JAVA_HOME_DETECTED := $(shell dirname $$(dirname $$(readlink -f $$(which $(JAVA)))))
    JNI_PLATFORM_INCLUDE := linux
    NATIVE_LIB := libjinferjni.so
    NATIVE_SHARED_FLAGS := -shared
endif

$(NATIVE_LIB): jinferjni.c
	gcc -O3 -march=native $(NATIVE_SHARED_FLAGS) -fPIC -pthread \
		-I$(JAVA_HOME_DETECTED)/include -I$(JAVA_HOME_DETECTED)/include/$(JNI_PLATFORM_INCLUDE) \
		jinferjni.c -o $(NATIVE_LIB)

libnative: $(NATIVE_LIB)

libjinferjni.a: jinferjni.c
	gcc -O3 -march=native -c -fPIC -pthread \
		-I$(JAVA_HOME_DETECTED)/include -I$(JAVA_HOME_DETECTED)/include/linux \
		jinferjni.c -o jinferjni.o
	ar rcs libjinferjni.a jinferjni.o

target/buildtools/JinferStaticGemmFeature.class: buildtools/JinferStaticGemmFeature.java
	$(JAVAC) --add-modules org.graalvm.nativeimage \
		-cp $(JAVA_HOME_DETECTED)/lib/svm/builder/svm.jar \
		-d target/buildtools buildtools/JinferStaticGemmFeature.java

native-static-gemm: check-native-image jar libjinferjni.a target/buildtools/JinferStaticGemmFeature.class
	$(NATIVE_IMAGE) -O3 $(NATIVE_IMAGE_FLAGS) \
		--features=JinferStaticGemmFeature \
		-H:CLibraryPath=$(CURDIR) \
		-J--add-exports=org.graalvm.nativeimage.builder/com.oracle.svm.hosted=ALL-UNNAMED \
		-J--add-exports=org.graalvm.nativeimage.builder/com.oracle.svm.hosted.c=ALL-UNNAMED \
		-J--add-exports=org.graalvm.nativeimage.builder/com.oracle.svm.core.jdk=ALL-UNNAMED \
		-Djinfer.staticGemm=true \
		-cp $(JAR_FILE):$(DEPS_CLASSPATH):target/buildtools \
		$(JAVA_MAIN_CLASS) \
		-o $(NATIVE_FILE)

ARM64_CROSS_CC  ?= aarch64-linux-gnu-gcc
arm64-so: jinferjni.c
	$(ARM64_CROSS_CC) -O3 -shared -fPIC -pthread \
		-I$(JAVA_HOME_DETECTED)/include -I$(JAVA_HOME_DETECTED)/include/linux \
		jinferjni.c -o libjinferjni-arm64.so
	@echo "ARM64 shared library built (cannot be loaded on x86)."

gemm.metallib: gemm.metal
	xcrun -sdk macosx metal -c gemm.metal -o gemm.air
	xcrun -sdk macosx metallib gemm.air -o gemm.metallib
	rm -f gemm.air

metal-lib: gemm.metallib
	@echo "Metal library built."

.PHONY: check-native-image compile clean jar test test-compile test-server test-golden native native-pgo native-pgo-instrument native-static-gemm libnative metal-lib arm64-so run-command run-jar-command
.SUFFIXES: .java .class .jar
