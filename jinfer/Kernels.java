// Matmul backend seam: implementation chosen once at startup (Java Vector API or native C).
package com.llama4j;

import com.qxotic.format.gguf.GGMLType;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.lang.reflect.Field;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Backend seam for the Q8_0 matmul hot path. The implementation is chosen ONCE, when this
 * interface initializes (build time in a native image): capability probing, native-library
 * loading and tile-shape policy all live in the implementations, so call sites carry no
 * per-call capability checks and devirtualize to the single bound instance. A method returns
 * false to decline a shape; the caller then falls back to the generic FloatTensor path.
 *
 * <p>The backends hold POLICY only; vector kernel bodies stay static methods on the quant
 * class they decode (Q8_0FloatTensor.vectorGemm512F32 etc.), like the per-quant dot kernels.
 * Native kernels for further quant types should grow this interface rather than adding
 * dispatch flags to the tensors.
 */
interface Kernels {

    Kernels INSTANCE = NativeKernels.tryLoad() ? new NativeKernels() : new JavaKernels();

    boolean gemmQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                      F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1);

    boolean gemvQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                      F32FloatTensor out, int outOffset, int dim0, int dim1);
}

/** The Java Vector API kernels: 512-bit register-tiled GEMM/GEMV with a generic vector fallback. */
final class JavaKernels implements Kernels {

    // The Graal JIT only allocates zmm0-zmm15, so a 4x4 register tile (16 accumulators + 8 weight
    // vectors) spills there; C2 uses zmm16-zmm31 and runs 4x4 ~30% faster than 3x2. Native-image
    // AOT has all 32 zmm but its linear-scan allocator still spills the 16 vector loop-phis
    // (~19 spills/iteration in every tile shape tried), so Graal AOT also defaults to 3x2.
    static final boolean GRAAL_COMPILER = System.getProperty("org.graalvm.nativeimage.imagecode") != null
            || System.getProperty("java.vm.version", "").contains("jvmci");
    static final boolean GEMM_TILE_4X4 =
            "4x4".equals(System.getProperty("llama.Q8_0GemmTile", GRAAL_COMPILER ? "3x2" : "4x4"));

    @Override
    public boolean gemmQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                             F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if (!FloatTensor.USE_VECTOR_API) {
            return false;
        }
        if (FloatTensor.F_SPECIES.vectorBitSize() == 512
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            Q8_0FloatTensor.vectorGemm512F32(w, x, out, thatStride, outStride, sequenceLength, dim0, dim1, thisOffset);
            return true;
        }
        Q8_0FloatTensor.vectorGemm(w, x, out, thatStride, outStride, sequenceLength, dim0, dim1, thisOffset);
        return true;
    }

    @Override
    public boolean gemvQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                             F32FloatTensor out, int outOffset, int dim0, int dim1) {
        // Small gemvs (and narrow vectors) decline: parallel per-row dots win there.
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES.vectorBitSize() == 512
                && (long) dim0 * dim1 > (1 << 18)
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            Q8_0FloatTensor.vectorGemv512(w, x, thatOffset, out, outOffset, dim0, dim1, thisOffset);
            return true;
        }
        return false;
    }

}

/**
 * The AVX-512 C kernels (lfm25jni.c): statically linked into a native image via
 * LFM25StaticGemmFeature (-Dllama.staticGemm=true at image build time), or loaded into the JVM
 * from -Dllama.nativeGemmLib=/path/to/liblfm25jni.so. The exported symbols are JNI-mangled
 * against THIS class — renaming it or the native methods requires renaming the C exports.
 */
final class NativeKernels implements Kernels {

    // Native gemv loses ~6% to the Java path on decode (dispatch overhead on ~360 small calls
    // per token vs the always-warm ForkJoin pool); opt in with -Dllama.nativeGemv=true.
    private static final boolean NATIVE_GEMV = Boolean.getBoolean("llama.nativeGemv");

    private final JavaKernels java = new JavaKernels();

    static boolean tryLoad() {
        if (Boolean.getBoolean("llama.staticGemm")) {
            // liblfm25jni.a is statically linked into the image (LFM25StaticGemmFeature); the
            // native methods bind at image link time, no library loading required.
            return true;
        }
        String lib = System.getProperty("llama.nativeGemmLib");
        if (lib == null || lib.isBlank()) {
            return false;
        }
        try {
            System.load(Path.of(lib).toAbsolutePath().toString());
            return true;
        } catch (Throwable t) {
            System.err.println("Native GEMM library unavailable (" + t + "); using the Java kernels.");
            return false;
        }
    }

    @Override
    public boolean gemmQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatStride,
                             F32FloatTensor out, int outStride, int sequenceLength, int dim0, int dim1) {
        if ((dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            nativeGemmQ8F32Ptr(w.memorySegment.address(), x.memorySegment.address(), out.memorySegment.address(),
                    thatStride, outStride, sequenceLength, dim0, dim1, thisOffset,
                    RuntimeFlags.GEMM_ROW_TILE, RuntimeFlags.GEMM_SEQ_TILE, RuntimeFlags.GEMM_THREADS);
            return true;
        }
        return java.gemmQ8F32(w, thisOffset, x, thatStride, out, outStride, sequenceLength, dim0, dim1);
    }

    @Override
    public boolean gemvQ8F32(Q8_0FloatTensor w, int thisOffset, F32FloatTensor x, int thatOffset,
                             F32FloatTensor out, int outOffset, int dim0, int dim1) {
        if (NATIVE_GEMV
                && (dim1 & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0
                && (thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1)) == 0) {
            nativeGemvQ8F32Ptr(w.memorySegment.address(),
                    x.memorySegment.address() + 4L * thatOffset,
                    out.memorySegment.address() + 4L * outOffset,
                    dim0, dim1, thisOffset);
            return true;
        }
        return java.gemvQ8F32(w, thisOffset, x, thatOffset, out, outOffset, dim0, dim1);
    }

    private static native void nativeGemmQ8F32Ptr(long weightAddress, long rhsAddress, long outAddress,
                                                  int thatStride, int outStride, int sequenceLength,
                                                  int dim0, int dim1, int thisOffset,
                                                  int rowTile, int seqTile, int workers);

    private static native void nativeGemvQ8F32Ptr(long weightAddress, long rhsAddress, long outAddress,
                                                  int dim0, int dim1, int thisOffset);
}
