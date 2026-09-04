package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.Segments;
import com.qxotic.jota.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/**
 * The Java floor's k-quant decode dots (n == 1) against the per-element decoders, summed in double.
 * Runs at whatever Vector API width this JVM has; the pom's vector-128, vector-256 and scalar
 * executions cover the other lane counts, so every slice/part mapping of the dots is exercised.
 */
class KQuantFloorDotTest {

    @Test
    void q4k() {
        check(GGMLType.Q4_K);
    }

    @Test
    void q5k() {
        check(GGMLType.Q5_K);
    }

    @Test
    void q6k() {
        check(GGMLType.Q6_K);
    }

    private static void check(GGMLType type) {
        DataType dt = GGMLDataTypes.toDataType(type);
        String width =
                Segments.F_SPECIES == null ? "scalar" : Segments.F_SPECIES.vectorBitSize() + "-bit";
        try (Arena arena = Arena.ofConfined()) {
            for (int k : new int[] {256, 768}) {
                int m = 16;
                long elems = (long) m * k;
                MemorySegment w = Oracles.blockQuant(arena, type, elems, type.ordinal() + 11L);
                MemorySegment x = Oracles.f32(arena, k, 5);
                MemorySegment out = arena.allocate(4L * m, 64);
                MatMul.mmFloor(
                        Oracles.blockQuantView(w, type, elems),
                        0,
                        k,
                        Raw.f32(Oracles.f32View(x, k), "a"),
                        0,
                        k,
                        Raw.f32(Oracles.f32View(out, m), "c"),
                        0,
                        m,
                        m,
                        1,
                        k,
                        dt,
                        false);
                long rowBytes = type.byteSizeFor(k);
                for (int r = 0; r < m; r++) {
                    double expected = 0;
                    for (int i = 0; i < k; i++) {
                        expected +=
                                (double) MatMul.getLegacy(w, r * rowBytes, i, dt)
                                        * x.getAtIndex(ValueLayout.JAVA_FLOAT, i);
                    }
                    float actual = out.getAtIndex(ValueLayout.JAVA_FLOAT, r);
                    assertEquals(
                            expected,
                            actual,
                            1e-4 * (1 + Math.abs(expected)),
                            type + " row " + r + " k=" + k + " at " + width);
                }
            }
        }
    }
}
