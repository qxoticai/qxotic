package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class SituMultiplyTest {

    /** Whole vectors and a scalar tail, against the definition in double precision. */
    @Test
    void matchesTheDefinition() {
        MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(Arena.ofAuto());
        int n = 67;
        float beta = 4f, linearBeta = 25f;
        float[] gate = new float[n], up = new float[n];
        for (int i = 0; i < n; i++) {
            gate[i] = (i - 33) * 0.37f;
            up[i] = (i % 2 == 0 ? 1 : -1) * (i * 0.9f - 7f);
        }
        MemoryView<MemorySegment> gateView = Views.fromFloatArray(memory, gate);
        Activations.situMultiply(
                gateView, 0, Views.fromFloatArray(memory, up), 0, n, beta, linearBeta);
        for (int i = 0; i < n; i++) {
            double act = beta * Math.tanh(gate[i] / beta) / (1 + Math.exp(-gate[i]));
            double linear = linearBeta * Math.tanh(up[i] / linearBeta);
            // the vector tanh is within 1.9e-5 and the sigmoid within 1e-5, absolute
            double tolerance = 1.2e-4 * Math.abs(linear) + 5e-4 * Math.abs(act) + 1e-6;
            assertEquals(
                    act * linear, Views.getFloat(gateView, i, "gate"), tolerance, "element " + i);
        }
    }
}
