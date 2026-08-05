package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The sampling primitives ({@code max}, {@code maskBelowInPlace}, {@code expSum}, {@code
 * collectAtOrAbove}, {@code kthLargestThreshold}) against scalar references, across sizes that
 * exercise the SIMD main loop, the scalar tail, and tail-only spans - the F32 overrides and the
 * FloatTensor defaults must agree bit-for-bit (expSum: within FastMath's accuracy contract).
 */
class SamplingPrimitivesTest {

    // below species length (tail-only), exact multiples, off-by-one around lane boundaries
    private static final int[] SIZES = {1, 2, 15, 16, 17, 100, 1000, 4099};

    private static float[] logits(int n, long seed) {
        Random r = new Random(seed);
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * 4);
        if (n > 4) {
            a[n / 3] = Float.NEGATIVE_INFINITY; // pre-masked entries must stay inert
            a[n / 2] = a[0]; // duplicates
        }
        return a;
    }

    private static F32FloatTensor tensor(float[] values) {
        F32FloatTensor t = F32FloatTensor.allocate(Arena.ofAuto(), values.length);
        for (int i = 0; i < values.length; i++) t.setFloat(i, values[i]);
        return t;
    }

    private static float[] toArray(F32FloatTensor t, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = t.getFloat(i);
        return a;
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 15, 16, 17, 100, 1000, 4099})
    void maxMatchesScalarReference(int n) {
        float[] a = logits(n, n);
        float expected = Float.NEGATIVE_INFINITY;
        for (float v : a) expected = Math.max(expected, v);
        assertEquals(expected, tensor(a).max(0, n));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 15, 16, 17, 100, 1000, 4099})
    void maskBelowMatchesScalarReference(int n) {
        float[] a = logits(n, n + 1);
        float threshold = a[n - 1]; // an actual value: exercises the >= boundary
        float[] expected = a.clone();
        for (int i = 0; i < n; i++) {
            if (expected[i] < threshold) expected[i] = Float.NEGATIVE_INFINITY;
        }
        F32FloatTensor t = tensor(a);
        t.maskBelowInPlace(0, n, threshold);
        assertTrue(Arrays.equals(expected, toArray(t, n)));
    }

    @Test
    void maskBelowExtremes() {
        float[] a = logits(100, 7);
        F32FloatTensor t = tensor(a);
        t.maskBelowInPlace(0, 100, Float.NEGATIVE_INFINITY); // nothing is below -inf
        assertTrue(Arrays.equals(a, toArray(t, 100)));
        t.maskBelowInPlace(0, 100, Float.POSITIVE_INFINITY); // everything is below +inf
        for (int i = 0; i < 100; i++) assertEquals(Float.NEGATIVE_INFINITY, t.getFloat(i));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 15, 16, 17, 100, 1000, 4099})
    void expSumMatchesScalarWithinFastMathAccuracy(int n) {
        float[] a = logits(n, n + 2);
        float max = Float.NEGATIVE_INFINITY;
        for (float v : a) max = Math.max(max, v);
        double expected = 0;
        for (float v : a) expected += Math.exp(v - max);
        double actual = tensor(a).expSum(0, n, max);
        assertEquals(expected, actual, expected * 1e-3, "n=" + n);
    }

    @Test
    void expSumIsReadOnly() {
        float[] a = logits(1000, 11);
        F32FloatTensor t = tensor(a);
        t.expSum(0, 1000, t.max(0, 1000));
        assertTrue(Arrays.equals(a, toArray(t, 1000)));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 15, 16, 17, 100, 1000, 4099})
    void collectAtOrAboveMatchesScalarReference(int n) {
        float[] a = logits(n, n + 3);
        float threshold = a[(2 * n) / 3];
        // scalar reference: surviving indices in order, everything else masked
        var expectedIds = new java.util.ArrayList<Integer>();
        float[] expected = a.clone();
        for (int i = 0; i < n; i++) {
            if (a[i] >= threshold) expectedIds.add(i);
            else expected[i] = Float.NEGATIVE_INFINITY;
        }
        F32FloatTensor t = tensor(a);
        int[] out = new int[n];
        int count = t.collectAtOrAbove(0, n, threshold, out);
        assertEquals(expectedIds.size(), count);
        for (int i = 0; i < count; i++) assertEquals(expectedIds.get(i), out[i]);
        assertTrue(Arrays.equals(expected, toArray(t, n)));
    }

    @Test
    void collectAtOrAboveAllAndNone() {
        float[] a = logits(200, 13);
        int[] out = new int[200];
        assertEquals(0, tensor(a).collectAtOrAbove(0, 200, Float.POSITIVE_INFINITY, out));
        F32FloatTensor t = tensor(a);
        // -inf entries stay below any finite threshold but everything finite survives
        int finite = 0;
        for (float v : a) if (v != Float.NEGATIVE_INFINITY) finite++;
        assertEquals(finite, t.collectAtOrAbove(0, 200, -1e30f, out));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 5, 40, 100})
    void kthLargestThresholdMatchesSortedReference(int k) {
        int n = 1000;
        float[] a = logits(n, k);
        float[] sorted = a.clone();
        Arrays.sort(sorted); // ascending
        float expected = sorted[n - k];
        assertEquals(expected, tensor(a).kthLargestThreshold(0, n, new float[k]), "k=" + k);
    }

    @Test
    void kthLargestHandlesTiesAndReuse() {
        float[] a = new float[100];
        Arrays.fill(a, 1f);
        a[7] = 5f;
        F32FloatTensor t = tensor(a);
        float[] heap = new float[3];
        assertEquals(1f, t.kthLargestThreshold(0, 100, heap)); // 5,1,1 -> third largest is 1
        // scratch reuse: a second call with different data must not see stale heap state
        float[] b = new float[100];
        for (int i = 0; i < 100; i++) b[i] = i;
        assertEquals(97f, tensor(b).kthLargestThreshold(0, 100, heap));
    }
}
