package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The model-free pieces, each against the value llama.cpp (or libm) gives. */
final class Lfm3UnitTest {

    @Test
    void smartResizeMatchesTheSizesMtmdPicked() {
        long min = 64 * 1024, max = 4096 * 1024; // LFM3-VL: 64..4096 tokens of 32x32 pixels
        assertArrayEquals(
                new int[] {512, 608}, Lfm3VisionPreprocess.targetSize(519, 600, 32, min, max));
        assertArrayEquals(
                new int[] {800, 608}, Lfm3VisionPreprocess.targetSize(800, 600, 32, min, max));
        assertArrayEquals(
                new int[] {320, 224}, Lfm3VisionPreprocess.targetSize(100, 70, 32, min, max));
        assertArrayEquals(
                new int[] {1984, 512}, Lfm3VisionPreprocess.targetSize(2000, 500, 32, min, max));
        // over the budget: scaled down keeping the aspect, floor to the factor
        assertArrayEquals(
                new int[] {2496, 1664}, Lfm3VisionPreprocess.targetSize(3000, 2000, 32, min, max));
        // round half to even, as Python's round(): 16.5 -> 16, 17.5 -> 18
        assertArrayEquals(
                new int[] {512, 576}, Lfm3VisionPreprocess.targetSize(528, 560, 32, min, max));
    }

    @Test
    void erfIsLibmsToTheLastBit() {
        // erff() of glibc at these points
        assertEquals(0f, GeluErf.erf(0f));
        assertEquals(0.5204998778f, GeluErf.erf(0.5f));
        assertEquals(0.8427007929f, GeluErf.erf(1f));
        assertEquals(-0.9953222650f, GeluErf.erf(-2f));
        assertEquals(0.9999999846f, GeluErf.erf(3.9f), 1e-7f);
        assertEquals(1f, GeluErf.erf(6f));
    }

    @Test
    void toolValuesRenderAsTheTemplatesRenderValue() {
        assertEquals("Paris", Lfm3ChatTemplate.value("Paris"));
        assertEquals("true", Lfm3ChatTemplate.value(true));
        assertEquals("null", Lfm3ChatTemplate.value(null));
        assertEquals("3", Lfm3ChatTemplate.value(3L));
        assertEquals("[1, 2]", Lfm3ChatTemplate.value(List.of(1L, 2L)));
        assertEquals("{\"a\": \"b\"}", Lfm3ChatTemplate.value(Map.of("a", "b")));
    }
}
