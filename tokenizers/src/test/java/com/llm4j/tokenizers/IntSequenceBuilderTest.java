package com.llm4j.tokenizers;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class IntSequenceBuilderTest {
    private IntSequence.Builder builder;

    @BeforeEach
    public void setUp() {
        builder = IntSequence.newBuilder();
    }

    @Test
    public void testNewBuilderDefault() {
        assertNotNull(builder);
        assertEquals(0, builder.length());
        assertTrue(builder.isEmpty());
    }

    @Test
    public void testNewBuilderWithCapacity() {
        for (int capacity : new int[]{0, 1, 10, 100, 1000}) {
            IntSequence.Builder builder = IntSequence.newBuilder(capacity);
            assertNotNull(builder);
            assertEquals(0, builder.length());
        }
    }

    @Test
    public void testAdd() {
        builder.add(1);
        assertEquals(1, builder.length());
        assertEquals(1, builder.intAt(0));

        builder.add(2).add(3); // Testing chaining
        assertEquals(3, builder.length());
        assertEquals(2, builder.intAt(1));
        assertEquals(3, builder.intAt(2));
    }

    @Test
    public void testBuilderAsIntSequence() {
        builder.add(1).add(2).add(3);

        // Test that Builder implements IntSequence methods correctly
        assertEquals(3, builder.length());
        assertFalse(builder.isEmpty());
        assertEquals(1, builder.getFirst());
        assertEquals(3, builder.getLast());
        assertArrayEquals(new int[]{1, 2, 3}, builder.toArray());

        // Test subSequence
        IntSequence subSeq = builder.subSequence(1, 3);
        assertArrayEquals(new int[]{2, 3}, subSeq.toArray());
    }

    @Test
    public void testEnsureCapacity() {
        builder.ensureCapacity(100);
        // Add more elements than initial capacity to verify it grows correctly
        for (int i = 0; i < 150; i++) {
            builder.add(i);
        }
        assertEquals(150, builder.length());
        for (int i = 0; i < 150; i++) {
            assertEquals(i, builder.intAt(i));
        }
    }

    @Test
    public void testAddAll() {
        // Test adding from another sequence
        IntSequence seq1 = IntSequence.of(1, 2, 3);
        builder.addAll(seq1);
        assertEquals(3, builder.length());
        assertArrayEquals(new int[]{1, 2, 3}, builder.toArray());

        // Test adding from another builder
        IntSequence.Builder builder2 = IntSequence.newBuilder();
        builder2.add(4).add(5);
        builder.addAll(builder2);
        assertEquals(5, builder.length());
        assertArrayEquals(new int[]{1, 2, 3, 4, 5}, builder.toArray());
    }

    @Test
    public void testAddAllEmpty() {
        IntSequence emptySeq = IntSequence.of();
        builder.add(1);
        builder.addAll(emptySeq);
        assertEquals(1, builder.length());
        assertEquals(1, builder.intAt(0));
    }

    @Test
    public void testBuild() {
        builder.add(1).add(2).add(3);
        IntSequence sequence = builder.build();

        // Verify the built sequence
        assertEquals(3, sequence.length());
        assertArrayEquals(new int[]{1, 2, 3}, sequence.toArray());

        // Verify builder can still be used after build
        builder.add(4);
        assertEquals(4, builder.length());
        assertEquals(4, builder.intAt(3));

        // Verify the previously built sequence remains unchanged
        assertEquals(3, sequence.length());
        assertArrayEquals(new int[]{1, 2, 3}, sequence.toArray());
    }

    @Test
    public void testBuilderReuse() {
        // First use
        builder.add(1).add(2);
        IntSequence seq1 = builder.build();

        // Second use
        builder.add(3).add(4);
        IntSequence seq2 = builder.build();

        // Verify first sequence
        assertArrayEquals(new int[]{1, 2}, seq1.toArray());

        // Verify second sequence includes all elements
        assertArrayEquals(new int[]{1, 2, 3, 4}, seq2.toArray());
    }

    @Test
    public void testBuilderSelfAddAll() {
        builder.add(1).add(2);
        IntSequence original = builder.build();

        builder.addAll(builder);
        IntSequence doubled = builder.build();

        // Original builder content should be [1, 2, 1, 2]
        assertArrayEquals(new int[]{1, 2}, original.toArray());
        assertArrayEquals(new int[]{1, 2, 1, 2}, doubled.toArray());
    }

    @Test
    public void testBuilderIteration() {
        builder.add(1).add(2).add(3);

        int sum = 0;
        for (int value : builder) {
            sum += value;
        }
        assertEquals(6, sum);
    }

    @Test
    public void testBuilderStream() {
        builder.add(1).add(2).add(3);

        int sum = builder.stream().sum();
        assertEquals(6, sum);

        long count = builder.stream().count();
        assertEquals(3, count);
    }

    @Test
    public void testLargeCapacityAndAdd() {
        IntSequence.Builder largeBuilder = IntSequence.newBuilder(1_000_000);
        for (int i = 0; i < 1_000_000; i++) {
            largeBuilder.add(i);
        }
        assertEquals(1_000_000, largeBuilder.length());
        assertEquals(0, largeBuilder.intAt(0));
        assertEquals(999_999, largeBuilder.intAt(999_999));
    }

    @Test
    public void testNegativeCapacity() {
        assertThrows(IllegalArgumentException.class, () -> IntSequence.newBuilder(-1));
    }

    @Test
    public void testNegativeEnsureCapacity() {
        assertThrows(IllegalArgumentException.class, () -> builder.ensureCapacity(-1));
    }

    @Disabled("(Concurrent) modification tracking is not yet implemented")
    @Test
    public void testStreamAddToItself() {
        builder.stream().forEach(builder::add);
        builder.add(1);
        builder.stream().forEach(builder::add);
        assertEquals(2, builder.length());
    }

    @Test
    public void testAddAllToItself() {
        builder.add(1);
        builder.addAll(builder);
        assertEquals(2, builder.length());
    }
}
