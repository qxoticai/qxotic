package com.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Field;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class IntSequenceBuilderTest {
    private IntSequence.Builder builder;

    @BeforeEach
    public void setUp() {
        builder = IntSequence.newBuilder();
    }

    @Test
    public void testNewBuilderDefault() {
        assertNotNull(builder);
        assertEquals(0, builder.size());
        assertTrue(builder.isEmpty());
    }

    @Test
    public void testNewBuilderWithCapacity() {
        for (int capacity : new int[] {0, 1, 10, 100, 1000}) {
            IntSequence.Builder builder = IntSequence.newBuilder(capacity);
            assertNotNull(builder);
            assertEquals(0, builder.size());
        }
    }

    @Test
    public void testAdd() {
        builder.add(1);
        assertEquals(1, builder.size());
        assertEquals(1, builder.asSequenceView().intAt(0));

        builder.add(2).add(3); // Testing chaining
        assertEquals(3, builder.size());
        assertEquals(2, builder.asSequenceView().intAt(1));
        assertEquals(3, builder.asSequenceView().intAt(2));
    }

    @Test
    public void testBuilderAsSequenceView() {
        builder.add(1).add(2).add(3);
        IntSequence view = builder.asSequenceView();

        assertEquals(3, view.length());
        assertFalse(view.isEmpty());
        assertEquals(1, view.getFirst());
        assertEquals(3, view.getLast());
        assertArrayEquals(new int[] {1, 2, 3}, view.toArray());

        // Test subSequence
        IntSequence subSeq = view.subSequence(1, 3);
        assertArrayEquals(new int[] {2, 3}, subSeq.toArray());
    }

    @Test
    public void testEnsureCapacity() {
        builder.ensureCapacity(100);
        // Add more elements than initial capacity to verify it grows correctly
        for (int i = 0; i < 150; i++) {
            builder.add(i);
        }
        assertEquals(150, builder.size());
        IntSequence view = builder.asSequenceView();
        for (int i = 0; i < 150; i++) {
            assertEquals(i, view.intAt(i));
        }
    }

    @Test
    public void testAddAll() {
        // Test adding from another sequence
        IntSequence seq1 = IntSequence.of(1, 2, 3);
        builder.addAll(seq1);
        assertEquals(3, builder.size());
        assertArrayEquals(new int[] {1, 2, 3}, builder.asSequenceView().toArray());

        // Test adding from another builder
        IntSequence.Builder builder2 = IntSequence.newBuilder();
        builder2.add(4).add(5);
        builder.addAll(builder2);
        assertEquals(5, builder.size());
        assertArrayEquals(new int[] {1, 2, 3, 4, 5}, builder.asSequenceView().toArray());
    }

    @Test
    public void testAddAllEmpty() {
        IntSequence emptySeq = IntSequence.of();
        builder.add(1);
        builder.addAll(emptySeq);
        assertEquals(1, builder.size());
        assertEquals(1, builder.asSequenceView().intAt(0));
    }

    @Test
    public void testBuild() {
        builder.add(1).add(2).add(3);
        IntSequence sequence = builder.build();

        // Verify the built sequence
        assertEquals(3, sequence.length());
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());

        // Verify builder can still be used after build
        builder.add(4);
        assertEquals(4, builder.size());
        assertEquals(4, builder.asSequenceView().intAt(3));

        // Verify the previously built sequence remains unchanged
        assertEquals(3, sequence.length());
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());
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
        assertArrayEquals(new int[] {1, 2}, seq1.toArray());

        // Verify second sequence includes all elements
        assertArrayEquals(new int[] {1, 2, 3, 4}, seq2.toArray());
    }

    @Test
    public void testBuilderSelfAddAll() {
        builder.add(1).add(2);
        IntSequence original = builder.build();

        builder.addAll(builder);
        IntSequence doubled = builder.build();

        // Original builder content should be [1, 2, 1, 2]
        assertArrayEquals(new int[] {1, 2}, original.toArray());
        assertArrayEquals(new int[] {1, 2, 1, 2}, doubled.toArray());
    }

    @Test
    public void testSnapshotIsFixedLength() {
        builder.add(1).add(2);
        IntSequence snapshot = builder.snapshot();

        builder.add(3);

        assertArrayEquals(new int[] {1, 2}, snapshot.toArray());
        assertArrayEquals(new int[] {1, 2, 3}, builder.asSequenceView().toArray());
    }

    @Test
    public void testSnapshotReflectsUnderlyingElementMutationButNotSizeGrowth() throws Exception {
        builder.add(10).add(20).add(30);
        IntSequence snapshot = builder.snapshot();
        IntSequence live = builder.asSequenceView();

        // mutate backing data in-place (same size)
        Field dataField = builder.getClass().getDeclaredField("data");
        dataField.setAccessible(true);
        int[] data = (int[]) dataField.get(builder);
        data[1] = 999;

        assertArrayEquals(new int[] {10, 999, 30}, snapshot.toArray());
        assertArrayEquals(new int[] {10, 999, 30}, live.toArray());

        // grow builder size afterwards
        builder.add(40).add(50);
        assertEquals(3, snapshot.length());
        assertEquals(5, live.length());
        assertArrayEquals(new int[] {10, 999, 30}, snapshot.toArray());
        assertArrayEquals(new int[] {10, 999, 30, 40, 50}, live.toArray());
    }

    @Test
    public void testSequenceViewIsLive() {
        builder.add(7);
        IntSequence view = builder.asSequenceView();
        assertArrayEquals(new int[] {7}, view.toArray());

        builder.add(8).add(9);
        assertArrayEquals(new int[] {7, 8, 9}, view.toArray());
    }

    @Test
    public void testAsSequenceViewSubSequenceTracksLiveRange() {
        builder.add(1).add(2).add(3);
        IntSequence live = builder.asSequenceView();

        IntSequence firstTwo = live.subSequence(0, 2);
        assertArrayEquals(new int[] {1, 2}, firstTwo.toArray());

        builder.add(4);
        assertArrayEquals(new int[] {1, 2}, firstTwo.toArray());
        assertArrayEquals(new int[] {1, 2, 3, 4}, live.toArray());
    }

    @Test
    public void testSnapshotAndViewRangeValidation() {
        builder.add(1).add(2).add(3);
        IntSequence snapshot = builder.snapshot();
        IntSequence live = builder.asSequenceView();

        assertThrows(IndexOutOfBoundsException.class, () -> snapshot.intAt(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> snapshot.intAt(3));
        assertThrows(IndexOutOfBoundsException.class, () -> snapshot.subSequence(-1, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> snapshot.subSequence(2, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> snapshot.subSequence(0, 4));

        assertThrows(IndexOutOfBoundsException.class, () -> live.intAt(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> live.subSequence(0, 4));
    }

    @Test
    public void testBuilderIteration() {
        builder.add(1).add(2).add(3);
        IntSequence view = builder.asSequenceView();

        int sum = 0;
        for (int value : view) {
            sum += value;
        }
        assertEquals(6, sum);
    }

    @Test
    public void testBuilderStream() {
        builder.add(1).add(2).add(3);
        IntSequence view = builder.asSequenceView();

        int sum = view.stream().sum();
        assertEquals(6, sum);

        long count = view.stream().count();
        assertEquals(3, count);
    }

    @Test
    public void testLargeCapacityAndAdd() {
        IntSequence.Builder largeBuilder = IntSequence.newBuilder(1_000_000);
        for (int i = 0; i < 1_000_000; i++) {
            largeBuilder.add(i);
        }
        assertEquals(1_000_000, largeBuilder.size());
        IntSequence view = largeBuilder.asSequenceView();
        assertEquals(0, view.intAt(0));
        assertEquals(999_999, view.intAt(999_999));
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
        builder.asSequenceView().stream().forEach(builder::add);
        builder.add(1);
        builder.asSequenceView().stream().forEach(builder::add);
        assertEquals(2, builder.size());
    }

    @Test
    public void testAddAllToItself() {
        builder.add(1);
        builder.addAll(builder);
        assertEquals(2, builder.size());
    }

    @Test
    public void testAddAllBuilderReadsFixedSourceLength() {
        IntSequence.Builder source = IntSequence.newBuilder();
        source.add(1).add(2).add(3);

        IntSequence.Builder dest = IntSequence.newBuilder();
        dest.add(9);
        dest.addAll(source);

        assertArrayEquals(new int[] {9, 1, 2, 3}, dest.build().toArray());

        source.add(4).add(5);
        assertArrayEquals(new int[] {9, 1, 2, 3}, dest.build().toArray());
        assertArrayEquals(new int[] {1, 2, 3, 4, 5}, source.build().toArray());
    }
}
