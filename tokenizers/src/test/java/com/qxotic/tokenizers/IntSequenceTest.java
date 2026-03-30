package com.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
import org.junit.jupiter.api.Test;

public class IntSequenceTest {
    private final IntSequence emptySequence = IntSequence.of();
    private final IntSequence singleElementSequence = IntSequence.of(42);
    private final IntSequence multiElementSequence = IntSequence.of(1, 2, 3, 4, 5);

    @Test
    void testOf() {
        IntSequence sequence = IntSequence.of(1, 2, 3);
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());
    }

    @Test
    void testOfMakesDefensiveCopy() {
        int[] backing = {1, 2, 3};
        IntSequence sequence = IntSequence.of(backing);
        backing[1] = 99;
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());
    }

    @Test
    void testWrapArray() {
        int[] array = {1, 2, 3};
        IntSequence sequence = IntSequence.wrap(array);
        assertArrayEquals(array, sequence.toArray());

        array[0] = 99;
        assertEquals(99, sequence.intAt(0));
    }

    @Test
    void testWrapArrayNull() {
        assertThrows(NullPointerException.class, () -> IntSequence.wrap((int[]) null));
    }

    @Test
    void testWrapList() {
        List<Integer> list = new java.util.ArrayList<>(Arrays.asList(1, 2, 3));
        IntSequence sequence = IntSequence.wrap(list);
        assertEquals(list, sequence.toList());

        list.set(1, 77);
        assertEquals(77, sequence.intAt(1));
    }

    @Test
    void testWrapListNull() {
        assertThrows(NullPointerException.class, () -> IntSequence.wrap((List<Integer>) null));
    }

    @Test
    void testCopyOfArray() {
        int[] source = {1, 2, 3};
        IntSequence sequence = IntSequence.copyOf(source);
        source[0] = 99;
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());
    }

    @Test
    void testCopyOfArrayNull() {
        assertThrows(NullPointerException.class, () -> IntSequence.copyOf((int[]) null));
    }

    @Test
    void testCopyOfList() {
        List<Integer> source = new java.util.ArrayList<>(Arrays.asList(1, 2, 3));
        IntSequence sequence = IntSequence.copyOf(source);
        source.set(0, 99);
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());
    }

    @Test
    void testCopyOfListNull() {
        assertThrows(NullPointerException.class, () -> IntSequence.copyOf((List<Integer>) null));
    }

    @Test
    void testToArrayIsIndependent() {
        IntSequence sequence = IntSequence.wrap(new int[] {1, 2, 3});
        int[] copy = sequence.toArray();
        copy[0] = 42;
        assertEquals(1, sequence.intAt(0));
    }

    @Test
    void testLength() {
        assertEquals(0, emptySequence.length());
        assertEquals(1, singleElementSequence.length());
        assertEquals(5, multiElementSequence.length());
    }

    @Test
    void testIntAt() {
        assertEquals(42, singleElementSequence.intAt(0));
        assertEquals(1, multiElementSequence.intAt(0));
        assertEquals(5, multiElementSequence.intAt(4));
    }

    @Test
    void testIntAtOutOfBounds() {
        assertThrows(IndexOutOfBoundsException.class, () -> emptySequence.intAt(0));
        assertThrows(IndexOutOfBoundsException.class, () -> multiElementSequence.intAt(5));
        assertThrows(IndexOutOfBoundsException.class, () -> multiElementSequence.intAt(-1));
    }

    @Test
    void testSubSequence() {
        IntSequence subSeq = multiElementSequence.subSequence(1, 4);
        assertArrayEquals(new int[] {2, 3, 4}, subSeq.toArray());
    }

    @Test
    void testToArray() {
        assertArrayEquals(new int[] {}, emptySequence.toArray());
        assertArrayEquals(new int[] {42}, singleElementSequence.toArray());
        assertArrayEquals(new int[] {1, 2, 3, 4, 5}, multiElementSequence.toArray());
    }

    @Test
    void testToList() {
        assertEquals(Arrays.asList(), emptySequence.toList());
        assertEquals(Arrays.asList(42), singleElementSequence.toList());
        assertEquals(Arrays.asList(1, 2, 3, 4, 5), multiElementSequence.toList());
    }

    @Test
    void testIterator() {
        List<Integer> collected =
                StreamSupport.stream(multiElementSequence.spliterator(), false)
                        .collect(Collectors.toList());
        assertEquals(Arrays.asList(1, 2, 3, 4, 5), collected);
    }

    @Test
    void testIteratorNoSuchElement() {
        var iterator = emptySequence.iterator();
        assertThrows(NoSuchElementException.class, iterator::nextInt);
    }

    @Test
    void testStream() {
        int sum = multiElementSequence.stream().sum();
        assertEquals(15, sum);
    }

    @Test
    void testGetFirst() {
        assertThrows(NoSuchElementException.class, emptySequence::getFirst);
        assertEquals(42, singleElementSequence.getFirst());
        assertEquals(1, multiElementSequence.getFirst());
    }

    @Test
    void testGetLast() {
        assertThrows(NoSuchElementException.class, emptySequence::getLast);
        assertEquals(42, singleElementSequence.getLast());
        assertEquals(5, multiElementSequence.getLast());
    }

    @Test
    void testIsEmpty() {
        assertTrue(emptySequence.isEmpty());
        assertFalse(singleElementSequence.isEmpty());
        assertFalse(multiElementSequence.isEmpty());
    }

    @Test
    void testBuilder() {
        IntSequence.Builder builder = IntSequence.newBuilder();
        IntSequence sequence = builder.add(1).add(2).add(3).build();
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());
    }

    @Test
    void testBuilderWithInitialCapacity() {
        IntSequence.Builder builder = IntSequence.newBuilder(10);
        IntSequence sequence = builder.add(1).add(2).add(3).build();
        assertArrayEquals(new int[] {1, 2, 3}, sequence.toArray());
    }

    @Test
    void testBuilderAddAll() {
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(1).add(2);
        builder.addAll(IntSequence.of(3, 4, 5));
        IntSequence sequence = builder.build();
        assertArrayEquals(new int[] {1, 2, 3, 4, 5}, sequence.toArray());
    }

    @Test
    void testBuilderSelfAddAll() {
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(1).add(2);
        builder.addAll(builder);
        IntSequence sequence = builder.build();
        assertArrayEquals(new int[] {1, 2, 1, 2}, sequence.toArray());
    }

    @Test
    void testBuilderSnapshotAndLiveView() {
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(1).add(2);

        IntSequence snapshot = builder.snapshot();
        IntSequence live = builder.asSequenceView();

        builder.add(3);

        assertArrayEquals(new int[] {1, 2}, snapshot.toArray());
        assertArrayEquals(new int[] {1, 2, 3}, live.toArray());
    }

    @Test
    void testCopyTo() {
        IntSequence sequence = IntSequence.of(1, 2, 3);
        int[] dest = new int[] {9, 9, 9, 9, 9};
        sequence.copyTo(dest, 1);
        assertArrayEquals(new int[] {9, 1, 2, 3, 9}, dest);
    }

    @Test
    void testCopyToWithCount() {
        IntSequence sequence = IntSequence.of(1, 2, 3, 4);
        int[] dest = new int[] {9, 9, 9, 9, 9, 9};
        sequence.copyTo(dest, 2, 3);
        assertArrayEquals(new int[] {9, 9, 1, 2, 3, 9}, dest);
    }

    @Test
    void testCopyToWithSourceOffset() {
        IntSequence sequence = IntSequence.of(1, 2, 3, 4, 5);
        int[] dest = new int[] {9, 9, 9, 9, 9, 9};
        sequence.copyTo(2, dest, 1, 2);
        assertArrayEquals(new int[] {9, 3, 4, 9, 9, 9}, dest);
    }

    @Test
    void testCopyToEdgeCases() {
        IntSequence sequence = IntSequence.of(1, 2, 3);

        int[] exact = new int[3];
        sequence.copyTo(exact, 0);
        assertArrayEquals(new int[] {1, 2, 3}, exact);

        assertThrows(NullPointerException.class, () -> sequence.copyTo(null, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(new int[3], -1));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(new int[3], 4));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(new int[3], 1));

        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(new int[3], 0, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(new int[3], 0, 4));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(new int[3], 1, 3));

        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(-1, new int[3], 0, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(4, new int[3], 0, 1));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(1, new int[3], 0, 3));
        assertThrows(IndexOutOfBoundsException.class, () -> sequence.copyTo(1, new int[3], 3, 1));

        int[] untouched = new int[] {7, 8, 9};
        sequence.copyTo(3, untouched, 1, 0);
        assertArrayEquals(new int[] {7, 8, 9}, untouched);
    }

    @Test
    void testForEachInt() {
        IntSequence sequence = IntSequence.of(1, 2, 3, 4);
        int[] sum = new int[1];
        sequence.forEachInt(v -> sum[0] += v);
        assertEquals(10, sum[0]);
    }

    @Test
    void testForEachIntOrderAndNullAction() {
        IntSequence sequence = IntSequence.of(3, 1, 4, 1, 5);
        StringBuilder seen = new StringBuilder();
        AtomicInteger count = new AtomicInteger(0);

        sequence.forEachInt(
                value -> {
                    if (seen.length() > 0) {
                        seen.append(',');
                    }
                    seen.append(value);
                    count.incrementAndGet();
                });

        assertEquals("3,1,4,1,5", seen.toString());
        assertEquals(sequence.length(), count.get());
        assertThrows(NullPointerException.class, () -> sequence.forEachInt(null));
    }

    @Test
    void testToIntStream() {
        IntSequence sequence = IntSequence.of(1, 2, 3, 4);
        assertEquals(10, sequence.toIntStream().sum());
    }

    @Test
    void testToIntStreamMatchesStream() {
        IntSequence sequence = IntSequence.of(10, 20, 30);
        assertEquals(sequence.stream().sum(), sequence.toIntStream().sum());
        assertEquals(sequence.stream().count(), sequence.toIntStream().count());
    }

    @Test
    void testConcatStartsWithEndsWith() {
        IntSequence left = IntSequence.of(1, 2);
        IntSequence right = IntSequence.of(3, 4);

        IntSequence merged = left.concat(right);
        assertArrayEquals(new int[] {1, 2, 3, 4}, merged.toArray());
        assertTrue(merged.startsWith(left));
        assertTrue(merged.endsWith(right));
        assertFalse(merged.startsWith(IntSequence.of(2, 3)));
        assertFalse(merged.endsWith(IntSequence.of(2, 3)));

        IntSequence all =
                IntSequences.concat(IntSequence.of(1), IntSequence.of(), IntSequence.of(2));
        assertArrayEquals(new int[] {1, 2}, all.toArray());
    }

    @Test
    void testStartsWithEndsWithEdgeCases() {
        IntSequence sequence = IntSequence.of(1, 2, 3);

        assertTrue(sequence.startsWith(IntSequence.empty()));
        assertTrue(sequence.endsWith(IntSequence.empty()));
        assertTrue(sequence.startsWith(sequence));
        assertTrue(sequence.endsWith(sequence));

        assertFalse(sequence.startsWith(IntSequence.of(1, 2, 3, 4)));
        assertFalse(sequence.endsWith(IntSequence.of(1, 2, 3, 4)));

        assertThrows(NullPointerException.class, () -> sequence.startsWith(null));
        assertThrows(NullPointerException.class, () -> sequence.endsWith(null));
    }

    @Test
    void testConcatEdgeCases() {
        IntSequence a = IntSequence.of(1, 2);
        IntSequence b = IntSequence.of(3);

        assertSame(a, a.concat(IntSequence.empty()));
        assertSame(b, IntSequence.empty().concat(b));
        assertEquals(IntSequence.empty(), IntSequences.concatAll());
        assertEquals(
                IntSequence.empty(),
                IntSequences.concatAll(IntSequence.empty(), IntSequence.empty()));

        assertArrayEquals(new int[] {1, 2, 3}, IntSequences.concat(a, b).toArray());

        assertThrows(NullPointerException.class, () -> a.concat((IntSequence) null));
        assertThrows(
                NullPointerException.class, () -> IntSequences.concatAll((IntSequence[]) null));
        assertThrows(NullPointerException.class, () -> IntSequences.concat(a, null, b));
    }

    @Test
    void testCompareAvoidsOverflow() {
        IntSequence min = IntSequence.of(Integer.MIN_VALUE);
        IntSequence max = IntSequence.of(Integer.MAX_VALUE);
        assertTrue(IntSequences.compare(min, max) < 0);
        assertTrue(IntSequences.compare(max, min) > 0);
    }

    @Test
    void testSubSequenceEdgeCases() {
        IntSequence intSeq = multiElementSequence;
        List<Integer> list = intSeq.toList();
        for (int start = 0; start < intSeq.length(); ++start) {
            for (int end = start; end < intSeq.length(); ++end) {
                IntSequence subSeq = intSeq.subSequence(start, end);
                List<Integer> subList = list.subList(start, end);
                assertEquals(subList, subSeq.toList());
            }
        }
    }

    @Test
    void testToString() {
        assertEquals("[]", IntSequence.empty().toString());
        assertEquals("[42]", IntSequence.of(42).toString());
        assertEquals("[1, 2, 3]", IntSequence.of(1, 2, 3).toString());
    }

    @Test
    void testToStringExtended() {
        assertEquals("{}", IntSequence.empty().toString("", "{", "}"));
        assertEquals("<42>", IntSequence.of(42).toString(" & ", "<", ">"));
        assertEquals("<1 & 2 & 3>", IntSequence.of(1, 2, 3).toString(" & ", "<", ">"));
        assertEquals("123", IntSequence.of(1, 2, 3).toString("", "", ""));
    }

    @Test
    void testEquals() {
        assertEquals(IntSequence.wrap(new int[0]), IntSequence.wrap(List.of()));
        assertEquals(IntSequence.of(1), IntSequence.of(1, 2, 3).subSequence(0, 1));
        assertEquals(IntSequence.of(1, 2, 3).subSequence(0, 1), IntSequence.of(1));
        assertEquals(IntSequence.of(1, 2, 3), IntSequence.of(1, 2, 3));

        assertEquals(new ZeroIntSequence(3), IntSequence.of(0, 0, 0));
        assertEquals(EmptyIntSequence.get(), IntSequence.empty());

        assertNotEquals(IntSequence.of(1), IntSequence.of(1, 2, 3));
        assertNotEquals(IntSequence.wrap(List.of(1)), IntSequence.of(2));
    }

    @Test
    void testHashCode() {
        assertEquals(IntSequence.empty().hashCode(), Arrays.hashCode(new int[0]));
        int hash123 = Arrays.hashCode(new int[] {1, 2, 3});
        assertEquals(IntSequence.of(1, 2, 3).hashCode(), hash123);
        assertEquals(IntSequence.wrap(new int[] {1, 2, 3}).hashCode(), hash123);
        assertEquals(IntSequence.wrap(List.of(1, 2, 3)).hashCode(), hash123);
        assertEquals(new ZeroIntSequence(3).hashCode(), Objects.hash(0, 0, 0));
        assertEquals(EmptyIntSequence.get().hashCode(), IntSequence.empty().hashCode());
        assertNotEquals(IntSequence.empty().hashCode(), IntSequence.of(1).hashCode());
        assertEquals(
                IntSequence.of(1, 42, 3).subSequence(1, 2).hashCode(),
                Arrays.hashCode(new int[] {42}));
    }

    static void assertCompare(int expected, IntSequence first, IntSequence second) {
        assertEquals(Integer.signum(expected), Integer.signum(first.compareTo(second)));
        assertEquals(-Integer.signum(expected), Integer.signum(second.compareTo(first)));
    }

    @Test
    void testCompare() {
        assertCompare(0, IntSequence.empty(), IntSequence.of());
        assertCompare(0, IntSequence.empty(), EmptyIntSequence.get());
        assertCompare(0, IntSequence.of(), EmptyIntSequence.get());

        assertCompare(-1, new ZeroIntSequence(2), IntSequence.of(0, 1));
        assertCompare(0, new ZeroIntSequence(2), IntSequence.of(0, 0));
        assertCompare(+1, new ZeroIntSequence(2), IntSequence.of(0));

        // Same length.
        assertCompare(-1, IntSequence.of(1, 2), IntSequence.of(2, 1));
        assertCompare(+1, IntSequence.of(2, 1), IntSequence.wrap(List.of(1, 2)));

        // Different length.
        assertCompare(-1, IntSequence.of(0), IntSequence.of(1, 2));
        assertCompare(+1, IntSequence.of(3), IntSequence.of(1, 2));
        assertCompare(+1, IntSequence.of(3, 4, 5, 6), IntSequence.of(1, 2));
    }
}
