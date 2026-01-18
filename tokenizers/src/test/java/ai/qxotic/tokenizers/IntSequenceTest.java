package ai.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
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
    void testWrapArray() {
        int[] array = {1, 2, 3};
        IntSequence sequence = IntSequence.wrap(array);
        assertArrayEquals(array, sequence.toArray());
    }

    @Test
    void testWrapList() {
        List<Integer> list = Arrays.asList(1, 2, 3);
        IntSequence sequence = IntSequence.wrap(list);
        assertEquals(list, sequence.toList());
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
