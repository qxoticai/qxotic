package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Test;

class IteratorCombinerTest {

    @Test
    void combinesMultipleIterators() {
        Iterator<String> it =
                IteratorCombiner.of(List.of("a", "b").iterator(), List.of("c", "d").iterator());

        assertTrue(it.hasNext());
        assertEquals("a", it.next());
        assertTrue(it.hasNext());
        assertEquals("b", it.next());
        assertTrue(it.hasNext());
        assertEquals("c", it.next());
        assertTrue(it.hasNext());
        assertEquals("d", it.next());
        assertFalse(it.hasNext());
    }

    @Test
    void emptyFirstIterator() {
        Iterator<String> it =
                IteratorCombiner.of(List.<String>of().iterator(), List.of("x", "y").iterator());

        assertTrue(it.hasNext());
        assertEquals("x", it.next());
        assertEquals("y", it.next());
        assertFalse(it.hasNext());
    }

    @Test
    void allIteratorsEmpty() {
        Iterator<String> it =
                IteratorCombiner.of(List.<String>of().iterator(), List.<String>of().iterator());

        assertFalse(it.hasNext());
        assertThrows(NoSuchElementException.class, it::next);
    }

    @Test
    void singleIterator() {
        Iterator<Integer> it = IteratorCombiner.of(List.of(1, 2, 3).iterator());

        assertTrue(it.hasNext());
        assertEquals(1, it.next());
        assertEquals(2, it.next());
        assertEquals(3, it.next());
        assertFalse(it.hasNext());
    }

    @Test
    void emptyArgList() {
        @SuppressWarnings("unchecked")
        Iterator<String> it = IteratorCombiner.of();
        assertFalse(it.hasNext());
        assertThrows(NoSuchElementException.class, it::next);
    }

    @Test
    void nextAfterExhaustionThrows() {
        Iterator<String> it = IteratorCombiner.of(List.of("a").iterator(), List.of("b").iterator());
        assertEquals("a", it.next());
        assertEquals("b", it.next());
        assertFalse(it.hasNext());
        assertThrows(NoSuchElementException.class, it::next);
    }

    @Test
    void hasNextSkipsExhaustedIterators() {
        Iterator<String> it =
                IteratorCombiner.of(
                        List.of("a").iterator(),
                        List.<String>of().iterator(),
                        List.of("b").iterator());

        assertEquals("a", it.next());
        assertTrue(it.hasNext());
        assertEquals("b", it.next());
        assertFalse(it.hasNext());
    }
}
