package com.qxotic.toknroll.vector;

import java.util.Arrays;

/**
 * Token ids as an array the encoder writes into directly: it reserves room, writes past the size
 * and sets the size.
 */
final class IdBuffer {
    private int[] array;
    private int size;

    IdBuffer(int capacity) {
        this.array = new int[Math.max(capacity, 8)];
    }

    /** The array, with room for at least {@code extra} ids past the size. */
    int[] reserve(int extra) {
        int needed = size + extra;
        if (needed > array.length) {
            array = Arrays.copyOf(array, Math.max(needed, array.length + (array.length >>> 1)));
        }
        return array;
    }

    void add(int id) {
        reserve(1)[size++] = id;
    }

    /** The array as it is now; ids at and past the size mean nothing. */
    int[] array() {
        return array;
    }

    int size() {
        return size;
    }

    /** Sets the size, after the ids up to it were written within a {@link #reserve}. */
    void size(int size) {
        if (size < 0 || size > array.length) {
            throw new IllegalArgumentException("size " + size + " of " + array.length);
        }
        this.size = size;
    }
}
