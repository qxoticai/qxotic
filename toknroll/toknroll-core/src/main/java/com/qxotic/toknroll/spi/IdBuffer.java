package com.qxotic.toknroll.spi;

import java.util.Arrays;

/**
 * Token ids as an array an encoder writes into directly: it reserves room, writes past the size and
 * sets the size. Core lends a builder's own array this way and takes it back, so ids are written
 * once.
 */
public final class IdBuffer {
    private int[] array;
    private int size;

    public IdBuffer(int capacity) {
        this(new int[Math.max(capacity, 8)], 0);
    }

    /** Takes {@code array} over, with its first {@code size} ids in use. */
    public IdBuffer(int[] array, int size) {
        if (size < 0 || size > array.length) {
            throw new IllegalArgumentException("size " + size + " of " + array.length);
        }
        this.array = array;
        this.size = size;
    }

    /** The array, with room for at least {@code extra} ids past the size. */
    public int[] reserve(int extra) {
        int needed = size + extra;
        if (needed > array.length) {
            array = Arrays.copyOf(array, Math.max(needed, array.length + (array.length >>> 1)));
        }
        return array;
    }

    public void add(int id) {
        reserve(1)[size++] = id;
    }

    /** The array as it is now; ids at and past the size mean nothing. */
    public int[] array() {
        return array;
    }

    public int size() {
        return size;
    }

    /** Sets the size, after the ids up to it were written within a {@link #reserve}. */
    public void size(int size) {
        if (size < 0 || size > array.length) {
            throw new IllegalArgumentException("size " + size + " of " + array.length);
        }
        this.size = size;
    }
}
