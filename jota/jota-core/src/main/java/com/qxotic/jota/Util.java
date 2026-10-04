package com.qxotic.jota;

/**
 * Internal helper shared by jota modules. Not part of the supported API: it stays public only
 * because it shipped in 0.2.0, and it may change or disappear in any release. Do not use it.
 */
public final class Util {
    /**
     * Resolves a Python-style index: a negative {@code _index} counts back from {@code size}.
     *
     * @throws IllegalArgumentException if the resolved index is outside {@code [0, size)}
     */
    public static int wrapAround(int _index, int size) {
        assert size >= 0;
        int index = _index >= 0 ? _index : _index + size;
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException("wrap-around index out of bounds");
        }
        return index;
    }
}
