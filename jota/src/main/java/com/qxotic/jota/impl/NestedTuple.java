package com.qxotic.jota.impl;

public interface NestedTuple<T extends NestedTuple<T>> {

    int rank();

    T modeAt(int _modeIndex);

    int flatRank();

    long flatAt(int _flatIndex);

    default boolean isFlat() {
        return rank() == flatRank();
    }

    default boolean isScalar() {
        return rank() == 0;
    }

    T flatten();

    boolean isCongruentWith(NestedTuple<?> other);

    T replace(int _modeIndex, T newMode);

    T insert(int _modeIndex, T mode);

    T remove(int _modeIndex);

    T permute(int... _modeIndices);

    String toString();

    long[] toArray();
}
