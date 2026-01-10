package com.qxotic.jota.memory.impl;

import sun.misc.Unsafe;

import java.lang.reflect.Field;

final class UnsafeAccess {

    private UnsafeAccess() {
        // no instances
    }

    private static final Unsafe UNSAFE = initializeUnsafe();

    static Unsafe get() {
        return UNSAFE;
    }

    private static Unsafe initializeUnsafe() {
        try {
            Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
            theUnsafe.setAccessible(true);
            return (Unsafe) theUnsafe.get(null);
        } catch (IllegalAccessException | NoSuchFieldException e) {
            throw new RuntimeException(e);
        }
    }
}
