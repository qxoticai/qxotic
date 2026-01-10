package com.qxotic.jota.memory.impl;

final class Util {
    static boolean isPowerOf2(long n) {
        return n > 0 && ((n & (n - 1)) == 0);
    }
}
