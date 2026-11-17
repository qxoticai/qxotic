package com.qxotic.jam.libjam;

/** A child JVM's view of the native backend, for {@link NativeLoaderTest}. */
public final class NativeLoaderProbe {
    public static void main(String[] args) {
        System.out.println("available=" + new NativeJAMProvider().isAvailable());
    }
}
