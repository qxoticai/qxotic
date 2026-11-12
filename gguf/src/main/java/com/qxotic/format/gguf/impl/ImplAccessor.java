package com.qxotic.format.gguf.impl;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import java.io.IOException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Path;

public final class ImplAccessor {
    public static Builder newBuilder() {
        return new BuilderImpl();
    }

    public static Builder newBuilder(GGUF gguf) {
        return BuilderImpl.fromExisting(gguf);
    }

    public static int defaultAlignment() {
        return ReaderImpl.ALIGNMENT_DEFAULT_VALUE;
    }

    public static String alignmentKey() {
        return ReaderImpl.ALIGNMENT_KEY;
    }

    public static GGUF read(ReadableByteChannel byteChannel) throws IOException {
        return new ReaderImpl().readImpl(byteChannel);
    }

    public static void write(GGUF gguf, WritableByteChannel byteChannel) throws IOException {
        WriterImpl.write(gguf, byteChannel);
    }

    public static void write(GGUF gguf, Path path) throws IOException {
        WriterImpl.write(gguf, path);
    }

    public static String toString(GGUF gguf, boolean showKeys, boolean showTensors) {
        return GGUFFormatter.toString(gguf, showKeys, showTensors);
    }

    public static String toString(
            GGUF gguf,
            boolean showKeys,
            boolean showTensors,
            int maxArrayElements,
            int maxStringLength) {
        return GGUFFormatter.toString(
                gguf, showKeys, showTensors, maxArrayElements, maxStringLength);
    }
}
