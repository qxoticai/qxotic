package com.llm4j.gguf.impl;

import com.llm4j.gguf.Builder;
import com.llm4j.gguf.GGUF;
import java.io.IOException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;

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
        WriterImpl.writeImpl(gguf, byteChannel);
    }
}
