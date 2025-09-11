package com.llm4j.jota.memory;

public interface ScopedMemory<B> extends Memory<B>, AutoCloseable {
    @Override
    void close();
}
