package ai.qxotic.jota.memory;

public interface ScopedMemory<B> extends Memory<B>, AutoCloseable {
    @Override
    void close();
}
