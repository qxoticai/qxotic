package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import java.io.IOException;
import java.nio.channels.FileChannel;

/**
 * One port's entry in the architecture dispatch: a {@link java.util.ServiceLoader} service each
 * port module registers (META-INF/services), so {@link Models#load} finds exactly the ports on the
 * classpath - no hand-maintained arch table in every consumer.
 */
public interface ModelProvider {

    /** Whether this port loads GGUFs with the given {@code general.architecture}. */
    boolean supports(String architecture);

    /**
     * Loads the model from an already-parsed GGUF; {@code fileChannel} supplies the tensor data.
     * {@code contextLength} -1 means the model's full context.
     */
    ChatModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength) throws IOException;
}
