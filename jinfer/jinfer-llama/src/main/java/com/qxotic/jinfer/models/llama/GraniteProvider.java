package com.qxotic.jinfer.models.llama;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ChatModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the Granite port's arch-dispatch entry. */
public final class GraniteProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "granite".equals(architecture);
    }

    @Override
    public ChatModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength)
            throws IOException {
        return Granite.loadModel(fileChannel, gguf, contextLength, true).chatModel();
    }
}
