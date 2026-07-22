package com.qxotic.jinfer.models.qwen35;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ChatModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the Qwen35 port's arch-dispatch entry. */
public final class Qwen35Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "qwen35".equals(architecture) || "qwen35moe".equals(architecture);
    }

    @Override
    public ChatModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength)
            throws IOException {
        return Qwen35.loadModel(fileChannel, gguf, contextLength, true).chatModel();
    }
}
