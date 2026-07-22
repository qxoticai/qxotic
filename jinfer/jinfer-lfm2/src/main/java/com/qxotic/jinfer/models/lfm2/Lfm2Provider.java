package com.qxotic.jinfer.models.lfm2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ChatModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the Lfm2 port's arch-dispatch entry. */
public final class Lfm2Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architecture.startsWith("lfm");
    }

    @Override
    public ChatModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength)
            throws IOException {
        return Lfm2.loadModel(fileChannel, gguf, contextLength, true).chatModel();
    }
}
