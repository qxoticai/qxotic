package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ChatModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the NemotronH port's arch-dispatch entry. */
public final class NemotronHProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "nemotron_h".equals(architecture) || "nemotron_h_moe".equals(architecture);
    }

    @Override
    public ChatModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength)
            throws IOException {
        return NemotronH.loadModel(fileChannel, gguf, contextLength).chatModel();
    }
}
