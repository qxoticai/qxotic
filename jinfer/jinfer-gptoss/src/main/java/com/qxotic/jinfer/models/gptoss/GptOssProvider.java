package com.qxotic.jinfer.models.gptoss;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ChatModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the GptOss port's arch-dispatch entry. */
public final class GptOssProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "gpt-oss".equals(architecture);
    }

    @Override
    public ChatModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength)
            throws IOException {
        return GptOss.loadModel(fileChannel, gguf, contextLength, true).chatModel();
    }
}
