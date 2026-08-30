package com.qxotic.jinfer.models.bailingmoe3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Service-provider entry for BailingMoe3 GGUF models. */
public final class BailingMoe3Provider implements ModelProvider {
    @Override
    public boolean supports(String architecture) {
        return "bailingmoe3".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("bailingmoe3");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel channel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        BailingMoe3 model = BailingMoe3.loadModel(channel, gguf, arena, tokenizer);
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        return new LoadedModel<>(
                model,
                model.tokenizer(),
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                SpecialTokens.stops(model.tokenizer(), eos, "<|role_end|>", "<|endoftext|>"),
                Models.modelSeed(channel),
                Optional.of(new BailingMoe3ChatTemplate(model.tokenizer())),
                LoadedModel.SamplingDefaults.NONE);
    }
}
