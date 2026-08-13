package com.qxotic.jinfer.x.models.gptoss;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.chat.ModelProvider;
import com.qxotic.jinfer.x.chat.Models;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** {@link ModelProvider} service: the GPT-OSS port's arch-dispatch entry. */
public final class GptOssProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "gpt-oss".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("gpt-oss");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        GptOss model = GptOss.loadModel(fileChannel, gguf, arena, tokenizer);
        Tokenizer tok = model.tokenizer();
        return new LoadedModel<>(
                model,
                tok,
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                SpecialTokens.stops(tok, -1, "<|return|>", "<|call|>", "<|endoftext|>"),
                Models.modelSeed(fileChannel),
                // the GGUF template frames; the attached Harmony codec keeps the reply grammar
                // (call parsing, constrained decoding, forced calls) on the whole-render path
                Optional.of(new GptOssChatTemplate(tok)),
                // OpenAI's recommended sampling for GPT-OSS: near-deterministic nucleus
                new LoadedModel.SamplingDefaults(1.0f, 1.0f, null, null));
    }
}
