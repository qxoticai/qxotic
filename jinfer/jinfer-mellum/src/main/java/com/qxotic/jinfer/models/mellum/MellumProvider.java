package com.qxotic.jinfer.models.mellum;

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

/** {@link ModelProvider} service for {@code general.architecture} "mellum": JetBrains Mellum 2. */
public final class MellumProvider implements ModelProvider {
    /** The model card's recommendation: temperature 0.6, top-p 0.95, top-k 20. */
    static final LoadedModel.SamplingDefaults SAMPLING_DEFAULTS =
            new LoadedModel.SamplingDefaults(0.6f, 0.95f, 20, null);

    @Override
    public Set<String> architectures() {
        return Set.of(Mellum.ARCHITECTURE);
    }

    @Override
    public LoadedModel<?> loadLanguage(
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Mellum model = Mellum.loadModel(channel, gguf, arena, tokenizer);
        Tokenizer tok = model.tokenizer();
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        String template = gguf.getStringOrDefault("tokenizer.chat_template", "");
        return new LoadedModel<>(
                model,
                tok,
                template,
                SpecialTokens.stops(tok, eos, "<|im_end|>", "<|endoftext|>"),
                Models.modelSeed(channel),
                Optional.of(new MellumChatTemplate(tok, template.contains("enable_thinking"))),
                SAMPLING_DEFAULTS);
    }
}
