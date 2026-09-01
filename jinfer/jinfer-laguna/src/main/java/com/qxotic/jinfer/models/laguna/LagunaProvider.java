package com.qxotic.jinfer.models.laguna;

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

/** Service-provider entry for Poolside Laguna XS 2.1 GGUF models. */
public final class LagunaProvider implements ModelProvider {
    @Override
    public boolean supports(String architecture) {
        return "laguna".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("laguna");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel channel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Laguna model = Laguna.loadModel(channel, gguf, arena, tokenizer);
        int bos = gguf.getValue(int.class, "tokenizer.ggml.bos_token_id");
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        int eot = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eot_token_id", -1);
        Set<Integer> stops = SpecialTokens.stops(model.tokenizer(), eos, "</assistant>");
        if (eot >= 0 && !stops.contains(eot)) {
            java.util.LinkedHashSet<Integer> expanded = new java.util.LinkedHashSet<>(stops);
            expanded.add(eot);
            stops = expanded;
        }
        return new LoadedModel<>(
                model,
                model.tokenizer(),
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                stops,
                Models.modelSeed(channel),
                Optional.of(new LagunaChatTemplate(model.tokenizer(), bos)),
                new LoadedModel.SamplingDefaults(
                        gguf.getValueOrDefault(float.class, "general.sampling.temp", 1f),
                        gguf.getValueOrDefault(float.class, "general.sampling.top_p", 1f),
                        gguf.getValueOrDefault(int.class, "general.sampling.top_k", 20),
                        gguf.getValueOrDefault(float.class, "general.sampling.min_p", 0f)));
    }
}
