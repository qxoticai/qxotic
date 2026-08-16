package com.qxotic.jinfer.models.llama;

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

/** {@link ModelProvider} service: the Granite port's arch-dispatch entry. */
public final class GraniteProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "granite".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("granite");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Granite model = Granite.loadModel(fileChannel, gguf, arena, tokenizer);
        Tokenizer tok = model.tokenizer();
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        return new LoadedModel<>(
                model,
                tok,
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                SpecialTokens.stops(
                        tok, eos, "<|end_of_text|>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"),
                Models.modelSeed(fileChannel),
                // the GGUF template frames; the attached codec keeps Granite's reply grammar
                // (call parsing, constrained decoding, forced calls) on the whole-render path
                Optional.of(new GraniteChatTemplate(tok)),
                LoadedModel.SamplingDefaults.NONE);
    }
}
