package com.qxotic.jinfer.models.maple;

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

/** Model-provider service for Maple GGUFs. */
public final class MapleProvider implements ModelProvider {
    @Override
    public boolean supports(String architecture) {
        return "maple".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("maple");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Maple model = Maple.loadModel(fileChannel, gguf, arena, tokenizer);
        Tokenizer tok = model.tokenizer();
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        return new LoadedModel<>(
                model,
                tok,
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                SpecialTokens.stops(tok, eos, "<|im_end|>", "<|endoftext|>"),
                Models.modelSeed(fileChannel),
                Optional.of(new MapleChatTemplate(tok)),
                LoadedModel.SamplingDefaults.NONE);
    }
}
