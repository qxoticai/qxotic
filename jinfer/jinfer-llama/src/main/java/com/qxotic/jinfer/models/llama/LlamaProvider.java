package com.qxotic.jinfer.models.llama;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ChatTemplate;
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

/** {@link ModelProvider} service: the Llama-family port's arch-dispatch entry. */
public final class LlamaProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architectures().contains(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("llama", "minicpm", "mistral3", "smollm3");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Llama model = Llama.loadModel(fileChannel, gguf, arena, tokenizer);
        Tokenizer tok = model.tokenizer();
        String source = gguf.getStringOrDefault("tokenizer.chat_template", "");
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        // same-graph variants (minicpm et al.) lack the Llama 3 scaffold specials: no native
        // framing for that GGUF - chat falls back to the whole-render Jinja path, with the
        // family's reply codec attached so the fallback keeps its call grammar
        Optional<ChatTemplate> template =
                switch (gguf.getStringOrDefault("general.architecture", "")) {
                    case "minicpm" -> Optional.of(new MiniCpm5ChatTemplate(tok));
                    case "smollm3" -> Optional.of(new SmolLm3ChatTemplate(tok));
                    case "mistral3" -> Optional.of(new MistralChatTemplate(tok));
                    default ->
                            // MiniCPM5 reports general.architecture=llama; its trusted function
                            // specials are the tell
                            SpecialTokens.find(tok, "<function").isPresent()
                                            && SpecialTokens.find(tok, "</function>").isPresent()
                                    ? Optional.of(new MiniCpm5ChatTemplate(tok))
                                    : SpecialTokens.find(tok, "<|begin_of_text|>").isPresent()
                                                    && SpecialTokens.find(
                                                                    tok, "<|start_header_id|>")
                                                            .isPresent()
                                            ? Optional.of(new Llama32ChatTemplate(tok))
                                            : Optional.empty();
                };
        return new LoadedModel<>(
                model,
                tok,
                source,
                SpecialTokens.stops(
                        tok,
                        eos,
                        "<|eot_id|>",
                        // Llama 3.1+ ends a TOOL-CALL turn with <|eom_id|> instead of <|eot_id|> -
                        // without it the turn never ends and the repeated call keeps the reply
                        // from parsing at all. Absent names are skipped, so pre-3.1 checkpoints
                        // are unaffected.
                        "<|eom_id|>",
                        "<|im_end|>",
                        "<|endoftext|>",
                        "<|end_of_text|>",
                        // a reply that OPENS A TURN has ended, whatever it forgot to close with
                        // (SmolLM3 observed writing the user's next question itself). Listed
                        // LAST: the first stop is the grammar's dead-end token, and that must
                        // stay the real EOS.
                        "<|im_start|>",
                        "<|start_header_id|>"),
                Models.modelSeed(fileChannel),
                template,
                LoadedModel.SamplingDefaults.NONE);
    }
}
