package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;

/**
 * Hand-written Llama 3 chat framing, token-exact with the GGUF's Jinja {@code
 * tokenizer.chat_template} for plain conversations (no tools) and validated against it offline
 * (LlamaTurnTemplateOracle).
 *
 * <p>Layout: {@code <|begin_of_text|>} once, then per turn {@code
 * <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>} (no separator between turns),
 * generation prompt {@code <|start_header_id|>assistant<|end_header_id|>\n\n}. The {@code \n\n} and
 * the content are tokenized as ONE contiguous plain-encoded run — that is how the rendered template
 * tokenizes (specials force the only splits), and BPE merges across the boundary, so encoding them
 * separately would drift. Content is trimmed, matching the template's {@code | trim}.
 *
 * <p>The template hoists an implicit system block (knowledge-cutoff + date preamble) even when the
 * conversation has none; here that maps to an EXPLICIT system turn — callers open every
 * conversation with {@code Message.system(...)} (empty text is fine), and {@link #encodeTurn}
 * prepends the same preamble to system content. The date is fixed at construction (defaults to the
 * template's own fallback) so the token stream — and therefore the prompt cache — stays
 * deterministic across runs.
 *
 * <p>Two domains: header/turn markers are emitted as trusted ids; everything else goes through
 * plain {@link Tokenizer#encode} so conversation text can never mint control tokens.
 */
public final class LlamaTurnTemplate implements TurnTemplate {

    /**
     * The template's own fallback when {@code strftime_now} is undefined — the deterministic
     * default.
     */
    public static final String DEFAULT_DATE = "26 Jul 2024";

    private final Tokenizer tokenizer;
    private final String systemPreamble;
    private final int bos; // <|begin_of_text|>
    private final int startHeader; // <|start_header_id|>
    private final int endHeader; // <|end_header_id|>
    private final int eot; // <|eot_id|>

    public LlamaTurnTemplate(Tokenizer tokenizer) {
        this(tokenizer, DEFAULT_DATE);
    }

    public LlamaTurnTemplate(Tokenizer tokenizer, String dateString) {
        this.tokenizer = tokenizer;
        this.systemPreamble =
                "Cutting Knowledge Date: December 2023\nToday Date: " + dateString + "\n\n";
        this.bos = SpecialTokens.require(tokenizer, "<|begin_of_text|>");
        this.startHeader = SpecialTokens.require(tokenizer, "<|start_header_id|>");
        this.endHeader = SpecialTokens.require(tokenizer, "<|end_header_id|>");
        this.eot = SpecialTokens.require(tokenizer, "<|eot_id|>");
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[] {bos}));
    }

    /**
     * The template renders its system preamble even for conversations with no system message:
     * inject an empty system turn when absent, so turn-by-turn callers frame identically.
     */
    @Override
    public List<Message> normalize(List<Message> conversation) {
        if (!conversation.isEmpty() && conversation.get(0).role().equals(Role.SYSTEM)) {
            return conversation;
        }
        List<Message> out = new java.util.ArrayList<>(conversation.size() + 1);
        out.add(Message.system(""));
        out.addAll(conversation);
        return out;
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        String content = message.textOnly().strip();
        if (message.role().equals(Role.SYSTEM)) content = systemPreamble + content;
        // <|start_header_id|> {role} <|end_header_id|> \n\n{content} <|eot_id|>
        IntSequence ids =
                IntSequence.of(startHeader)
                        .concat(tokenizer.encode(message.role().name()))
                        .concat(IntSequence.of(endHeader))
                        .concat(tokenizer.encode("\n\n" + content))
                        .concat(IntSequence.of(eot));
        return List.of(Batch.prefill(ids.toArray()));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        IntSequence ids =
                IntSequence.of(startHeader)
                        .concat(tokenizer.encode("assistant"))
                        .concat(IntSequence.of(endHeader))
                        .concat(tokenizer.encode("\n\n"));
        return List.of(Batch.prefill(ids.toArray()));
    }

    @Override
    public List<Batch> closeTurn() {
        return List.of(Batch.prefill(new int[] {eot}));
    }

    @Override
    public ReplyParser parser() {
        return ReplyParser.spans(tokenizer);
    }
}
