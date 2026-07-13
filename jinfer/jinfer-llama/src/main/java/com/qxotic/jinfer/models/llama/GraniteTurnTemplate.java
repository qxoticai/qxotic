package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.GgufTokenizer;
import com.qxotic.toknroll.IntSequence;
import java.util.List;

/**
 * Hand-written Granite 4.1 chat framing, token-exact with the GGUF's Jinja {@code
 * tokenizer.chat_template} for plain conversations (no tools/documents) and validated against it
 * offline (GraniteTurnTemplateOracle).
 *
 * <p>Layout: no bos ({@code add_bos_token} is false and the template never emits one), no default
 * system message; every turn — system, user and assistant alike — frames as {@code
 * <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n}; generation prompt {@code
 * <|start_of_role|>assistant<|end_of_role|>} with nothing after it. Content is NOT trimmed (the
 * template has no {@code | trim}) and is tokenized as one contiguous plain-encoded run between the
 * role-close and turn-close specials, exactly as the rendered template tokenizes. The {@code
 * thinking} flag is ignored: the vocab carries think tokens but the template has no reasoning
 * scaffold.
 *
 * <p>Empty system messages are the caller's to omit (the template drops them; a turn here would
 * still frame). Two domains: the three role/turn markers are emitted as trusted ids; everything
 * else goes through plain {@link GgufTokenizer#encode} so conversation text can never mint control
 * tokens.
 */
public final class GraniteTurnTemplate implements TurnTemplate {

    private final GgufTokenizer tokenizer;
    private final int startRole; // <|start_of_role|>
    private final int endRole; // <|end_of_role|>
    private final int endText; // <|end_of_text|>
    private final IntSequence newline; // encode("\n"), constant
    private final List<Batch>
            generationPrompt; // <|start_of_role|>assistant<|end_of_role|>, constant
    private final List<Batch> closeTurn; // <|end_of_text|>\n, constant

    public GraniteTurnTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.startRole = tokenizer.requiredSpecial("<|start_of_role|>");
        this.endRole = tokenizer.requiredSpecial("<|end_of_role|>");
        this.endText = tokenizer.requiredSpecial("<|end_of_text|>");
        this.newline = tokenizer.encode("\n");
        IntSequence gen =
                IntSequence.of(startRole)
                        .concat(tokenizer.encode("assistant"))
                        .concat(IntSequence.of(endRole));
        this.generationPrompt = List.of(Batch.prefill(gen.toArray()));
        IntSequence close = IntSequence.of(endText).concat(newline);
        this.closeTurn = List.of(Batch.prefill(close.toArray()));
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(); // no bos, no preamble
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        // <|start_of_role|> {role} <|end_of_role|> {content} <|end_of_text|> \n
        IntSequence ids =
                IntSequence.of(startRole)
                        .concat(tokenizer.encode(message.role().name()))
                        .concat(IntSequence.of(endRole))
                        .concat(tokenizer.encode(message.textOnly()))
                        .concat(IntSequence.of(endText))
                        .concat(newline);
        return List.of(Batch.prefill(ids.toArray()));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return generationPrompt;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }
}
