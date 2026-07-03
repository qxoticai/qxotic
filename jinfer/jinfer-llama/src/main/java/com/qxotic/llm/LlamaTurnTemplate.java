package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;

import java.util.List;
import java.util.Map;

/** Hand-written Llama 3 chat framing, token-exact with the GGUF's Jinja
 *  {@code tokenizer.chat_template} for plain conversations (no tools) and validated against it
 *  offline (LlamaTurnTemplateOracle).
 *
 *  <p>Layout: {@code <|begin_of_text|>} once, then per turn
 *  {@code <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>} (no separator between
 *  turns), generation prompt {@code <|start_header_id|>assistant<|end_header_id|>\n\n}. The
 *  {@code \n\n} and the content are tokenized as ONE contiguous plain-encoded run — that is how
 *  the rendered template tokenizes (specials force the only splits), and BPE merges across the
 *  boundary, so encoding them separately would drift. Content is trimmed, matching the template's
 *  {@code | trim}.
 *
 *  <p>The template hoists an implicit system block (knowledge-cutoff + date preamble) even when
 *  the conversation has none; here that maps to an EXPLICIT system turn — callers open every
 *  conversation with {@code Message.system(...)} (empty text is fine), and {@link #encodeTurn}
 *  prepends the same preamble to system content. The date is fixed at construction (defaults to
 *  the template's own fallback) so the token stream — and therefore the prompt cache — stays
 *  deterministic across runs.
 *
 *  <p>Two domains: header/turn markers are emitted as trusted ids; everything else goes through
 *  plain {@link GgufTokenizer#encode} so conversation text can never mint control tokens. */
public final class LlamaTurnTemplate implements TurnTemplate {

    /** The template's own fallback when {@code strftime_now} is undefined — the deterministic default. */
    public static final String DEFAULT_DATE = "26 Jul 2024";

    private final GgufTokenizer tokenizer;
    private final String systemPreamble;
    private final int bos;          // <|begin_of_text|>
    private final int startHeader;  // <|start_header_id|>
    private final int endHeader;    // <|end_header_id|>
    private final int eot;          // <|eot_id|>

    public LlamaTurnTemplate(GgufTokenizer tokenizer) {
        this(tokenizer, DEFAULT_DATE);
    }

    public LlamaTurnTemplate(GgufTokenizer tokenizer, String dateString) {
        this.tokenizer = tokenizer;
        this.systemPreamble = "Cutting Knowledge Date: December 2023\nToday Date: " + dateString + "\n\n";
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.bos = tokenizer.requiredSpecial("<|begin_of_text|>");
        this.startHeader = tokenizer.requiredSpecial("<|start_header_id|>");
        this.endHeader = tokenizer.requiredSpecial("<|end_header_id|>");
        this.eot = tokenizer.requiredSpecial("<|eot_id|>");
    }


    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[]{bos}));
    }

    /** The template renders its system preamble even for conversations with no system message:
     *  inject an empty system turn when absent, so turn-by-turn callers frame identically. */
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
        List<Integer> role = tokenizer.encode(message.role().name());
        List<Integer> body = tokenizer.encode("\n\n" + content);
        int[] ids = new int[1 + role.size() + 1 + body.size() + 1];
        int i = 0;
        ids[i++] = startHeader;
        for (int id : role) ids[i++] = id;
        ids[i++] = endHeader;
        for (int id : body) ids[i++] = id;
        ids[i++] = eot;
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        List<Integer> role = tokenizer.encode("assistant");
        List<Integer> sep = tokenizer.encode("\n\n");
        int[] ids = new int[1 + role.size() + 1 + sep.size()];
        int i = 0;
        ids[i++] = startHeader;
        for (int id : role) ids[i++] = id;
        ids[i++] = endHeader;
        for (int id : sep) ids[i++] = id;
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> closeTurn() {
        return List.of(Batch.prefill(new int[]{eot}));
    }

}
