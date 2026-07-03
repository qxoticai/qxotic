package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LFMTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
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
 *  plain {@link LFMTokenizer#encode} so conversation text can never mint control tokens. */
public final class LlamaTurnTemplate implements TurnTemplate {

    /** The template's own fallback when {@code strftime_now} is undefined — the deterministic default. */
    public static final String DEFAULT_DATE = "26 Jul 2024";

    private final LFMTokenizer tokenizer;
    private final String systemPreamble;
    private final int bos;          // <|begin_of_text|>
    private final int startHeader;  // <|start_header_id|>
    private final int endHeader;    // <|end_header_id|>
    private final int eot;          // <|eot_id|>

    public LlamaTurnTemplate(LFMTokenizer tokenizer) {
        this(tokenizer, DEFAULT_DATE);
    }

    public LlamaTurnTemplate(LFMTokenizer tokenizer, String dateString) {
        this.tokenizer = tokenizer;
        this.systemPreamble = "Cutting Knowledge Date: December 2023\nToday Date: " + dateString + "\n\n";
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.bos = required(special, "<|begin_of_text|>");
        this.startHeader = required(special, "<|start_header_id|>");
        this.endHeader = required(special, "<|end_header_id|>");
        this.eot = required(special, "<|eot_id|>");
    }

    private static int required(Map<String, Integer> special, String name) {
        Integer id = special.get(name);
        if (id == null) throw new IllegalArgumentException("tokenizer lacks " + name);
        return id;
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[]{bos}));
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        String content = text(message).strip();
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

    private static String text(Message message) {
        StringBuilder sb = new StringBuilder();
        for (Part p : message.content()) {
            if (p instanceof Part.Text t) {
                sb.append(t.text());
            } else {
                throw new IllegalArgumentException("Llama is text-only: unsupported part " + p.getClass().getSimpleName());
            }
        }
        return sb.toString();
    }
}
