package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.TurnTemplate;

import java.util.List;
import java.util.Map;

/** Hand-written Granite 4.1 chat framing, token-exact with the GGUF's Jinja
 *  {@code tokenizer.chat_template} for plain conversations (no tools/documents) and validated
 *  against it offline (GraniteTurnTemplateOracle).
 *
 *  <p>Layout: no bos ({@code add_bos_token} is false and the template never emits one), no default
 *  system message; every turn — system, user and assistant alike — frames as
 *  {@code <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n}; generation prompt
 *  {@code <|start_of_role|>assistant<|end_of_role|>} with nothing after it. Content is NOT
 *  trimmed (the template has no {@code | trim}) and is tokenized as one contiguous plain-encoded
 *  run between the role-close and turn-close specials, exactly as the rendered template
 *  tokenizes. The {@code thinking} flag is ignored: the vocab carries think tokens but the
 *  template has no reasoning scaffold.
 *
 *  <p>Empty system messages are the caller's to omit (the template drops them; a turn here would
 *  still frame). Two domains: the three role/turn markers are emitted as trusted ids; everything
 *  else goes through plain {@link GgufTokenizer#encode} so conversation text can never mint
 *  control tokens. */
public final class GraniteTurnTemplate implements TurnTemplate {

    private final GgufTokenizer tokenizer;
    private final int startRole;  // <|start_of_role|>
    private final int endRole;    // <|end_of_role|>
    private final int endText;    // <|end_of_text|>

    public GraniteTurnTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.startRole = required(special, "<|start_of_role|>");
        this.endRole = required(special, "<|end_of_role|>");
        this.endText = required(special, "<|end_of_text|>");
    }

    private static int required(Map<String, Integer> special, String name) {
        Integer id = special.get(name);
        if (id == null) throw new IllegalArgumentException("tokenizer lacks " + name);
        return id;
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of();                          // no bos, no preamble
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        // <|start_of_role|> {role} <|end_of_role|> {content} <|end_of_text|> \n
        List<Integer> role = tokenizer.encode(message.role().name());
        List<Integer> body = tokenizer.encode(text(message));
        List<Integer> newline = tokenizer.encode("\n");
        int[] ids = new int[1 + role.size() + 1 + body.size() + 1 + newline.size()];
        int i = 0;
        ids[i++] = startRole;
        for (int id : role) ids[i++] = id;
        ids[i++] = endRole;
        for (int id : body) ids[i++] = id;
        ids[i++] = endText;
        for (int id : newline) ids[i++] = id;
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        List<Integer> role = tokenizer.encode("assistant");
        int[] ids = new int[1 + role.size() + 1];
        int i = 0;
        ids[i++] = startRole;
        for (int id : role) ids[i++] = id;
        ids[i++] = endRole;
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> closeTurn() {
        List<Integer> newline = tokenizer.encode("\n");
        int[] ids = new int[1 + newline.size()];
        int i = 0;
        ids[i++] = endText;
        for (int id : newline) ids[i++] = id;
        return List.of(Batch.prefill(ids));
    }

    private static String text(Message message) {
        StringBuilder sb = new StringBuilder();
        for (Part p : message.content()) {
            if (p instanceof Part.Text t) {
                sb.append(t.text());
            } else {
                throw new IllegalArgumentException("Granite is text-only: unsupported part " + p.getClass().getSimpleName());
            }
        }
        return sb.toString();
    }
}
