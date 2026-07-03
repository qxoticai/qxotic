package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LFMTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;

import java.util.List;
import java.util.Map;

/** Hand-written Gemma 4 chat framing, matching the GGUF chat_template's plain-conversation shape
 *  and the hand-built prompt precedent (Gemma4VisionRun/Gemma4AudioRun).
 *
 *  <p>Layout: {@code <bos>} once, then per turn {@code <|turn>{role}\n{content}<turn|>\n},
 *  generation prompt {@code <|turn>model\n}. Gemma's assistant role name is {@code model}
 *  ({@link Role#ASSISTANT} maps to it). The role header and the content are tokenized as ONE
 *  contiguous plain-encoded run — that is how a rendered template tokenizes (specials force the
 *  only splits), and BPE merges across the header/content boundary.
 *
 *  <p>Two domains: {@code <bos>}/{@code <|turn>}/{@code <turn|>} are emitted as trusted ids;
 *  everything else goes through plain {@link LFMTokenizer#encode} so conversation text can never
 *  mint control tokens. Text-only for now — media parts land with the multimodal wiring. */
public final class Gemma4TurnTemplate implements TurnTemplate {

    private final LFMTokenizer tokenizer;
    private final int bos;        // <bos>
    private final int turnOpen;   // <|turn>
    private final int turnClose;  // <turn|>

    public Gemma4TurnTemplate(LFMTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.bos = required(special, "<bos>");
        this.turnOpen = required(special, "<|turn>");
        this.turnClose = required(special, "<turn|>");
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
        // <|turn> {role}\n{content} <turn|> \n   — one contiguous plain run per text span
        List<Integer> body = tokenizer.encode(roleName(message.role()) + "\n" + text(message));
        List<Integer> newline = tokenizer.encode("\n");
        int[] ids = new int[1 + body.size() + 1 + newline.size()];
        int i = 0;
        ids[i++] = turnOpen;
        for (int id : body) ids[i++] = id;
        ids[i++] = turnClose;
        for (int id : newline) ids[i++] = id;
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        List<Integer> body = tokenizer.encode("model\n");
        int[] ids = new int[1 + body.size()];
        int i = 0;
        ids[i++] = turnOpen;
        for (int id : body) ids[i++] = id;
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> closeTurn() {
        List<Integer> newline = tokenizer.encode("\n");
        int[] ids = new int[1 + newline.size()];
        int i = 0;
        ids[i++] = turnClose;
        for (int id : newline) ids[i++] = id;
        return List.of(Batch.prefill(ids));
    }

    /** Gemma's template names the assistant turn {@code model}. */
    private static String roleName(Role role) {
        return role.equals(Role.ASSISTANT) ? "model" : role.name();
    }

    private static String text(Message message) {
        StringBuilder sb = new StringBuilder();
        for (Part p : message.content()) {
            if (p instanceof Part.Text t) {
                sb.append(t.text());
            } else {
                throw new IllegalArgumentException("Gemma4TurnTemplate is text-only for now: " + p.getClass().getSimpleName());
            }
        }
        return sb.toString();
    }
}
