package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import java.util.List;
import java.util.Map;

/**
 * Hand-written LFM2.5 chat framing (ChatML dialect), byte-exact with the GGUF's Jinja {@code
 * tokenizer.chat_template} and validated against it offline.
 *
 * <p>Layout: {@code <|startoftext|>} once, then per turn {@code
 * <|im_start|>{role}\n{content}<|im_end|>\n}, generation prompt {@code <|im_start|>assistant\n}.
 * The role header and the content are tokenized as ONE contiguous plain-encoded run — that is how
 * the rendered template tokenizes (specials force the only splits), and BPE merges across the
 * header/content boundary, so encoding them separately would drift.
 *
 * <p>Two domains: {@code <|startoftext|>}/{@code <|im_start|>}/{@code <|im_end|>} are emitted as
 * trusted ids; everything else goes through plain {@link GgufTokenizer#encode} so conversation text
 * can never mint control tokens. Matching the template, a historical assistant turn keeps only the
 * text after its last {@code </think>} (trimmed).
 *
 * <p>Out of scope, matching the model: media parts (text-only port) and tool-call rendering. Empty
 * system messages are the caller's to omit (the template drops them; a turn here would still
 * frame).
 */
public final class Lfm2TurnTemplate implements TurnTemplate {

    private final GgufTokenizer tokenizer;
    private final int bos; // <|startoftext|>
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int[] newline; // encode("\n"), constant
    private final List<Batch> generationPrompt; // <|im_start|>assistant\n, constant
    private final List<Batch> closeTurn; // <|im_end|>\n, constant

    public Lfm2TurnTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.bos = tokenizer.requiredSpecial("<|startoftext|>");
        this.imStart = tokenizer.requiredSpecial("<|im_start|>");
        this.imEnd = tokenizer.requiredSpecial("<|im_end|>");
        this.newline = ids(tokenizer.encode("\n"));
        List<Integer> gen = new java.util.ArrayList<>(List.of(imStart));
        gen.addAll(tokenizer.encode("assistant\n"));
        this.generationPrompt = List.of(Batch.prefill(gen));
        List<Integer> close = new java.util.ArrayList<>(List.of(imEnd));
        close.addAll(tokenizer.encode("\n"));
        this.closeTurn = List.of(Batch.prefill(close));
    }

    private static int[] ids(List<Integer> l) {
        int[] a = new int[l.size()];
        for (int i = 0; i < a.length; i++) a[i] = l.get(i);
        return a;
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[] {bos}));
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        String content = message.textOnly();
        if (message.role().equals(Role.ASSISTANT)) content = stripThinking(content);
        // <|im_start|> {role}\n{content} <|im_end|> \n   — one contiguous plain run per text span
        List<Integer> body = tokenizer.encode(message.role().name() + "\n" + content);
        int[] ids = new int[1 + body.size() + 1 + newline.length];
        int i = 0;
        ids[i++] = imStart;
        for (int id : body) ids[i++] = id;
        ids[i++] = imEnd;
        for (int id : newline) ids[i++] = id;
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return generationPrompt;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    /**
     * The template keeps only the text after the last {@code </think>} in historical assistant
     * turns: {@code content.split("</think>")[-1] | trim}.
     */
    private static String stripThinking(String content) {
        int at = content.lastIndexOf("</think>");
        return at < 0 ? content : content.substring(at + "</think>".length()).strip();
    }
}
