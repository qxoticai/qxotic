package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import java.util.ArrayList;
import java.util.List;

/**
 * Hand-written Nemotron-H chat framing (ChatML dialect), matching the GGUF chat_template's
 * plain-conversation shape (no tools, {@code truncate_history_thinking=true} default).
 *
 * <p>Layout: no bos; per turn {@code <|im_start|>{role}\n{content}<|im_end|>\n}. The template
 * ALWAYS renders a system turn - when the conversation lacks one it injects a default persona - so
 * {@link #encode} prepends {@link #DEFAULT_SYSTEM} when the first message is not a system turn
 * (incremental drivers supply their own system turn; both existing harnesses do).
 *
 * <p>Historical assistant turns match the template's truncation: content with neither think marker
 * is prefixed with an empty {@code <think></think>}; content with both keeps only the text after
 * the LAST {@code </think>} behind the empty pair; the result is trimmed. The think pair is emitted
 * as trusted special ids; everything else is ONE contiguous plain {@link GgufTokenizer#encode} run
 * per span between specials (that is how a rendered template tokenizes), so conversation text
 * cannot mint control tokens. Unclosed-{@code <think>} content (the template's "broken thought"
 * path) is passed through plain-encoded - a documented divergence from the render+rescan oracle.
 *
 * <p>Generation prompt: {@code <|im_start|>assistant\n<think>\n} (thinking) or {@code
 * <|im_start|>assistant\n<think></think>} - no trailing newline - matching the template's {@code
 * enable_thinking} branches.
 */
public final class NemotronHTurnTemplate implements TurnTemplate {

    public static final String DEFAULT_SYSTEM =
            "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.";

    private final GgufTokenizer tokenizer;
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    private final List<Integer> newline; // encode("\n"), constant
    private final List<Integer> assistantNl; // encode("assistant\n"), constant
    private final List<Batch> genThinking, genDirect; // generation prompts, encoded once
    private final List<Batch> closeTurn; // <|im_end|>\n, constant

    public NemotronHTurnTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.imStart = tokenizer.requiredSpecial("<|im_start|>");
        this.imEnd = tokenizer.requiredSpecial("<|im_end|>");
        this.think = tokenizer.requiredSpecial("<think>");
        this.endThink = tokenizer.requiredSpecial("</think>");
        this.newline = tokenizer.encode("\n");
        this.assistantNl = tokenizer.encode("assistant\n");
        List<Integer> head = new ArrayList<>(List.of(imStart));
        head.addAll(assistantNl);
        head.add(think);
        List<Integer> thinking = new ArrayList<>(head);
        thinking.addAll(newline); // <|im_start|>assistant\n<think>\n
        this.genThinking = List.of(Batch.prefill(thinking));
        List<Integer> direct = new ArrayList<>(head);
        direct.add(endThink); // <|im_start|>assistant\n<think></think>  (no newline)
        this.genDirect = List.of(Batch.prefill(direct));
        List<Integer> close = new ArrayList<>(List.of(imEnd));
        close.addAll(newline);
        this.closeTurn = List.of(Batch.prefill(close));
    }

    /**
     * No unconditional tokens (no bos). The default-system injection lives in {@link #normalize}.
     */
    @Override
    public List<Batch> conversationStart() {
        return List.of();
    }

    /**
     * The template unconditionally renders a system turn: inject the default when absent, so every
     * caller (whole-render AND turn-by-turn drivers) frames identically.
     */
    @Override
    public List<Message> normalize(List<Message> conversation) {
        if (!conversation.isEmpty() && conversation.get(0).role().equals(Role.SYSTEM)) {
            return conversation;
        }
        List<Message> out = new ArrayList<>(conversation.size() + 1);
        out.add(Message.system(DEFAULT_SYSTEM));
        out.addAll(conversation);
        return out;
    }

    @Override
    public List<Batch> encodeTurn(Message m) {
        List<Integer> ids = new ArrayList<>();
        ids.add(imStart);
        if (m.role().equals(Role.ASSISTANT)) {
            String c = m.textOnly();
            int lastClose = c.lastIndexOf("</think>");
            ids.addAll(assistantNl);
            if (c.contains("<think>") == (lastClose >= 0)) {
                // framed (both markers -> keep the tail after the last </think>) or plain (no
                // markers -> the whole content): emit the empty pair, then the text with its
                // trailing whitespace stripped (leading whitespace survives, as in the template)
                String rest =
                        (lastClose >= 0 ? c.substring(lastClose + "</think>".length()) : c)
                                .stripTrailing();
                ids.add(think);
                ids.add(endThink);
                if (!rest.isEmpty()) ids.addAll(tokenizer.encode(rest));
            } else {
                // "broken thought" (unpaired marker): plain-encoded passthrough, fully stripped
                String rest = c.strip();
                if (!rest.isEmpty()) ids.addAll(tokenizer.encode(rest));
            }
        } else {
            // user/system: role header + content is ONE contiguous run between the specials
            ids.addAll(tokenizer.encode(m.role().name() + "\n" + m.textOnly()));
        }
        ids.add(imEnd);
        ids.addAll(newline);
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return thinking ? genThinking : genDirect;
    }

    /** Closes the assistant turn: {@code <|im_end|>\n} (the stop token is never ingested). */
    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }
}
