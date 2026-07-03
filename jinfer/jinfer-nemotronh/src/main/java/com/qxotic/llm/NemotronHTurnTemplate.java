package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Hand-written Nemotron-H chat framing (ChatML dialect), matching the GGUF chat_template's
 *  plain-conversation shape (no tools, {@code truncate_history_thinking=true} default).
 *
 *  <p>Layout: no bos; per turn {@code <|im_start|>{role}\n{content}<|im_end|>\n}. The template
 *  ALWAYS renders a system turn - when the conversation lacks one it injects a default persona -
 *  so {@link #encode} prepends {@link #DEFAULT_SYSTEM} when the first message is not a system
 *  turn (incremental drivers supply their own system turn; both existing harnesses do).
 *
 *  <p>Historical assistant turns match the template's truncation: content with neither think
 *  marker is prefixed with an empty {@code <think></think>}; content with both keeps only the
 *  text after the LAST {@code </think>} behind the empty pair; the result is trimmed. The think
 *  pair is emitted as trusted special ids; everything else is ONE contiguous plain
 *  {@link GgufTokenizer#encode} run per span between specials (that is how a rendered template
 *  tokenizes), so conversation text cannot mint control tokens. Unclosed-{@code <think>} content
 *  (the template's "broken thought" path) is passed through plain-encoded - a documented
 *  divergence from the render+rescan oracle.
 *
 *  <p>Generation prompt: {@code <|im_start|>assistant\n<think>\n} (thinking) or
 *  {@code <|im_start|>assistant\n<think></think>} - no trailing newline - matching the template's
 *  {@code enable_thinking} branches. */
public final class NemotronHTurnTemplate implements TurnTemplate {

    public static final String DEFAULT_SYSTEM =
            "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.";
    private static final String THINK_PAIR = "<think></think>";

    private final GgufTokenizer tokenizer;
    private final int imStart;    // <|im_start|>
    private final int imEnd;      // <|im_end|>
    private final int think;      // <think>
    private final int endThink;   // </think>
    private final List<Integer> newline;        // encode("\n"), constant
    private final List<Integer> assistantNl;    // encode("assistant\n"), constant

    public NemotronHTurnTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.imStart = required(special, "<|im_start|>");
        this.imEnd = required(special, "<|im_end|>");
        this.think = required(special, "<think>");
        this.endThink = required(special, "</think>");
        this.newline = tokenizer.encode("\n");
        this.assistantNl = tokenizer.encode("assistant\n");
    }

    private static int required(Map<String, Integer> special, String name) {
        Integer id = special.get(name);
        if (id == null) throw new IllegalArgumentException("tokenizer lacks " + name);
        return id;
    }

    /** No unconditional tokens (no bos). The default-system injection lives in {@link #encode}. */
    @Override
    public List<Batch> conversationStart() {
        return List.of();
    }

    /** The whole conversation; injects the template's default system turn when absent. */
    @Override
    public List<Batch> encode(List<Message> conversation) {
        List<Batch> out = new ArrayList<>();
        if (conversation.isEmpty() || !conversation.get(0).role().equals(Role.SYSTEM)) {
            out.addAll(encodeTurn(Message.system(DEFAULT_SYSTEM)));
        }
        for (Message m : conversation) out.addAll(encodeTurn(m));
        return out;
    }

    @Override
    public List<Batch> encodeTurn(Message m) {
        List<Integer> ids = new ArrayList<>();
        ids.add(imStart);
        if (m.role().equals(Role.ASSISTANT)) {
            String c = text(m);
            if (!c.contains("<think>") && !c.contains("</think>")) c = THINK_PAIR + c;
            if (c.contains("<think>") && c.contains("</think>")) {
                c = THINK_PAIR + c.substring(c.lastIndexOf("</think>") + "</think>".length());
            }
            c = c.strip();
            ids.addAll(assistantNl);
            if (c.startsWith(THINK_PAIR)) {
                ids.add(think);
                ids.add(endThink);
                String rest = c.substring(THINK_PAIR.length());
                if (!rest.isEmpty()) ids.addAll(tokenizer.encode(rest));
            } else if (!c.isEmpty()) {
                ids.addAll(tokenizer.encode(c));    // "broken thought" passthrough, plain-encoded
            }
        } else {
            // user/system: role header + content is ONE contiguous run between the specials
            ids.addAll(tokenizer.encode(m.role().name() + "\n" + text(m)));
        }
        ids.add(imEnd);
        ids.addAll(newline);
        return List.of(Batch.prefill(arr(ids)));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        List<Integer> ids = new ArrayList<>();
        ids.add(imStart);
        ids.addAll(assistantNl);
        ids.add(think);
        if (thinking) {
            ids.addAll(newline);            // <|im_start|>assistant\n<think>\n
        } else {
            ids.add(endThink);              // <|im_start|>assistant\n<think></think>  (no newline)
        }
        return List.of(Batch.prefill(arr(ids)));
    }

    /** Closes the assistant turn: {@code <|im_end|>\n} (the stop token is never ingested). */
    @Override
    public List<Batch> closeTurn() {
        List<Integer> ids = new ArrayList<>();
        ids.add(imEnd);
        ids.addAll(newline);
        return List.of(Batch.prefill(arr(ids)));
    }

    private static String text(Message message) {
        StringBuilder sb = new StringBuilder();
        for (Part p : message.content()) {
            if (p instanceof Part.Text t) {
                sb.append(t.text());
            } else {
                throw new IllegalArgumentException("Nemotron-H is text-only: unsupported part " + p.getClass().getSimpleName());
            }
        }
        return sb.toString();
    }

    private static int[] arr(List<Integer> ids) {
        int[] a = new int[ids.size()];
        for (int i = 0; i < a.length; i++) a[i] = ids.get(i);
        return a;
    }
}
