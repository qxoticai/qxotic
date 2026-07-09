package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import java.util.ArrayList;
import java.util.List;

/**
 * Hand-written Qwen3.5 chat framing (ChatML dialect), token-exact with the GGUF's Jinja
 * chat_template over plain conversations.
 *
 * <p>Layout: NO bos (Qwen3.5 has none; {@link #conversationStart} is empty), per turn {@code
 * <|im_start|>{role}\n{content|trim}<|im_end|>\n}. Matching the template, every turn's content is
 * trimmed, and a historical assistant turn keeps only the text after its last {@code </think>}
 * (leading newlines stripped) - the template's frozen middle-turn form; a trailing assistant turn
 * after the final user query renders differently (thinking kept) and is out of scope for
 * turn-stable encoding, as in the other curated templates.
 *
 * <p>Generation prompt: {@code <|im_start|>assistant\n} then the thinking scaffold - {@code
 * <think>\n} to reason, or the pre-closed {@code <think>\n\n</think>\n\n} to answer directly. The
 * 2B template defaults to NON-thinking ({@code enable_thinking} must be defined and true to
 * reason); note the 35B-A3B template INVERTS that default (thinking unless {@code enable_thinking}
 * is defined and false) - the scaffolds themselves are identical, only the default flag differs, so
 * this template serves both.
 *
 * <p>Each text run between specials is ONE contiguous plain {@link GgufTokenizer#encode};
 * conversation content never goes through special-aware encoding, so text cannot mint control
 * tokens ({@code <think>}/{@code </think>} in the scaffold are emitted as trusted ids).
 */
public final class Qwen35TurnTemplate implements TurnTemplate {

    private final GgufTokenizer tokenizer;
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    private final List<Integer> newline; // encode("\n"), constant
    private final List<Batch> genThinking, genDirect; // generation prompts, encoded once
    private final List<Batch> closeTurn; // <|im_end|>\n, constant

    public Qwen35TurnTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.imStart = tokenizer.requiredSpecial("<|im_start|>");
        this.imEnd = tokenizer.requiredSpecial("<|im_end|>");
        this.think = tokenizer.requiredSpecial("<think>");
        this.endThink = tokenizer.requiredSpecial("</think>");
        this.newline = tokenizer.encode("\n");
        // <|im_start|>assistant\n<think>\n            (reasoning)
        // <|im_start|>assistant\n<think>\n\n</think>\n\n   (direct answer)
        List<Integer> head = new ArrayList<>();
        head.add(imStart);
        head.addAll(tokenizer.encode("assistant\n"));
        List<Integer> thinking = new ArrayList<>(head);
        thinking.add(think);
        thinking.addAll(tokenizer.encode("\n"));
        this.genThinking = List.of(Batch.prefill(thinking));
        List<Integer> direct = new ArrayList<>(head);
        direct.add(think);
        direct.addAll(tokenizer.encode("\n\n"));
        direct.add(endThink);
        direct.addAll(tokenizer.encode("\n\n"));
        this.genDirect = List.of(Batch.prefill(direct));
        List<Integer> close = new ArrayList<>(List.of(imEnd));
        close.addAll(newline);
        this.closeTurn = List.of(Batch.prefill(close));
    }

    /** Qwen3.5 emits no bos and no fixed preamble. */
    @Override
    public List<Batch> conversationStart() {
        return List.of();
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        String content = message.textOnly().strip(); // template: content|trim
        if (message.role().equals(Role.ASSISTANT)) content = stripThinking(content);
        List<Integer> ids = new ArrayList<>();
        ids.add(imStart);
        ids.addAll(tokenizer.encode(message.role().name() + "\n" + content)); // one contiguous run
        ids.add(imEnd);
        ids.addAll(newline);
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return thinking ? genThinking : genDirect;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    /**
     * The template keeps only the text after the last {@code </think>}, leading newlines stripped:
     * {@code content.split('</think>')[-1].lstrip('\n')}.
     */
    private static String stripThinking(String content) {
        int at = content.lastIndexOf("</think>");
        if (at < 0) return content;
        String tail = content.substring(at + "</think>".length());
        int i = 0;
        while (i < tail.length() && tail.charAt(i) == '\n') i++;
        return tail.substring(i);
    }
}
