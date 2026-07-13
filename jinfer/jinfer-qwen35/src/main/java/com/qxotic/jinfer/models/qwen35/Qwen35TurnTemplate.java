package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.GgufTokenizer;
import com.qxotic.toknroll.IntSequence;
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
    private final IntSequence newline; // encode("\n"), constant
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
        IntSequence head = IntSequence.of(imStart).concat(tokenizer.encode("assistant\n"));
        IntSequence thinking = head.concat(IntSequence.of(think)).concat(tokenizer.encode("\n"));
        this.genThinking = List.of(Batch.prefill(thinking.toArray()));
        IntSequence direct =
                head.concat(IntSequence.of(think))
                        .concat(tokenizer.encode("\n\n"))
                        .concat(IntSequence.of(endThink))
                        .concat(tokenizer.encode("\n\n"));
        this.genDirect = List.of(Batch.prefill(direct.toArray()));
        IntSequence close = IntSequence.of(imEnd).concat(newline);
        this.closeTurn = List.of(Batch.prefill(close.toArray()));
    }

    @Override
    public java.util.Optional<com.qxotic.jinfer.chat.ToolCallDetector> toolCallDetector() {
        return java.util.Optional.of(
                new com.qxotic.jinfer.chat.SpanToolCallDetector(
                        tokenizer,
                        "<tool_call>",
                        "</tool_call>",
                        Qwen35ToolCallDetector::parsePayload));
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
        IntSequence ids =
                IntSequence.of(imStart)
                        .concat(tokenizer.encode(message.role().name() + "\n" + content))
                        .concat(IntSequence.of(imEnd))
                        .concat(newline);
        return List.of(Batch.prefill(ids.toArray()));
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
