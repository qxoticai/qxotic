package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Hand-written gpt-oss (Harmony) chat framing, matching the GGUF chat_template's plain-conversation
 * shape.
 *
 * <p>Layout: a fixed system preamble once ({@code <|start|>system<|message|>{identity, cutoff,
 * date, reasoning, channels}<|end|>}), the conversation's system message as a developer block
 * ({@code <|start|>developer<|message|># Instructions\n\n{content}<|end|>}), then per turn {@code
 * <|start|>user<|message|>{content}<|end|>} and, for assistant history, {@code
 * <|start|>assistant<|channel|>final<|message|>{content}<|end|>} (the template drops CoT from
 * history - only the final channel is re-rendered). Generation prompt is {@code
 * <|start|>assistant}; the model then emits its own channel tokens ({@code
 * <|channel|>analysis<|message|>...}), so {@code thinking} is a no-op here - Harmony always
 * reasons, and the effort knob lives in the system preamble ({@code Reasoning: medium}).
 *
 * <p>Each text run between specials is ONE contiguous plain {@link Tokenizer#encode} (that is how a
 * rendered template tokenizes; specials force the only splits), and conversation content never goes
 * through special-aware encoding, so text cannot mint control tokens.
 *
 * <p>The preamble embeds the current date ({@code strftime_now("%Y-%m-%d")} in the template); it is
 * a constructor argument so tests are deterministic - the convenience constructor pins today,
 * matching what the template renders.
 */
public final class GptOssTurnTemplate implements TurnTemplate {

    static final String DEFAULT_IDENTITY =
            "You are ChatGPT, a large language model trained by OpenAI.";
    static final String DEFAULT_EFFORT = "medium";

    private final Tokenizer tokenizer;
    private final String systemText;
    private final List<Batch> conversationStart; // fixed preamble, encoded once
    private final int start; // <|start|>
    private final int message; // <|message|>
    private final int channel; // <|channel|>
    private final int end; // <|end|>

    public GptOssTurnTemplate(Tokenizer tokenizer) {
        this(tokenizer, LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
    }

    public GptOssTurnTemplate(Tokenizer tokenizer, String currentDate) {
        this.tokenizer = tokenizer;
        this.start = SpecialTokens.require(tokenizer, "<|start|>");
        this.message = SpecialTokens.require(tokenizer, "<|message|>");
        this.channel = SpecialTokens.require(tokenizer, "<|channel|>");
        this.end = SpecialTokens.require(tokenizer, "<|end|>");
        this.systemText =
                DEFAULT_IDENTITY
                        + "\n"
                        + "Knowledge cutoff: 2024-06\n"
                        + "Current date: "
                        + currentDate
                        + "\n\n"
                        + "Reasoning: "
                        + DEFAULT_EFFORT
                        + "\n\n"
                        + "# Valid channels: analysis, commentary, final. Channel must be included"
                        + " for every message.";
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        ids.addAll(tokenizer.encode("system"));
        ids.add(message);
        ids.addAll(tokenizer.encode(systemText));
        ids.add(end);
        this.conversationStart = List.of(Batch.prefill(ids.build().toArray()));
    }

    /** The fixed Harmony system preamble: {@code <|start|>system<|message|>{...}<|end|>}. */
    @Override
    public List<Batch> conversationStart() {
        return conversationStart;
    }

    @Override
    public List<Batch> encodeTurn(Message m) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        if (m.role().equals(Role.SYSTEM)) { // conversation system -> developer block
            ids.addAll(tokenizer.encode("developer"));
            ids.add(message);
            ids.addAll(tokenizer.encode("# Instructions\n\n" + m.textOnly()));
        } else if (m.role().equals(Role.ASSISTANT)) { // history keeps only the final channel
            ids.addAll(tokenizer.encode("assistant"));
            ids.add(channel);
            ids.addAll(tokenizer.encode("final"));
            ids.add(message);
            ids.addAll(tokenizer.encode(m.textOnly()));
        } else {
            ids.addAll(tokenizer.encode(m.role().name()));
            ids.add(message);
            ids.addAll(tokenizer.encode(m.textOnly()));
        }
        ids.add(end);
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    /** {@code <|start|>assistant} - the model emits its own channel tokens from here. */
    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        ids.addAll(tokenizer.encode("assistant"));
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    /** Closes the open message: {@code <|end|>} (the {@code <|return|>} stop is never ingested). */
    @Override
    public List<Batch> closeTurn() {
        return List.of(Batch.prefill(new int[] {end}));
    }

    @Override
    public ReplyParser parser() {
        return new HarmonyReplyParser(tokenizer);
    }
}
