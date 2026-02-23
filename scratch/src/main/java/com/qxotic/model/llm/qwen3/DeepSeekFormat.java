package com.qxotic.model.llm.qwen3;

import com.qxotic.model.llm.ChatFormat;
import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import java.util.OptionalInt;
import java.util.Set;

/** Chat format used by DeepSeek's distills of Qwen 3 models (thinking always enabled). */
public class DeepSeekFormat extends ChatFormat {

    public final int imStart;
    public final int endOfSentence;
    public final int user;
    public final int assistant;
    public final int beginOfSentence;
    public final int thinkStart;
    public final int thinkEnd;

    private final Set<Integer> stopTokens;

    public DeepSeekFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();
        this.beginOfSentence = vocabulary.id("<｜begin▁of▁sentence｜>");
        this.imStart = vocabulary.id("<|im_start|>");

        this.user = vocabulary.id("<｜User｜>");
        this.assistant = vocabulary.id("<｜Assistant｜>");

        this.thinkStart = vocabulary.id("<think>");
        this.thinkEnd = vocabulary.id("</think>");

        this.endOfSentence = vocabulary.id("<｜end▁of▁sentence｜>");
        this.stopTokens = Set.of(endOfSentence);
    }

    @Override
    public OptionalInt beginOfText() {
        return OptionalInt.of(beginOfSentence);
    }

    @Override
    public Set<Integer> stopTokens() {
        return stopTokens;
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        if (SYSTEM.equals(role)) {
            // nothing
        } else if (USER.equals(role)) {
            builder.add(this.user);
        } else if (ASSISTANT.equals(role)) {
            builder.add(this.assistant);
        }
        return builder.build();
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        validateRole(message.role());
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(message.role()));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        builder.addAll(this.tokenizer.encode("\n"));
        return builder.build();
    }
}
