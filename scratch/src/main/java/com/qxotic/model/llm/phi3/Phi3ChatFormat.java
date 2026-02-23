package com.qxotic.model.llm.phi3;

import com.qxotic.model.llm.ChatFormat;
import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import java.util.OptionalInt;
import java.util.Set;

/** Utility tailored for Phi3 instruct prompt format. */
public class Phi3ChatFormat extends ChatFormat {

    protected final int unknownToken;
    protected final int beginOfSentence;
    protected final int endOfSentence;
    protected final int endOfText;
    protected final int system;
    protected final int user;
    protected final int assistant;
    protected final int endOfTurn;
    private final Set<Integer> stopTokens;

    public Phi3ChatFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();
        this.unknownToken = vocabulary.id("<unk>");
        this.beginOfSentence = vocabulary.id("<s>");
        this.endOfSentence = vocabulary.id("</s>");
        this.endOfText = vocabulary.id("<|endoftext|>");

        this.system = vocabulary.id("<|system|>");
        this.user = vocabulary.id("<|user|>");
        this.assistant = vocabulary.id("<|assistant|>");
        this.endOfTurn = vocabulary.id("<|end|>");

        this.stopTokens = Set.of(endOfTurn, endOfSentence, endOfText);
    }

    @Override
    public Set<Integer> stopTokens() {
        return this.stopTokens;
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        switch (role.name()) {
            case "system" -> builder.add(this.system);
            case "user" -> builder.add(this.user);
            case "assistant" -> builder.add(this.assistant);
        }
        return builder.build();
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        validateRole(message.role());
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(message.role()));
        builder.addAll(this.tokenizer.encode("\n"));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        builder.add(this.endOfTurn);
        builder.addAll(this.tokenizer.encode("\n"));
        return builder.build();
    }

    @Override
    public OptionalInt beginOfText() {
        return OptionalInt.of(this.beginOfSentence);
    }

    @Override
    public OptionalInt endOfText() {
        return OptionalInt.of(this.endOfText);
    }
}
