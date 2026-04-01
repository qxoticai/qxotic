package com.qxotic.model.llm.apertus;

import com.qxotic.model.llm.ChatFormat;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.util.OptionalInt;
import java.util.Set;

public class ApertusChatFormat extends ChatFormat {

    private final int beginOfSentence;
    private final int endOfSentence;

    private final int systemToken;
    private final int endSystemToken;
    private final int developerToken;
    private final int endDeveloperToken;
    private final int userToken;
    private final int endUserToken;
    private final int assistantToken;
    private final int endAssistantToken;
    private final int innerToken;
    private final int outerToken;
    private final int toolCallsToken;
    private final int endToolCallsToken;

    public ApertusChatFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();
        this.beginOfSentence = vocabulary.id("<s>");
        this.endOfSentence = vocabulary.id("</s>");
        this.systemToken = vocabulary.id("<|system_start|>");
        this.endSystemToken = vocabulary.id("<|system_end|>");
        this.developerToken = vocabulary.id("<|developer_start|>");
        this.endDeveloperToken = vocabulary.id("<|developer_end|>");
        this.userToken = vocabulary.id("<|user_start|>");
        this.endUserToken = vocabulary.id("<|user_end|>");
        this.assistantToken = vocabulary.id("<|assistant_start|>");
        this.endAssistantToken = vocabulary.id("<|assistant_end|>");
        this.innerToken = vocabulary.id("<|inner_prefix|>");
        this.outerToken = vocabulary.id("<|inner_suffix|>");
        this.toolCallsToken = vocabulary.id("<|tools_prefix|>");
        this.endToolCallsToken = vocabulary.id("<|tools_suffix|>");
    }

    @Override
    public Set<Integer> stopTokens() {
        return Set.of(endOfSentence, endToolCallsToken, endAssistantToken);
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        Role role = message.role();
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(role));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        String name = role.name();
        switch (name) {
            case "system" -> builder.add(this.endSystemToken);
            case "assistant" -> builder.add(this.endAssistantToken);
            case "user" -> builder.add(this.endUserToken);
        }
        return builder.build();
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        String name = role.name();
        switch (name) {
            case "system" -> builder.add(this.systemToken);
            case "assistant" -> builder.add(this.assistantToken);
            case "user" -> builder.add(this.userToken);
        }
        return builder.build();
    }

    @Override
    public OptionalInt beginOfText() {
        return OptionalInt.of(beginOfSentence);
    }

    @Override
    public OptionalInt endOfText() {
        return OptionalInt.of(endOfSentence);
    }
}
