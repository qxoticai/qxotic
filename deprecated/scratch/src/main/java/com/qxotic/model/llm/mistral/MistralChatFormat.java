package com.qxotic.model.llm.mistral;

import com.qxotic.model.llm.ChatFormat;
import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;

/** Utility tailored for Mistral v0.3 instruct prompt format. */
public class MistralChatFormat extends ChatFormat {

    public final int unknownToken;
    public final int beginOfText;
    public final int endOfText;
    public final int beginOfInstruction;
    public final int endOfInstruction;
    public final int toolCalls;
    public final int beginOfAvailableTools;
    public final int endOfAvailableTools;
    public final int beginOfToolResults;
    public final int endOfToolResults;
    public final int prefix;
    public final int middle;
    public final int suffix;
    private final Set<Integer> stopTokens;

    public MistralChatFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();
        this.unknownToken = vocabulary.id("<unk>");
        this.beginOfText = vocabulary.id("<s>");
        this.endOfText = vocabulary.id("</s>");
        this.beginOfInstruction = vocabulary.id("[INST]");
        this.endOfInstruction = vocabulary.id("[/INST]");
        this.toolCalls = vocabulary.id("[TOOL_CALLS]");
        this.beginOfAvailableTools = vocabulary.id("[AVAILABLE_TOOLS]");
        this.endOfAvailableTools = vocabulary.id("[/AVAILABLE_TOOLS]");
        this.beginOfToolResults = vocabulary.id("[TOOL_RESULTS]");
        this.endOfToolResults = vocabulary.id("[/TOOL_RESULTS]");

        // Only Codestral supports FIM tokens.
        this.prefix = vocabulary.contains("[PREFIX]") ? vocabulary.id("[PREFIX]") : unknownToken;
        this.suffix = vocabulary.contains("[SUFFIX]") ? vocabulary.id("[SUFFIX]") : unknownToken;
        this.middle = vocabulary.contains("[MIDDLE]") ? vocabulary.id("[MIDDLE]") : unknownToken;

        this.stopTokens = Set.of(endOfText, endOfInstruction);
    }

    @Override
    public Set<Integer> stopTokens() {
        return stopTokens;
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        validateRole(message.role());
        IntSequence.Builder builder = IntSequence.newBuilder();
        switch (message.role().name()) {
            case "user" -> {
                builder.add(this.beginOfInstruction);
                // SPACES ARE IMPORTANT, otherwise the model is completely lost.
                String userMessage = " " + message.textContent() + " ";
                builder.addAll(this.tokenizer.encode(userMessage));
                builder.add(this.endOfInstruction);
            }
            case "assistant" -> {
                String assistantMessage = message.textContent();
                builder.addAll(this.tokenizer.encode(assistantMessage));
                builder.add(this.endOfText);
            }
            default ->
                    throw new IllegalArgumentException(
                            "Message role not supported " + message.role());
        }
        return builder.build();
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        switch (role.name()) {
            case "user" -> builder.add(this.beginOfInstruction);
            case "assistant" -> {}
        }
        return builder.build();
    }

    public IntSequence encodeFillInTheMiddle(String prefix, String suffix) {
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(this.suffix);
        builder.addAll(tokenizer.encode(suffix));
        builder.add(this.prefix);
        builder.addAll(tokenizer.encode(prefix));
        return builder.build();
    }

    @Override
    public boolean isValidRole(Role role) {
        // Mistral doesn't support system prompts.
        return List.of(USER.name(), ASSISTANT.name()).contains(role.name());
    }

    @Override
    public OptionalInt beginOfText() {
        return OptionalInt.of(this.beginOfText);
    }

    @Override
    public OptionalInt endOfText() {
        return OptionalInt.of(this.endOfText);
    }
}
