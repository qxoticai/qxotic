package ai.qxotic.model.llm.gemma3;

import ai.qxotic.model.llm.ChatFormat;
import ai.qxotic.tokenizers.IntSequence;
import ai.qxotic.tokenizers.Tokenizer;
import ai.qxotic.tokenizers.Vocabulary;

import java.util.OptionalInt;
import java.util.Set;

/**
 * Chat format used by DeepSeek's distills of Qwen 3 models (thinking always enabled).
 */
public class GemmaChatFormat extends ChatFormat {


    public final int beginOfSentence;
    public final int endOfSentence;
    public final int startOfTurn;
    public final int endOfTurn;

    private final Set<Integer> stopTokens;

    public GemmaChatFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();

        this.beginOfSentence = vocabulary.id("<bos>");
        this.startOfTurn = vocabulary.id("<start_of_turn>");
        this.endOfTurn = vocabulary.id("<end_of_turn>");
        this.endOfSentence = vocabulary.id("<eos>");

//        this.fimSuffix = vocabulary.getOrDefault("<|fim_suffix|>", -1);
//        this.fimPrefix = vocabulary.getOrDefault("<|fim_prefix|>", -1);
//        this.fimMiddle = vocabulary.getOrDefault("<|fim_middle|>", -1);
//        this.fileSeparator = vocabulary.getOrDefault("<|file_separator|>", -1);

        this.stopTokens = Set.of(endOfTurn, endOfSentence);
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
        builder.add(this.startOfTurn);
        if (ASSISTANT.equals(role)) {
            builder.addAll(tokenizer.encode("model"));
        } else {
            builder.addAll(tokenizer.encode(role.name()));
        }
        builder.addAll(tokenizer.encode("\n"));
        return builder.build();
    }

    @Override
    public boolean isValidRole(Role role) {
        return USER.equals(role) || ASSISTANT.equals(role);
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        validateRole(message.role());
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(message.role()));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        builder.add(this.endOfTurn);
        builder.addAll(this.tokenizer.encode("\n"));
        return builder.build();
    }
}
