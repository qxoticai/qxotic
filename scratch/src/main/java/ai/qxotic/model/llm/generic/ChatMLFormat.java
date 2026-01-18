package ai.qxotic.model.llm.generic;

import ai.qxotic.model.llm.ChatFormat;
import ai.qxotic.tokenizers.IntSequence;
import ai.qxotic.tokenizers.Tokenizer;
import ai.qxotic.tokenizers.Vocabulary;
import java.util.Set;

/** Utility tailored for the Chat Markup Language (ChatML) prompt format. */
public class ChatMLFormat extends ChatFormat {

    public final int imStart;
    public final int endOfText;
    public final int imEnd;
    private final Set<Integer> stopTokens;

    public ChatMLFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();
        this.imStart = vocabulary.id("<|im_start|>");
        this.imEnd = vocabulary.id("<|im_end|>");
        this.endOfText = vocabulary.id("<|endoftext|>");
        this.stopTokens = Set.of(imEnd, endOfText);
    }

    @Override
    public Set<Integer> stopTokens() {
        return stopTokens;
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(imStart);
        builder.addAll(this.tokenizer.encode(role.name()));
        builder.addAll(this.tokenizer.encode("\n"));
        return builder.build();
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        validateRole(message.role());
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(message.role()));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        builder.add(imEnd);
        builder.addAll(this.tokenizer.encode("\n"));
        return builder.build();
    }
}
