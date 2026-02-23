package com.qxotic.model.llm.llama;

import com.qxotic.model.llm.ChatFormat;
import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import java.util.OptionalInt;
import java.util.Set;

/** Utility tailored for Llama 3 instruct prompt format. */
public class Llama3ChatFormat extends ChatFormat {

    public final int beginOfText;
    public final int endHeader;
    public final int startHeader;
    public final int endOfTurn;
    public final int endOfText;
    public final int endOfMessage;
    final Set<Integer> stopTokens;

    public Llama3ChatFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();
        this.beginOfText = vocabulary.id("<|begin_of_text|>");
        this.startHeader = vocabulary.id("<|start_header_id|>");
        this.endHeader = vocabulary.id("<|end_header_id|>");
        this.endOfTurn = vocabulary.id("<|eot_id|>");
        this.endOfText = vocabulary.id("<|end_of_text|>");
        this.endOfMessage =
                vocabulary.contains("<|eom_id|>") ? vocabulary.id("<|eom_id|>") : -1; // since 3-.1+
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }

    @Override
    public Set<Integer> stopTokens() {
        return stopTokens;
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(startHeader);
        builder.addAll(this.tokenizer.encode(role.name()));
        builder.add(endHeader);
        builder.addAll(this.tokenizer.encode("\n\n"));
        return builder;
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(message.role()));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        builder.add(endOfTurn);
        return builder;
    }

    @Override
    public OptionalInt beginOfText() {
        return OptionalInt.of(this.beginOfText);
    }

    @Override
    public OptionalInt endOfText() {
        return OptionalInt.of(this.endOfText);
    }

    //
    //    public IntSequence encodeText(String text, boolean prependBOS, boolean appendEOS) {
    //        IntSequence.Builder builder = IntSequence.newBuilder();
    //        if (prependBOS) {
    //            builder.add(this.beginOfText);
    //        }
    //        builder.addAll(this.tokenizer.encode(text));
    //        if (appendEOS) {
    //            builder.add(this.endOfText);
    //        }
    //        return builder;
    //    }
}
