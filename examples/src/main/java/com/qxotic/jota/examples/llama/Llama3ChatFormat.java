package com.qxotic.jota.examples.llama;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.StandardTokenType;
import java.util.OptionalInt;
import java.util.Set;

final class Llama3ChatFormat {
    public static final Role SYSTEM = new Role("system");
    public static final Role USER = new Role("user");
    public static final Role ASSISTANT = new Role("assistant");

    private final Tokenizer tokenizer;
    private final int beginOfText;
    private final int startHeader;
    private final int endHeader;
    private final int endOfTurn;
    private final int endOfText;
    private final Set<Integer> stopTokens;

    Llama3ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Vocabulary v = tokenizer.vocabulary();
        this.beginOfText = v.id("<|begin_of_text|>");
        this.startHeader = v.id("<|start_header_id|>");
        this.endHeader = v.id("<|end_header_id|>");
        this.endOfTurn = v.id("<|eot_id|>");
        this.endOfText = v.id("<|end_of_text|>");
        this.stopTokens = Set.of(endOfTurn, endOfText);
    }

    OptionalInt beginOfText() {
        return OptionalInt.of(beginOfText);
    }

    Set<Integer> stopTokens() {
        return stopTokens;
    }

    IntSequence encodeHeader(Role role) {
        IntSequence.Builder b = IntSequence.newBuilder();
        b.add(startHeader);
        b.addAll(tokenizer.encode(role.name));
        b.add(endHeader);
        b.addAll(tokenizer.encode("\n\n"));
        return b.build();
    }

    IntSequence encodeMessage(MessageLike message) {
        IntSequence.Builder b = IntSequence.newBuilder();
        b.addAll(encodeHeader(message.role()));
        b.addAll(tokenizer.encode(message.text().strip()));
        b.add(endOfTurn);
        return b.build();
    }

    String stream(IntSequence tokens) {
        IntSequence.Builder b = IntSequence.newBuilder(tokens.length());
        Vocabulary vocabulary = tokenizer.vocabulary();
        for (int i = 0; i < tokens.length(); i++) {
            int t = tokens.intAt(i);
            if (vocabulary.isTokenOfType(t, StandardTokenType.NORMAL)
                    || vocabulary.isTokenOfType(t, StandardTokenType.BYTE)) {
                b.add(t);
            }
        }
        return tokenizer.decode(b.build());
    }

    String echo(IntSequence tokens) {
        return tokenizer.decode(tokens);
    }

    static final class Role {
        final String name;

        Role(String name) {
            this.name = name;
        }
    }

    interface MessageLike {
        Role role();

        String text();
    }
}
