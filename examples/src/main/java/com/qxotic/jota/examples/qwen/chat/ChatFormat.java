package com.qxotic.jota.examples.qwen.chat;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.StandardTokenType;
import java.util.HexFormat;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;

public abstract class ChatFormat {

    protected final Tokenizer tokenizer;

    public ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    public abstract Set<Integer> stopTokens();

    public OptionalInt beginOfText() {
        return OptionalInt.empty();
    }

    public OptionalInt endOfText() {
        return OptionalInt.empty();
    }

    public static class Message {
        final Role role;
        final String textContent;

        public Message(Role role, String textContent) {
            this.role = role;
            this.textContent = textContent;
        }

        public Role role() {
            return role;
        }

        public String textContent() {
            return textContent;
        }
    }

    public static class Role {
        final String name;

        public Role(String name) {
            this.name = name;
        }

        public String name() {
            return name;
        }
    }

    public boolean isValidRole(Role role) {
        return List.of(SYSTEM.name(), USER.name(), ASSISTANT.name()).contains(role.name());
    }

    protected void validateRole(Role role) {
        if (!isValidRole(role)) {
            throw new IllegalArgumentException("Invalid role: " + role);
        }
    }

    public abstract IntSequence encodeMessage(Message message);

    public abstract IntSequence encodeHeader(Role role);

    public IntSequence encodeDialog(List<Message> messages) {
        IntSequence.Builder builder = IntSequence.newBuilder();
        for (Message message : messages) {
            builder.addAll(encodeMessage(message));
        }
        return builder.build();
    }

    public static final Role SYSTEM = new Role("system");
    public static final Role USER = new Role("user");
    public static final Role ASSISTANT = new Role("assistant");

    private static String replaceControlCharacters(int[] codePoints) {
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
            } else {
                chars.appendCodePoint(cp);
            }
        }
        return chars.toString();
    }

    private static String replaceControlCharacters(String text) {
        return replaceControlCharacters(text.codePoints().toArray());
    }

    public String echo(IntSequence tokens) {
        return replaceControlCharacters(this.tokenizer.decode(tokens));
    }

    public String stream(IntSequence tokens) {
        IntSequence.Builder builder = IntSequence.newBuilder(tokens.length());
        Vocabulary vocabulary = tokenizer.vocabulary();
        for (int i = 0; i < tokens.length(); i++) {
            int tokenIndex = tokens.intAt(i);
            if (vocabulary.isTokenOfType(tokenIndex, StandardTokenType.NORMAL)
                    || vocabulary.isTokenOfType(tokenIndex, StandardTokenType.BYTE)) {
                builder.add(tokenIndex);
            }
        }
        return this.tokenizer.decode(builder.build());
    }
}
