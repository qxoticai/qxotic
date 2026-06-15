// The built-in ChatML-style chat format: turns OpenAI messages into LFM2 token streams using
// explicit special-token ids (header/turn/FIM markers), and exposes the generation stop set.
// Used by the CLI and as the server's template-less fallback.
package com.llama4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

class LFMChatFormat {

    final LFMTokenizer tokenizer;
    final int beginOfSentence;
    final int startOfTurn;
    final int endOfTurn;
    final int endOfSentence;
    final int fimSuffix;
    final int fimPrefix;
    final int fimMiddle;
    final int fileSeparator;
    private final Set<Integer> stopTokens;

    LFMChatFormat(LFMTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
        this.beginOfSentence = specialTokens.getOrDefault("<bos>", specialTokens.getOrDefault("<|startoftext|>", 1));
        this.startOfTurn = specialTokens.getOrDefault("<|im_start|>", specialTokens.getOrDefault("<|turn>", beginOfSentence));
        this.endOfTurn = specialTokens.getOrDefault("<|im_end|>", specialTokens.getOrDefault("<turn|>", -1));
        this.endOfSentence = specialTokens.getOrDefault("<eos>", specialTokens.getOrDefault("<|endoftext|>", 2));

        this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
        this.fileSeparator = specialTokens.getOrDefault("<|file_separator|>", -1);

        Set<Integer> tokens = new HashSet<>();
        tokens.add(endOfSentence);
        if (endOfTurn >= 0) tokens.add(endOfTurn);
        if (fimSuffix != -1) tokens.add(fimSuffix);
        if (fimPrefix != -1) tokens.add(fimPrefix);
        if (fimMiddle != -1) tokens.add(fimMiddle);
        if (fileSeparator != -1) tokens.add(fileSeparator);
        this.stopTokens = Collections.unmodifiableSet(tokens);
    }

    Set<Integer> getStopTokens() {
        return stopTokens;
    }

    List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startOfTurn);
        tokens.addAll(tokenizer.encode(message.role().toString()));
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encode(message.content().strip()));
        if (endOfTurn >= 0) tokens.add(endOfTurn);
        tokens.addAll(this.tokenizer.encode("\n"));
        return tokens;
    }

    List<Integer> encodeSystemThinkingTurn(String systemPrompt) {
        return encodeMessage(new Message(Role.SYSTEM, systemPrompt == null ? "" : systemPrompt));
    }

    /** Appends an empty think span ("&lt;think&gt;\n&lt;/think&gt;\n\n") so a non-thinking turn
     *  still matches the template the model was trained on. No-op without think markers. */
    void appendThinkSurrogate(List<Integer> tokens) {
        Integer start = tokenizer.getSpecialTokens().get("<think>");
        Integer end = tokenizer.getSpecialTokens().get("</think>");
        if (start == null || end == null) return;
        List<Integer> nl = tokenizer.encode("\n");
        tokens.add(start);
        tokens.addAll(nl);
        tokens.add(end);
        tokens.addAll(nl);
        tokens.addAll(nl);
    }

    List<Integer> encodeGenerationPrompt() {
        return encodeHeader(new Message(Role.ASSISTANT, ""));
    }

    record Message(Role role, String content) {
    }

    List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(this.fimPrefix);
        tokens.addAll(tokenizer.encode(prefix));
        tokens.add(this.fimSuffix);
        tokens.addAll(tokenizer.encode(suffix));
        tokens.add(this.fimMiddle);
        return tokens;
    }

    record Role(String name) {
        public static final Role SYSTEM = new Role("system");
        public static final Role USER = new Role("user");
        public static final Role ASSISTANT = new Role("assistant");
        public static final Role TOOL = new Role("tool");

        /** OpenAI role string to template role; unknown roles render as user turns. */
        public static Role of(String role) {
            return switch (role) {
                case "system" -> SYSTEM;
                case "assistant" -> ASSISTANT;
                case "tool" -> TOOL; // native tool turn, not flattened into user
                default -> USER;
            };
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
