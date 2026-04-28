package com.qxotic.toknroll.hf;

import com.qxotic.toknroll.ByteLevel;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Shared test fixture helpers for HuggingFace tokenizer tests. Reduces duplication of JSON builder
 * methods across test classes.
 */
final class HuggingFaceTokenizerTestFixtures {

    private HuggingFaceTokenizerTestFixtures() {}

    /**
     * Builds a tokenizer.json root object wrapping the given model JSON and optional root fields.
     */
    static String buildTokenizerJson(String modelJson, String... rootFields) {
        StringBuilder sb = new StringBuilder();
        sb.append('{').append("\"model\":").append(modelJson);
        for (String rootField : rootFields) {
            sb.append(',').append(rootField);
        }
        sb.append('}');
        return sb.toString();
    }

    /** Builds a BPE model JSON fragment with vocab, merges, and optional extra fields. */
    static String buildBpeModel(String vocabJson, String mergesJson, String extraFields) {
        return "{\"type\":\"BPE\",\"vocab\":"
                + vocabJson
                + ",\"merges\":"
                + mergesJson
                + extraFields
                + "}";
    }

    /**
     * Builds a byte-level vocabulary JSON with all 256 GPT-2 byte tokens plus extra tokens at
     * specified IDs.
     */
    static String buildByteLevelVocab(Map<String, Integer> extraTokens) {
        StringBuilder vocab = new StringBuilder();
        vocab.append('{');
        for (int b = 0; b < 256; b++) {
            if (b > 0) {
                vocab.append(',');
            }
            vocab.append('"')
                    .append(escapeJson(String.valueOf(ByteLevel.encodeSingle((byte) b))))
                    .append('"')
                    .append(':')
                    .append(b);
        }

        Map<String, Integer> orderedExtras = new LinkedHashMap<>(extraTokens);
        for (Map.Entry<String, Integer> entry : orderedExtras.entrySet()) {
            vocab.append(',')
                    .append('"')
                    .append(escapeJson(entry.getKey()))
                    .append('"')
                    .append(':')
                    .append(entry.getValue());
        }
        vocab.append('}');
        return vocab.toString();
    }

    /**
     * Builds a SentencePiece-style vocabulary JSON with <0x00>...<0xFF> byte fallback tokens plus
     * extra tokens at specified IDs.
     */
    static String buildSentencePieceVocab(Map<String, Integer> extraTokens) {
        StringBuilder vocab = new StringBuilder();
        vocab.append('{');
        for (int b = 0; b < 256; b++) {
            if (b > 0) {
                vocab.append(',');
            }
            vocab.append('"')
                    .append(String.format("<0x%02X>", b))
                    .append('"')
                    .append(':')
                    .append(b);
        }

        Map<String, Integer> orderedExtras = new LinkedHashMap<>(extraTokens);
        for (Map.Entry<String, Integer> entry : orderedExtras.entrySet()) {
            vocab.append(',')
                    .append('"')
                    .append(escapeJson(entry.getKey()))
                    .append('"')
                    .append(':')
                    .append(entry.getValue());
        }
        vocab.append('}');
        return vocab.toString();
    }

    /** Escapes a string for inclusion in JSON double-quoted strings. */
    static String escapeJson(String value) {
        StringBuilder escaped = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '\\':
                    escaped.append("\\\\");
                    break;
                case '"':
                    escaped.append("\\\"");
                    break;
                case '\n':
                    escaped.append("\\n");
                    break;
                case '\r':
                    escaped.append("\\r");
                    break;
                case '\t':
                    escaped.append("\\t");
                    break;
                default:
                    escaped.append(c);
            }
        }
        return escaped.toString();
    }
}
