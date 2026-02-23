package com.qxotic.tokenizers.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Splitter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

public final class GGUFTokenizers {

    private static final GGUFPreTokenizerRegistry PRE_TOKENIZERS =
            GGUFPreTokenizerRegistry.defaults();
    private static final GGUFTokenizerRegistry TOKENIZERS = GGUFTokenizerRegistry.defaults();

    private GGUFTokenizers() {}

    public static GGUFPreTokenizerRegistry preTokenizers() {
        return PRE_TOKENIZERS;
    }

    public static GGUFTokenizerRegistry tokenizers() {
        return TOKENIZERS;
    }

    public static Tokenizer fromFile(Path ggufPath) {
        Objects.requireNonNull(ggufPath, "ggufPath");
        return fromFile(ggufPath, PRE_TOKENIZERS, TOKENIZERS);
    }

    public static Tokenizer fromFile(String ggufPath) {
        Objects.requireNonNull(ggufPath, "ggufPath");
        return fromFile(Path.of(ggufPath));
    }

    public static Tokenizer fromFile(Path ggufPath, GGUFPreTokenizerRegistry preTokenizers) {
        return fromFile(ggufPath, preTokenizers, TOKENIZERS);
    }

    public static Tokenizer fromFile(
            Path ggufPath,
            GGUFPreTokenizerRegistry preTokenizers,
            GGUFTokenizerRegistry tokenizers) {
        Objects.requireNonNull(ggufPath, "ggufPath");
        Objects.requireNonNull(preTokenizers, "preTokenizers");
        Objects.requireNonNull(tokenizers, "tokenizers");
        try {
            return fromGGUF(GGUF.read(ggufPath), preTokenizers, tokenizers);
        } catch (IOException e) {
            throw new GGUFTokenizerException("Failed to read GGUF file: " + ggufPath, e);
        }
    }

    public static Tokenizer fromGGUF(GGUF gguf) {
        Objects.requireNonNull(gguf, "gguf");
        return fromGGUF(gguf, PRE_TOKENIZERS, TOKENIZERS);
    }

    public static Tokenizer fromGGUF(GGUF gguf, GGUFPreTokenizerRegistry preTokenizers) {
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(preTokenizers, "preTokenizers");
        return fromGGUF(gguf, preTokenizers, TOKENIZERS);
    }

    public static Tokenizer fromGGUF(
            GGUF gguf, GGUFPreTokenizerRegistry preTokenizers, GGUFTokenizerRegistry tokenizers) {
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(preTokenizers, "preTokenizers");
        Objects.requireNonNull(tokenizers, "tokenizers");
        String tokenizerModel =
                gguf.getValueOrDefault(String.class, "tokenizer.ggml.model", "gpt2");
        GGUFTokenizerFactory tokenizerFactory = tokenizers.require(tokenizerModel);

        Splitter splitter = chooseSplitter(gguf, preTokenizers);
        return tokenizerFactory.create(gguf, splitter);
    }

    public static boolean isRegistered(String preTokenizer, String tokenizerModel) {
        return isRegistered(preTokenizer, tokenizerModel, PRE_TOKENIZERS, TOKENIZERS);
    }

    public static boolean isRegistered(
            String preTokenizer,
            String tokenizerModel,
            GGUFPreTokenizerRegistry preTokenizers,
            GGUFTokenizerRegistry tokenizers) {
        return preTokenizers.contains(preTokenizer) && tokenizers.contains(tokenizerModel);
    }

    private static Splitter chooseSplitter(GGUF gguf, GGUFPreTokenizerRegistry preTokenizers) {
        String pre = gguf.getValueOrDefault(String.class, "tokenizer.ggml.pre", "");
        return preTokenizers.require(pre);
    }
}
