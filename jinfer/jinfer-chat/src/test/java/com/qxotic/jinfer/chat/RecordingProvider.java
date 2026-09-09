package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.ContextModel;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Reranker;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.lang.reflect.Proxy;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * A {@link ModelProvider} on the test classpath (META-INF/services) for the {@code fake}
 * architecture: it records what {@link Models} hands it and answers with inert proxies, so the
 * dispatch can be exercised end to end without weights. Language, reranker and speech are
 * implemented; embedding is left to the default, to observe a refused kind.
 */
public final class RecordingProvider implements ModelProvider {

    /** What the last {@code load*} call received. */
    public record Call(
            String kind,
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer) {}

    private static volatile Call last;

    public static Call last() {
        return last;
    }

    public static void reset() {
        last = null;
    }

    @Override
    public Set<String> architectures() {
        return Set.of("fake", "fake-too");
    }

    @Override
    public Map<String, String> companionFiles() {
        return Map.of("media", "mmproj", "lexicon", "lexicon");
    }

    @Override
    public LoadedModel<?> loadLanguage(
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer) {
        last = new Call("language", channel, gguf, path, arena, companions, tokenizer);
        return new LoadedModel<>(
                proxy(LanguageModel.class),
                tokenizer != null ? tokenizer : tokenizer(0),
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                Set.of(),
                Models.modelSeed(channel),
                Optional.empty(),
                LoadedModel.SamplingDefaults.NONE);
    }

    @Override
    public LoadedReranker<?> loadReranker(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer) {
        last = new Call("reranker", channel, gguf, path, arena, Map.of(), tokenizer);
        return new LoadedReranker<>(
                proxy(ContextModel.class), proxy(Reranker.class), path.getFileName().toString());
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions) {
        last = new Call("speech", channel, gguf, path, arena, companions, null);
        return proxy(SpeechSynthesisModel.class);
    }

    @SuppressWarnings("unchecked")
    static <T> T proxy(Class<?> type) {
        return (T)
                Proxy.newProxyInstance(
                        RecordingProvider.class.getClassLoader(),
                        new Class<?>[] {type},
                        (proxy, method, args) -> {
                            throw new UnsupportedOperationException(method.getName());
                        });
    }

    /** A tokenizer whose vocabulary reports {@code size}; nothing else works. */
    static Tokenizer tokenizer(int size) {
        Vocabulary vocabulary =
                (Vocabulary)
                        Proxy.newProxyInstance(
                                RecordingProvider.class.getClassLoader(),
                                new Class<?>[] {Vocabulary.class},
                                (proxy, method, args) -> {
                                    if (method.getName().equals("size")) return size;
                                    throw new UnsupportedOperationException(method.getName());
                                });
        return (Tokenizer)
                Proxy.newProxyInstance(
                        RecordingProvider.class.getClassLoader(),
                        new Class<?>[] {Tokenizer.class},
                        (proxy, method, args) -> {
                            if (method.getName().equals("vocabulary")) return vocabulary;
                            throw new UnsupportedOperationException(method.getName());
                        });
    }
}
