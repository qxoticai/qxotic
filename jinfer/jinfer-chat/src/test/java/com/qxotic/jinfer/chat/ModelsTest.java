package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.lang.reflect.Proxy;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.util.function.Consumer;

class ModelsTest {

    private static ModelProvider provider(int priority, String... archs) {
        Set<String> claimed = Set.of(archs);
        return new ModelProvider() {
            @Override
            public boolean supports(String architecture) {
                return claimed.contains(architecture);
            }

            @Override
            public int priority() {
                return priority;
            }

            @Override
            public LoadedModel<?> load(
                    FileChannel fc,
                    GGUF gguf,
                    Arena arena,
                    Map<String, Path> companions,
                    Tokenizer tokenizer) {
                throw new UnsupportedOperationException();
            }
        };
    }

    @Test
    void selectHonorsPriorityAndIgnoresNonSupporters() {
        ModelProvider low = provider(0, "llama");
        ModelProvider high = provider(5, "llama");
        ModelProvider other = provider(9, "qwen35");

        assertSame(high, Models.select(List.of(low, high, other), "llama"));
        assertSame(low, Models.select(List.of(low, other), "llama"));
        assertNull(Models.select(List.of(low), "gemma4"));
    }

    @Test
    void selectBreaksEqualPriorityTiesDeterministically() {
        // two DISTINCT classes at equal priority: the class-name order decides, and list order
        // must not
        ModelProvider a = new AlphaProvider();
        ModelProvider z = new ZetaProvider();
        ModelProvider first = Models.select(List.of(a, z), "llama");
        assertSame(first, Models.select(List.of(z, a), "llama"));
        assertSame(a, first); // AlphaProvider sorts before ZetaProvider
    }

    private abstract static class NamedProvider implements ModelProvider {
        @Override
        public boolean supports(String architecture) {
            return architecture.equals("llama");
        }

        @Override
        public LoadedModel<?> load(
                FileChannel fc,
                GGUF gguf,
                Arena arena,
                Map<String, Path> companions,
                Tokenizer tokenizer) {
            throw new UnsupportedOperationException();
        }
    }

    private static final class AlphaProvider extends NamedProvider {}

    private static final class ZetaProvider extends NamedProvider {}

    @Test
    void companionSeedingIsOrderIndependentAndCapabilitySensitive(@TempDir Path dir)
            throws Exception {
        Path media = dir.resolve("mmproj.gguf");
        Path spec = dir.resolve("mtp.gguf");
        Files.write(media, new byte[] {1, 2, 3});
        Files.write(spec, new byte[] {4, 5, 6});

        LoadedModel<?> base = loadedModel(ContentKey.sha256(new byte[] {0}));
        LoadedModel<?> ab = Models.companionSeeded(base, Map.of("media", media, "spec", spec));
        Map<String, Path> reversed = new LinkedHashMap<>();
        reversed.put("spec", spec);
        reversed.put("media", media);
        LoadedModel<?> ba = Models.companionSeeded(base, reversed);

        assertEquals(ab.seed(), ba.seed()); // sorted: listing order is irrelevant
        assertNotEquals(base.seed(), ab.seed()); // companions change the key space
        assertSame(base, Models.companionSeeded(base, Map.of())); // no companions, no re-root
    }

    @Test
    void modelSeedIsStableAndContentSensitive(@TempDir Path dir) throws Exception {
        Path a = dir.resolve("a.gguf");
        Files.write(a, new byte[] {1, 2, 3, 4});
        ContentKey first = Models.modelSeed(a);
        assertEquals(first, Models.modelSeed(a));
        Files.write(a, new byte[] {1, 2, 3, 5});
        assertNotEquals(first, Models.modelSeed(a));
        assertTrue(first.value().startsWith("sha256:"));
    }

    @Test
    void loadedModelKeepsStopTokenOrderAndOwnership() {
        Set<Integer> source = new LinkedHashSet<>(List.of(5, 2, 9));
        LoadedModel<?> loaded = loadedModel(ContentKey.sha256(new byte[] {0}), source);

        source.add(4);
        assertIterableEquals(List.of(5, 2, 9), loaded.stopTokens());
        assertThrows(UnsupportedOperationException.class, () -> loaded.stopTokens().add(4));
    }

    @Test
    void customChatTemplateSourceIsImmutableAndKeepsOnlyThePromptStart() {
        IntSequence promptStart = IntSequence.of(7, 8);
        ChatTemplate nativeTemplate =
                new ChatTemplate() {
                    @Override
                    public IntSequence promptStart() {
                        return promptStart;
                    }

                    @Override
                    public ReplyState encode(
                            Conversation conversation,
                            int batchCapacity,
                            Consumer<Batch> sink) {
                        return null;
                    }
                };
        LoadedModel<?> base =
                new LoadedModel<>(
                        loadedModel(ContentKey.sha256(new byte[] {0})).model(),
                        tokenizer(0),
                        "original",
                        Set.of(2),
                        ContentKey.sha256(new byte[] {1}),
                        Optional.of(nativeTemplate),
                        LoadedModel.SamplingDefaults.NONE);

        LoadedModel<?> changed = base.withChatTemplateSource("replacement");

        assertEquals("original", base.chatTemplateSource());
        assertSame(nativeTemplate, base.template().orElseThrow());
        assertEquals("replacement", changed.chatTemplateSource());
        ChatTemplate fallback = changed.template().orElseThrow();
        assertEquals(promptStart, fallback.promptStart());
        assertThrows(UnsupportedConversation.class, () -> fallback.encode(null, 1, ignored -> {}));
        assertSame(base.model(), changed.model());
        assertSame(base.tokenizer(), changed.tokenizer());
        assertEquals(base.seed(), changed.seed());
        assertThrows(IllegalArgumentException.class, () -> base.withChatTemplateSource(null));
        assertThrows(IllegalArgumentException.class, () -> base.withChatTemplateSource("  "));
    }

    @Test
    void customChatTemplateDoesNotRetainAnEmptyCodec() {
        LoadedModel<?> base = loadedModel(ContentKey.sha256(new byte[] {0}));

        assertTrue(base.withChatTemplateSource("replacement").template().isEmpty());
    }

    @Test
    void tokenizerOverrideMustKeepTheModelsIdSpace() {
        GGUF gguf =
                Builder.newBuilder()
                        .putArrayOfString("tokenizer.ggml.tokens", new String[] {"a", "b"})
                        .build();

        assertDoesNotThrow(() -> Models.requireSameIdSpace(gguf, tokenizer(2)));
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Models.requireSameIdSpace(gguf, tokenizer(1)));
        assertTrue(failure.getMessage().contains("1 tokens"), failure.getMessage());
        assertTrue(failure.getMessage().contains("2"), failure.getMessage());
    }

    private static LoadedModel<?> loadedModel(ContentKey seed) {
        return loadedModel(seed, Set.of());
    }

    private static LoadedModel<?> loadedModel(ContentKey seed, Set<Integer> stopTokens) {
        LanguageModel<?, ?, ?> model =
                (LanguageModel<?, ?, ?>)
                        Proxy.newProxyInstance(
                                ModelsTest.class.getClassLoader(),
                                new Class<?>[] {LanguageModel.class},
                                (proxy, method, args) -> {
                                    throw new UnsupportedOperationException();
                                });
        return new LoadedModel<>(
                model,
                tokenizer(0),
                "",
                stopTokens,
                seed,
                Optional.empty(),
                LoadedModel.SamplingDefaults.NONE);
    }

    private static Tokenizer tokenizer(int vocabularySize) {
        Vocabulary vocabulary =
                (Vocabulary)
                        Proxy.newProxyInstance(
                                ModelsTest.class.getClassLoader(),
                                new Class<?>[] {Vocabulary.class},
                                (proxy, method, args) -> {
                                    if (method.getName().equals("size")) return vocabularySize;
                                    throw new UnsupportedOperationException();
                                });
        return (Tokenizer)
                Proxy.newProxyInstance(
                        ModelsTest.class.getClassLoader(),
                        new Class<?>[] {Tokenizer.class},
                        (proxy, method, args) -> {
                            if (method.getName().equals("vocabulary")) return vocabulary;
                            throw new UnsupportedOperationException();
                        });
    }
}
