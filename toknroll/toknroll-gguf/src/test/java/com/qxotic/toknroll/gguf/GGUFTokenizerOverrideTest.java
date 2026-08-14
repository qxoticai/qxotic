package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The pre-tokenizer extension points: symbolic aliases (order-independent, cycle-checked at build)
 * and the {@code -Dtoknroll.gguf.pre.<name>=alias:|regex:|file:} escape hatch, with the priority
 * builtin &lt; code registration &lt; system property. Everything runs on in-memory GGUF metadata
 * and a recording stub model, so no real model files are needed.
 */
class GGUFTokenizerOverrideTest {

    private static final String PREFIX = GGUFTokenizerLoader.OVERRIDE_PREFIX;

    @AfterEach
    void clearOverrides() {
        System.getProperties().stringPropertyNames().stream()
                .filter(key -> key.startsWith(PREFIX))
                .toList()
                .forEach(System::clearProperty);
    }

    // ---- symbolic aliases ----

    @Test
    void aliasDeclaredBeforeItsTargetResolves() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        builder.aliasPreTokenizer("yi", "charly"); // target not registered yet
        registerScheme(builder, "charly", charByChar());
        assertChunks(builder, "yi", "ab", List.of("a", "b"));
    }

    @Test
    void aliasChainResolvesToTheTerminalScheme() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        builder.aliasPreTokenizer("a", "b");
        builder.aliasPreTokenizer("b", "charly");
        registerScheme(builder, "charly", charByChar());
        assertChunks(builder, "a", "ab", List.of("a", "b"));
    }

    @Test
    void aliasCycleFailsAtBuildWithTheChain() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        builder.aliasPreTokenizer("a", "b");
        builder.aliasPreTokenizer("b", "a");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, builder::build);
        assertTrue(e.getMessage().contains("'a' -> 'b' -> 'a'"), e.getMessage());
    }

    @Test
    void selfAliasFailsAtBuild() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        builder.aliasPreTokenizer("a", "a");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, builder::build);
        assertTrue(e.getMessage().contains("cycle"), e.getMessage());
    }

    @Test
    void danglingAliasFailsAtBuildListingKnownNames() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        builder.aliasPreTokenizer("yi", "no-such-scheme");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, builder::build);
        assertTrue(e.getMessage().contains("no-such-scheme"), e.getMessage());
        assertTrue(e.getMessage().contains("llama-bpe"), e.getMessage());
    }

    @Test
    void concreteRegistrationAfterAliasWins() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        registerScheme(builder, "charly", charByChar());
        builder.aliasPreTokenizer("yi", "charly");
        registerScheme(builder, "yi", letterRuns()); // latest call decides: concrete
        assertChunks(builder, "yi", "foo123bar", List.of("foo", "123", "bar"));
    }

    @Test
    void aliasAfterConcreteRegistrationWins() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        registerScheme(builder, "charly", charByChar());
        registerScheme(builder, "yi", letterRuns());
        builder.aliasPreTokenizer("yi", "charly"); // latest call decides: alias
        assertChunks(builder, "yi", "ab", List.of("a", "b"));
    }

    // ---- the property escape hatch ----

    @Test
    void propertyRegexRegistersANewScheme() {
        System.setProperty(PREFIX + "my-scheme", "regex:.");
        assertChunks(
                GGUFTokenizerLoader.createBuilderWithBuiltins(),
                "my-scheme",
                "ab",
                List.of("a", "b"));
    }

    @Test
    void propertyAliasMakesTheNameBehaveLikeTheTarget() {
        System.setProperty(PREFIX + "my-scheme", "alias:llama-bpe");
        assertChunks(
                GGUFTokenizerLoader.createBuilderWithBuiltins(),
                "my-scheme",
                "hello world",
                List.of("hello", " world"));
    }

    @Test
    void propertyFileRegistersPatterns(@TempDir Path tempDir) throws Exception {
        Path patterns = tempDir.resolve("patterns.txt");
        Files.writeString(patterns, "# one pattern per line, comments skipped\n\n.\n");
        System.setProperty(PREFIX + "my-scheme", "file:" + patterns);
        assertChunks(
                GGUFTokenizerLoader.createBuilderWithBuiltins(),
                "my-scheme",
                "ab",
                List.of("a", "b"));
    }

    @Test
    void propertyAliasCanTargetAnotherPropertySuppliedScheme() {
        // sorted property order applies the alias before its target exists; symbolic
        // resolution does not care
        System.setProperty(PREFIX + "aa-alias", "alias:zz-scheme");
        System.setProperty(PREFIX + "zz-scheme", "regex:.");
        assertChunks(
                GGUFTokenizerLoader.createBuilderWithBuiltins(),
                "aa-alias",
                "ab",
                List.of("a", "b"));
    }

    @Test
    void emptyBuilderStillHonorsPropertyOverrides() {
        System.setProperty(PREFIX + "my-scheme", "regex:.");
        assertChunks(
                GGUFTokenizerLoader.createEmptyBuilder(), "my-scheme", "ab", List.of("a", "b"));
    }

    // ---- priority: builtin < code < property ----

    @Test
    void codeRegistrationOverridesTheBuiltin() {
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        registerScheme(builder, "llama-bpe", charByChar());
        assertChunks(builder, "llama-bpe", "ab", List.of("a", "b"));
    }

    @Test
    void propertyOverridesTheBuiltin() {
        System.setProperty(PREFIX + "llama-bpe", "regex:.");
        assertChunks(
                GGUFTokenizerLoader.createBuilderWithBuiltins(),
                "llama-bpe",
                "ab",
                List.of("a", "b"));
    }

    @Test
    void propertyOverridesACodeRegistration() {
        System.setProperty(PREFIX + "my-scheme", "regex:.");
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        registerScheme(builder, "my-scheme", letterRuns());
        assertChunks(builder, "my-scheme", "ab", List.of("a", "b"));
    }

    @Test
    void aliasFollowsAPropertyOverrideOfItsTarget() {
        // the symbolic-alias headline: yi tracks llama-bpe, the flag redefines llama-bpe,
        // so yi splits like the flag says - capture-at-call aliases would not
        System.setProperty(PREFIX + "llama-bpe", "regex:.");
        GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
        builder.aliasPreTokenizer("yi", "llama-bpe");
        assertChunks(builder, "yi", "ab", List.of("a", "b"));
    }

    @Test
    void replacingARegisteredNameIsLoggedWithItsFollowers() {
        Logger jul = Logger.getLogger(GGUFTokenizerLoader.class.getName());
        List<String> records = new ArrayList<>();
        Handler capture =
                new Handler() {
                    @Override
                    public void publish(LogRecord record) {
                        records.add(record.getMessage());
                    }

                    @Override
                    public void flush() {}

                    @Override
                    public void close() {}
                };
        jul.addHandler(capture);
        try {
            System.setProperty(PREFIX + "llama-bpe", "regex:.");
            GGUFTokenizerLoader.Builder builder = GGUFTokenizerLoader.createBuilderWithBuiltins();
            builder.aliasPreTokenizer("yi", "llama-bpe");
            builder.build();
        } finally {
            jul.removeHandler(capture);
        }
        assertEquals(1, records.size(), "exactly one replacement line: " + records);
        assertTrue(records.get(0).contains("'llama-bpe'"), records.get(0));
        assertTrue(records.get(0).contains("yi"), records.get(0));
    }

    // ---- eager validation ----

    @Test
    void malformedPropertyValueFailsAtBuildWithTheUsage() {
        System.setProperty(PREFIX + "foo", "banana");
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> GGUFTokenizerLoader.createBuilderWithBuiltins().build());
        assertTrue(e.getMessage().contains("alias:"), e.getMessage());
        assertTrue(e.getMessage().contains("regex:"), e.getMessage());
        assertTrue(e.getMessage().contains("file:"), e.getMessage());
    }

    @Test
    void propertyAliasToAnUnknownNameFailsAtBuildEvenWhenUnselected() {
        System.setProperty(PREFIX + "foo", "alias:no-such-scheme");
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> GGUFTokenizerLoader.createBuilderWithBuiltins().build());
        assertTrue(e.getMessage().contains("no-such-scheme"), e.getMessage());
        assertTrue(e.getMessage().contains("llama-bpe"), e.getMessage());
    }

    @Test
    void uncompilablePropertyRegexFailsAtBuild() {
        System.setProperty(PREFIX + "foo", "regex:([");
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> GGUFTokenizerLoader.createBuilderWithBuiltins().build());
        assertTrue(e.getMessage().contains(PREFIX + "foo"), e.getMessage());
    }

    @Test
    void unreadablePropertyFileFailsAtBuild() {
        System.setProperty(PREFIX + "foo", "file:/nonexistent/patterns.txt");
        assertThrows(
                UncheckedIOException.class,
                () -> GGUFTokenizerLoader.createBuilderWithBuiltins().build());
    }

    // ---- infrastructure ----

    private static void assertChunks(
            GGUFTokenizerLoader.Builder builder, String pre, String text, List<String> expected) {
        List<String> chunks = new ArrayList<>();
        GGUF gguf =
                com.qxotic.format.gguf.Builder.newBuilder()
                        .putString("tokenizer.ggml.model", "stub")
                        .putString("tokenizer.ggml.pre", pre)
                        .build();
        Tokenizer tokenizer =
                builder.registerModelFactory("stub", g -> recordingModel(chunks))
                        .build()
                        .fromGGUF(gguf);
        tokenizer.encode(text);
        assertEquals(expected, chunks);
    }

    /** Splits into one chunk per character: {@code "ab" -> ["a", "b"]}. */
    private static Function<GGUF, Splitter> charByChar() {
        return gguf -> Splitter.regex(Pattern.compile(".", Pattern.UNICODE_CHARACTER_CLASS));
    }

    /** Splits into letter runs (gaps kept): {@code "foo123bar" -> ["foo", "123", "bar"]}. */
    private static Function<GGUF, Splitter> letterRuns() {
        return gguf -> Splitter.regex(Pattern.compile("\\p{L}+", Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static void registerScheme(
            GGUFTokenizerLoader.Builder builder, String name, Function<GGUF, Splitter> splitter) {
        builder.registerPreTokenizer(name, splitter);
        builder.registerNormalizer(name, gguf -> Normalizer.identity());
    }

    /** Records every chunk the pipeline hands to the model, so tests see the splitter at work. */
    private static TokenizationModel recordingModel(List<String> chunks) {
        return new TokenizationModel() {
            @Override
            public Vocabulary vocabulary() {
                return Toknroll.vocabulary("a");
            }

            @Override
            public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
                chunks.add(text.subSequence(start, end).toString());
                out.add(0);
            }

            @Override
            public int countTokens(CharSequence text, int start, int end) {
                return 1;
            }

            @Override
            public int decodeBytesInto(IntSequence tokens, int idx, ByteBuffer out) {
                return 0;
            }

            @Override
            public float expectedTokensPerChar() {
                return 0.5f;
            }
        };
    }
}
