package com.qxotic.jinfer.llm;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * The one place GGUF tokenizer knowledge lives: builds a toknroll {@link Tokenizer} from a GGUF's
 * {@code tokenizer.ggml.*} metadata. Everything above this consumes the container-blind {@link
 * Tokenizer}.
 *
 * <p>WHY CENTRAL, not per-port: {@code tokenizer.ggml.pre} names are orthogonal to {@code
 * general.architecture} - one arch port serves many tokenizer families (Yi-style derivatives of the
 * llama arch each carry their own pre name), so the port that loads a model is not the owner of its
 * tokenizer knowledge. Known names live in toknroll's builtin table (llama.cpp's architecture: one
 * table in llama-vocab.cpp), updated with the core. The {@link #fromGGUF(GGUF,
 * java.util.function.UnaryOperator)} overload covers a port-PRIVATE piece; a novel pre-tokenizer on
 * a shared arch is an upstream-the-regex situation, and the unknown-name error says so loudly.
 */
public final class Tokenizers {

    private static final String OVERRIDE_USAGE =
            "-Djinfer.preTokenizer.<name>=alias:<known-name> to alias a known scheme,"
                    + " =regex:<pattern> to supply one, or =file:<path> with one regex per line"
                    + " (multiple lines = staged split)";

    private static final String OVERRIDE_PREFIX = "jinfer.preTokenizer.";

    private Tokenizers() {}

    public static Tokenizer fromGGUF(GGUF gguf) {
        return fromGGUF(gguf, b -> b);
    }

    /**
     * Whether any {@code -Djinfer.preTokenizer.*} override is configured - the FACT, for callers
     * whose policy depends on it (the AOT preload rebuilds instead of using its preloaded
     * tokenizer, so the escape hatch keeps outranking it). What the overrides do lives in {@link
     * #fromGGUF}.
     */
    public static boolean hasPropertyOverrides() {
        for (String key : System.getProperties().stringPropertyNames()) {
            if (key.startsWith(OVERRIDE_PREFIX)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Refuses a caller-SUPPLIED tokenizer whose id space cannot match this GGUF: the model's
     * embedding rows and the header's stop-token ids are indexed by token id, so a supplied
     * tokenizer may change surface behaviour (regexes, merges, normalization) but never the ids.
     * Same SIZE is the checkable half of that contract; same-ids-for-same-tokens is the caller's
     * oath.
     */
    public static void requireSameIdSpace(GGUF gguf, Tokenizer tokenizer) {
        if (!gguf.containsKey("tokenizer.ggml.tokens")) {
            return; // no vocabulary in the header: nothing checkable
        }
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        if (tokenizer.vocabulary().size() != tokens.length) {
            throw new IllegalArgumentException(
                    "the supplied tokenizer has "
                            + tokenizer.vocabulary().size()
                            + " tokens but this GGUF's vocabulary has "
                            + tokens.length
                            + " - token ids index the embedding table, so this tokenizer cannot"
                            + " serve this model");
        }
    }

    /**
     * As {@link #fromGGUF(GGUF)} with the caller's own registrations - the entry for a model port
     * whose family needs a pre-tokenizer, normalizer or tokenization model the builtins lack. The
     * port owns its load path, so it passes the pieces right here; no registry:
     *
     * <pre>{@code
     * Tokenizers.fromGGUF(gguf, b ->
     *         b.registerPreTokenizer("myfamily", g -> Splitter.regex(MY_PATTERN)));
     * }</pre>
     *
     * <p>Registrations apply after the builtins, so they can override. An unregistered {@code
     * tokenizer.ggml.pre} still fails loudly with the register-it remedy - nothing silently
     * mis-tokenizes.
     */
    public static Tokenizer fromGGUF(
            GGUF gguf,
            java.util.function.UnaryOperator<GGUFTokenizerLoader.Builder> registrations) {
        GGUFTokenizerLoader.Builder builder =
                registrations.apply(GGUFTokenizerLoader.createBuilderWithBuiltins());
        applyPropertyOverrides(builder);
        try {
            return builder.build().fromGGUF(gguf);
        } catch (GGUFTokenizerLoader.UnsupportedPreTokenizerException e) {
            throw new IllegalArgumentException(
                    e.getMessage() + ". Quick fix without code: " + OVERRIDE_USAGE, e);
        }
    }

    /**
     * The end-user escape hatch for a GGUF whose {@code tokenizer.ggml.pre} nobody registered yet:
     * {@code -Djinfer.preTokenizer.<name>=alias:<known-name>} aliases a known scheme (most "new"
     * pre-tokenizers are an existing scheme under a new name), {@code
     * -Djinfer.preTokenizer.<name>=regex:<pattern>} supplies one, and {@code
     * -Djinfer.preTokenizer.<name>=file:<path>} reads patterns from a file - one regex per line,
     * blank lines and {@code #} comments skipped, multiple lines forming a staged {@link
     * Splitter#sequence} (some schemes split digits or CJK first, then the main pattern). Supplied
     * patterns compile with {@link Pattern#UNICODE_CHARACTER_CLASS}, like every builtin, and get an
     * identity normalizer. Applied LAST, so a property can also override a registration; supplied
     * names register before aliases so an alias can target another override. Every override is
     * validated eagerly - a typo'd flag fails the load even when the GGUF never selects it.
     */
    private static void applyPropertyOverrides(GGUFTokenizerLoader.Builder builder) {
        Map<String, String> aliases = new TreeMap<>();
        for (String key : System.getProperties().stringPropertyNames()) {
            if (!key.startsWith(OVERRIDE_PREFIX)) continue;
            String name = key.substring(OVERRIDE_PREFIX.length());
            String value = System.getProperty(key);
            if (value.startsWith("regex:")) {
                registerSupplied(builder, name, List.of(value.substring("regex:".length())));
            } else if (value.startsWith("file:")) {
                Path path = Path.of(value.substring("file:".length()));
                registerSupplied(builder, name, readPatterns(key, path));
            } else if (value.startsWith("alias:")) {
                aliases.put(name, value.substring("alias:".length()));
            } else {
                throw new IllegalArgumentException(
                        "-D" + key + "=" + value + ": the value must be " + OVERRIDE_USAGE);
            }
        }
        // throws with the known names on a typo'd target
        aliases.forEach(builder::aliasPreTokenizer);
    }

    private static List<String> readPatterns(String key, Path path) {
        List<String> patterns;
        try {
            patterns =
                    Files.readAllLines(path).stream()
                            .filter(line -> !line.isBlank() && !line.startsWith("#"))
                            .toList();
        } catch (IOException e) {
            throw new UncheckedIOException("-D" + key + ": cannot read '" + path + "'", e);
        }
        if (patterns.isEmpty()) {
            throw new IllegalArgumentException(
                    "-D"
                            + key
                            + ": '"
                            + path
                            + "' holds no patterns - one regex per line, blank lines and #"
                            + " comments skipped");
        }
        return patterns;
    }

    private static void registerSupplied(
            GGUFTokenizerLoader.Builder builder, String name, List<String> patterns) {
        Splitter splitter =
                Splitter.sequence(
                        patterns.stream()
                                .map(
                                        p ->
                                                Splitter.regex(
                                                        Pattern.compile(
                                                                p,
                                                                Pattern.UNICODE_CHARACTER_CLASS)))
                                .toArray(Splitter[]::new));
        builder.registerPreTokenizer(name, g -> splitter)
                .registerNormalizer(name, g -> Normalizer.identity());
    }

    /** The GGUF's raw Jinja chat-template source, or {@code ""} when it carries none. */
    public static String chatTemplateSource(GGUF gguf) {
        return gguf.getStringOrDefault("tokenizer.chat_template", "");
    }
}
