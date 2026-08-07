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

    private Tokenizers() {}

    /**
     * Tokenizers built ahead of time, keyed by GGUF INSTANCE - the hand-off from whoever baked one
     * (the CLI's AOT preload, which owns the baked artifacts) to the port that will ask for it
     * through {@link #fromGGUF(GGUF)}. Identity is the exact key because the baker passes the very
     * same parsed-header object back at load time. Ports that pass their own registrations use the
     * two-argument overload and never consult this.
     */
    private static final Map<GGUF, Tokenizer> BAKED = new java.util.IdentityHashMap<>();

    /**
     * Registers a tokenizer built ahead of time - by {@link #fromGGUF(GGUF)} over this same {@code
     * gguf}, or the next load mis-tokenizes - so the coming {@code fromGGUF(gguf)} returns it
     * instead of rebuilding vocab, merges and pre-tokenizer regexes.
     */
    public static void preBaked(GGUF gguf, Tokenizer tokenizer) {
        BAKED.put(gguf, tokenizer);
    }

    public static Tokenizer fromGGUF(GGUF gguf) {
        Tokenizer baked = BAKED.get(gguf);
        // a runtime -Djinfer.preTokenizer.* override outranks the bake: the escape hatch exists
        // for exactly the moment a shipped tokenizer turns out wrong
        if (baked != null && !overridesPresent()) {
            return baked;
        }
        return fromGGUF(gguf, b -> b);
    }

    private static boolean overridesPresent() {
        for (String key : System.getProperties().stringPropertyNames()) {
            if (key.startsWith("jinfer.preTokenizer.")) {
                return true;
            }
        }
        return false;
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
        String prefix = "jinfer.preTokenizer.";
        Map<String, String> aliases = new TreeMap<>();
        for (String key : System.getProperties().stringPropertyNames()) {
            if (!key.startsWith(prefix)) continue;
            String name = key.substring(prefix.length());
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
