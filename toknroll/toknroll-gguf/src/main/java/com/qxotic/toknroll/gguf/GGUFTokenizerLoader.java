package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Collectors;

/**
 * Builds Tok'n'Roll tokenizers from GGUF metadata.
 *
 * <p>When loading from Hugging Face or ModelScope, only the GGUF header and metadata key-value
 * pairs are downloaded and cached - tensors and model weights are never fetched.
 *
 * <p>Pre-tokenizer schemes resolve in three layers, each outranking the previous: the builtins
 * ({@link #createBuilderWithBuiltins()}), programmatic registrations on the builder, and the
 * system-property escape hatch {@code -Dtoknroll.gguf.pre.<name>=...}, applied last inside {@link
 * Builder#build()} so a deployment can be fixed without a rebuild. The hatch can also REPLACE a
 * builtin or a code registration of the same name; such replacements are logged. See {@link
 * Builder#build()} for the value grammar.
 */
public final class GGUFTokenizerLoader {
    private static final String SOURCE_HUGGING_FACE = "huggingface";
    private static final String SOURCE_MODELSCOPE = "modelscope";
    private static final String MODEL_LLAMA = "llama";
    private static final String PRE_DEFAULT = "default";

    /** The system-property prefix of the pre-tokenizer escape hatch: {@code toknroll.gguf.pre.}. */
    public static final String OVERRIDE_PREFIX = "toknroll.gguf.pre.";

    private static final String OVERRIDE_USAGE =
            "-D"
                    + OVERRIDE_PREFIX
                    + "<name>=alias:<known-name> to alias a known scheme,"
                    + " =regex:<pattern> to supply one, or =file:<path> with one regex per line"
                    + " (multiple lines = staged split)";

    private static final System.Logger LOG = System.getLogger(GGUFTokenizerLoader.class.getName());

    private static final class Registries {
        private final Map<String, Function<GGUF, TokenizationModel>> modelFactories;
        private final Map<String, Function<GGUF, Normalizer>> normalizers;
        private final Map<String, Function<GGUF, Splitter>> splitters;
        private final Map<String, String> aliases;
        private final Map<String, String> preFallbackByModel;
        private final Map<String, String> normalizerFallbackByModel;

        private Registries(
                Map<String, Function<GGUF, TokenizationModel>> modelFactories,
                Map<String, Function<GGUF, Normalizer>> normalizers,
                Map<String, Function<GGUF, Splitter>> splitters,
                Map<String, String> aliases,
                Map<String, String> preFallbackByModel,
                Map<String, String> normalizerFallbackByModel) {
            this.modelFactories = modelFactories;
            this.normalizers = normalizers;
            this.splitters = splitters;
            this.aliases = aliases;
            this.preFallbackByModel = preFallbackByModel;
            this.normalizerFallbackByModel = normalizerFallbackByModel;
        }
    }

    /**
     * Builder for customizing GGUF tokenizer loading.
     *
     * <p>Register model factories, normalizers, splitters, and pre-tokenizer fallbacks for GGUF
     * model keys not covered by the built-in defaults. Call {@link
     * GGUFTokenizerLoader#createBuilderWithBuiltins()} to start with built-in support and add
     * custom entries, or {@link GGUFTokenizerLoader#createEmptyBuilder()} for full control.
     *
     * <p>A pre-tokenizer name is either CONCRETE (a registered splitter/normalizer pair) or an
     * ALIAS of another name - never both: the latest call for a name decides, so registering
     * concrete factories displaces a pending alias on the same name and vice versa. Aliases are
     * symbolic: they are followed at resolution time, so they can be declared before their target
     * and they track later re-registrations of the target.
     */
    public static final class Builder {
        private final LinkedHashMap<String, Function<GGUF, TokenizationModel>> modelFactories;
        private final LinkedHashMap<String, Function<GGUF, Normalizer>> normalizers;
        private final LinkedHashMap<String, Function<GGUF, Splitter>> splitters;
        private final LinkedHashMap<String, String> aliases;
        private final LinkedHashMap<String, String> preFallbackByModel;
        private final LinkedHashMap<String, String> normalizerFallbackByModel;

        private Builder() {
            this.modelFactories = new LinkedHashMap<>();
            this.normalizers = new LinkedHashMap<>();
            this.splitters = new LinkedHashMap<>();
            this.aliases = new LinkedHashMap<>();
            this.preFallbackByModel = new LinkedHashMap<>();
            this.normalizerFallbackByModel = new LinkedHashMap<>();
        }

        /**
         * Registers a model factory for the given GGUF {@code tokenizer.ggml.model} key. The
         * factory receives the parsed GGUF metadata and must produce a {@link TokenizationModel}.
         *
         * @param key GGUF model key
         * @param factory factory that creates a model from GGUF metadata
         * @return this builder
         */
        public Builder registerModelFactory(String key, Function<GGUF, TokenizationModel> factory) {
            modelFactories.put(
                    GGUFMetadataKeys.normalizeKey(key, "key"),
                    Objects.requireNonNull(factory, "factory"));
            return this;
        }

        /**
         * Registers a normalizer factory for the given GGUF pre-tokenizer key. The factory receives
         * the parsed GGUF metadata. Displaces a pending alias on the same key: a name is either
         * concrete or an alias, and the latest call decides.
         *
         * @param key GGUF pre-tokenizer key
         * @param factory factory that builds a normalizer from GGUF metadata
         * @return this builder
         */
        public Builder registerNormalizer(String key, Function<GGUF, Normalizer> factory) {
            String k = GGUFMetadataKeys.normalizeKey(key, "key");
            normalizers.put(k, Objects.requireNonNull(factory, "factory"));
            aliases.remove(k);
            return this;
        }

        /**
         * Registers a splitter factory for the given GGUF pre-tokenizer key. The factory receives
         * the parsed GGUF metadata. Displaces a pending alias on the same key: a name is either
         * concrete or an alias, and the latest call decides.
         *
         * @param key GGUF pre-tokenizer key
         * @param factory factory that builds a splitter from GGUF metadata
         * @return this builder
         */
        public Builder registerPreTokenizer(String key, Function<GGUF, Splitter> factory) {
            String k = GGUFMetadataKeys.normalizeKey(key, "key");
            splitters.put(k, Objects.requireNonNull(factory, "factory"));
            aliases.remove(k);
            return this;
        }

        /**
         * Aliases {@code key} to {@code targetKey}: the target's splitter and normalizer serve the
         * alias. For GGUFs whose {@code tokenizer.ggml.pre} names a scheme identical to a known
         * one.
         *
         * <p>The alias is SYMBOLIC - a name followed at resolution time, not a copy of the target's
         * factories: it may be declared before its target is registered, and it tracks later
         * re-registrations of the target (a property override of the target redirects every alias
         * pointing at it). Chains are allowed; cycles and dangling targets fail {@link #build()}
         * with the chain or the known names in the message.
         *
         * <p>Displaces concrete splitter/normalizer registrations on the same key: a name is either
         * concrete or an alias, and the latest call decides.
         */
        public Builder aliasPreTokenizer(String key, String targetKey) {
            String k = GGUFMetadataKeys.normalizeKey(key, "key");
            String t = GGUFMetadataKeys.normalizeKey(targetKey, "targetKey");
            splitters.remove(k);
            normalizers.remove(k);
            aliases.put(k, t);
            return this;
        }

        Builder registerPreFallback(String modelKey, String preKey) {
            preFallbackByModel.put(
                    GGUFMetadataKeys.normalizeKey(modelKey, "modelKey"),
                    GGUFMetadataKeys.normalizeKey(preKey, "preKey"));
            return this;
        }

        Builder registerNormalizerFallback(String modelKey, String normalizerKey) {
            normalizerFallbackByModel.put(
                    GGUFMetadataKeys.normalizeKey(modelKey, "modelKey"),
                    GGUFMetadataKeys.normalizeKey(normalizerKey, "normalizerKey"));
            return this;
        }

        /**
         * Applies the system-property escape hatch, then finishes configuration and returns a
         * ready-to-use loader.
         *
         * <p>Every {@code -Dtoknroll.gguf.pre.<name>=...} property is applied AFTER builtins and
         * programmatic registrations, so it can also replace them:
         *
         * <ul>
         *   <li>{@code =alias:<known-name>} - the name follows the target, like {@link
         *       #aliasPreTokenizer(String, String)}
         *   <li>{@code =regex:<pattern>} - supplies a splitter (with an identity normalizer),
         *       compiled with {@link Pattern#UNICODE_CHARACTER_CLASS} like every builtin
         *   <li>{@code =file:<path>} - one regex per line, blank lines and {@code #} comments
         *       skipped, multiple lines forming a staged {@link Splitter#sequence} (some schemes
         *       split digits or CJK first, then the main pattern)
         * </ul>
         *
         * <p>Every property is validated eagerly - a malformed value, an uncompilable regex, an
         * unreadable file, or an alias to an unknown name fails the build even when no GGUF ever
         * selects it. A property that REPLACES a registered name is logged, naming the aliases that
         * follow it. Finally every alias is validated: cycles and dangling targets fail here, never
         * mid-load.
         */
        public GGUFTokenizerLoader build() {
            applyPropertyOverrides();
            validateAliases(aliases, splitters);
            Registries registries =
                    new Registries(
                            Map.copyOf(modelFactories),
                            Map.copyOf(normalizers),
                            Map.copyOf(splitters),
                            Map.copyOf(aliases),
                            Map.copyOf(preFallbackByModel),
                            Map.copyOf(normalizerFallbackByModel));
            return new GGUFTokenizerLoader(registries);
        }

        private void applyPropertyOverrides() {
            List<String> keys = new ArrayList<>();
            for (String key : System.getProperties().stringPropertyNames()) {
                if (key.startsWith(OVERRIDE_PREFIX)) {
                    keys.add(key);
                }
            }
            Collections.sort(keys); // determinism only; symbolic aliases make order irrelevant
            for (String key : keys) {
                String name =
                        GGUFMetadataKeys.normalizeKey(
                                key.substring(OVERRIDE_PREFIX.length()), "name in -D" + key);
                String value = System.getProperty(key);
                if (value == null) {
                    continue; // cleared between listing and reading
                }
                value = value.trim();
                if (value.startsWith("alias:")) {
                    String target =
                            GGUFMetadataKeys.normalizeKey(
                                    value.substring("alias:".length()).trim(),
                                    "alias target in -D" + key);
                    boolean replaced = splitters.remove(name) != null;
                    replaced |= normalizers.remove(name) != null;
                    aliases.put(name, target);
                    if (replaced) {
                        logReplacement(key, value, name);
                    }
                } else if (value.startsWith("regex:") || value.startsWith("file:")) {
                    List<String> patterns =
                            value.startsWith("regex:")
                                    ? List.of(value.substring("regex:".length()))
                                    : readPatterns(key, value.substring("file:".length()));
                    validatePatterns(key, patterns);
                    String[] captured = patterns.toArray(String[]::new);
                    boolean replaced = splitters.containsKey(name) || normalizers.containsKey(name);
                    aliases.remove(name);
                    splitters.put(name, gguf -> stagedSplitter(captured));
                    normalizers.put(name, gguf -> Normalizer.identity());
                    if (replaced) {
                        logReplacement(key, value, name);
                    }
                } else {
                    throw new IllegalArgumentException(
                            "-D" + key + "=" + value + ": the value must be " + OVERRIDE_USAGE);
                }
            }
        }

        /**
         * Names every alias whose chain passes through {@code name}: a replacement of {@code name}
         * silently redirects all of them, so the log line must say who moved.
         */
        private void logReplacement(String key, String value, String name) {
            List<String> followers = new ArrayList<>();
            for (Map.Entry<String, String> alias : aliases.entrySet()) {
                LinkedHashSet<String> seen = new LinkedHashSet<>();
                String current = alias.getKey();
                while (seen.add(current)) {
                    String target = aliases.get(current);
                    if (target == null) {
                        break;
                    }
                    if (target.equals(name)) {
                        followers.add(alias.getKey());
                        break;
                    }
                    current = target;
                }
            }
            LOG.log(
                    System.Logger.Level.WARNING,
                    "-D"
                            + key
                            + "="
                            + value
                            + " replaces the registered pre-tokenizer '"
                            + name
                            + "'"
                            + (followers.isEmpty()
                                    ? ""
                                    : " - aliases following it: " + String.join(", ", followers)));
        }
    }

    private final Registries registries;

    private GGUFTokenizerLoader(Registries registries) {
        this.registries = registries;
    }

    /** Creates a builder with no pre-registered factories. */
    public static Builder createEmptyBuilder() {
        return new Builder();
    }

    /**
     * Creates a builder pre-loaded with built-in model factories, normalizers, and splitters for
     * common GGUF models.
     */
    public static Builder createBuilderWithBuiltins() {
        Builder builder = createEmptyBuilder();
        GGUFTokenizerDefaults.applyTo(builder);
        return builder;
    }

    /**
     * Builds a tokenizer from a local GGUF file.
     *
     * @param ggufFile path to a {@code .gguf} file
     * @return loaded tokenizer
     * @throws IllegalArgumentException if the path does not exist or is not a GGUF file
     * @throws TokenizerLoadException if file I/O fails
     */
    public Tokenizer fromLocal(Path ggufFile) {
        Objects.requireNonNull(ggufFile, "ggufFile");
        Path file = ggufFile.toAbsolutePath().normalize();
        if (!Files.exists(file)) {
            throw new IllegalArgumentException("Path does not exist: " + file);
        }
        if (Files.isDirectory(file)) {
            throw new IllegalArgumentException("Expected GGUF file path, got directory: " + file);
        }
        if (!file.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            throw new IllegalArgumentException("Expected .gguf file path, got: " + file);
        }

        try {
            GGUF gguf = GGUF.read(file);
            return fromGGUF(gguf);
        } catch (IOException e) {
            throw new TokenizerLoadException(
                    "[local] Failed to load GGUF tokenizer from " + file, e);
        }
    }

    /**
     * Fetches a GGUF file from Hugging Face and builds a tokenizer. Uses the default branch,
     * downloads if not cached, and does not force-refresh.
     *
     * @param user repository owner/namespace on HuggingFace
     * @param repository repository name on HuggingFace
     * @param ggufPath path to the GGUF file within the repository
     * @return loaded tokenizer
     */
    public Tokenizer fromHuggingFace(String user, String repository, String ggufPath) {
        return fromHuggingFace(user, repository, null, ggufPath, false, false);
    }

    /**
     * Fetches a GGUF file from Hugging Face with full parameter control.
     *
     * @param user repository owner/namespace on HuggingFace
     * @param repository repository name on HuggingFace
     * @param ggufPath path to the GGUF file within the repository
     * @param useCacheOnly if {@code true}, does not fetch over the network
     * @param forceRefresh if {@code true}, ignores cached data and re-fetches
     * @return loaded tokenizer
     * @throws TokenizerLoadException if remote fetch or file I/O fails
     */
    public Tokenizer fromHuggingFace(
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh) {
        return fromRemote(
                SOURCE_HUGGING_FACE,
                user,
                repository,
                revision,
                ggufPath,
                useCacheOnly,
                forceRefresh);
    }

    /**
     * Fetches a GGUF file from ModelScope and builds a tokenizer. Uses the default branch,
     * downloads if not cached, and does not force-refresh.
     *
     * @param user repository owner/namespace on ModelScope
     * @param repository repository name on ModelScope
     * @param ggufPath path to the GGUF file within the repository
     * @return loaded tokenizer
     */
    public Tokenizer fromModelScope(String user, String repository, String ggufPath) {
        return fromModelScope(user, repository, null, ggufPath, false, false);
    }

    /**
     * Fetches a GGUF file from ModelScope with full parameter control.
     *
     * @param user repository owner/namespace on ModelScope
     * @param repository repository name on ModelScope
     * @param ggufPath path to the GGUF file within the repository
     * @param useCacheOnly if {@code true}, does not fetch over the network
     * @param forceRefresh if {@code true}, ignores cached data and re-fetches
     * @return loaded tokenizer
     * @throws TokenizerLoadException if remote fetch or file I/O fails
     */
    public Tokenizer fromModelScope(
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh) {
        return fromRemote(
                SOURCE_MODELSCOPE,
                user,
                repository,
                revision,
                ggufPath,
                useCacheOnly,
                forceRefresh);
    }

    private Tokenizer fromRemote(
            String source,
            String user,
            String repository,
            String revision,
            String ggufPath,
            boolean useCacheOnly,
            boolean forceRefresh) {
        GGUFMetadataCache cache = GGUFMetadataCache.create();
        try {
            Path metadataPath;
            if (SOURCE_HUGGING_FACE.equals(source)) {
                metadataPath =
                        cache.fetchHuggingFace(
                                user, repository, revision, ggufPath, useCacheOnly, forceRefresh);
            } else if (SOURCE_MODELSCOPE.equals(source)) {
                metadataPath =
                        cache.fetchModelScope(
                                user, repository, revision, ggufPath, useCacheOnly, forceRefresh);
            } else {
                throw new IllegalArgumentException("Unsupported source: " + source);
            }
            GGUF gguf = GGUF.read(metadataPath);
            return fromGGUF(gguf);
        } catch (IOException e) {
            throw new TokenizerLoadException(
                    "["
                            + source
                            + "] Failed to load GGUF tokenizer for "
                            + user
                            + "/"
                            + repository
                            + "@"
                            + String.valueOf(revision)
                            + "#"
                            + ggufPath,
                    e);
        }
    }

    /**
     * Builds a tokenizer from a pre-parsed {@code GGUF} instance.
     *
     * @param gguf pre-parsed GGUF metadata container
     * @return loaded tokenizer
     * @throws IllegalArgumentException if the GGUF model key is unsupported
     */
    public Tokenizer fromGGUF(GGUF gguf) {
        String modelKey = GGUFMetadataKeys.requireKey(gguf, GGUFMetadataKeys.MODEL);
        String preKey = resolvePreTokenizerKey(gguf, modelKey);
        String normalizerKey = resolveNormalizerKey(gguf, modelKey);

        Function<GGUF, TokenizationModel> modelFactory = registries.modelFactories.get(modelKey);
        if (modelFactory == null) {
            throw new IllegalArgumentException(
                    "Unsupported GGUF tokenizer model '"
                            + modelKey
                            + "' (supported: "
                            + sortedKeys(registries.modelFactories.keySet())
                            + ")");
        }

        TokenizationModel model = modelFactory.apply(gguf);

        if (preKey == null) {
            throw new IllegalArgumentException(
                    "No pre-tokenizer key resolved for model '"
                            + modelKey
                            + "' (explicit pre-tokenizer key '"
                            + GGUFMetadataKeys.PRE
                            + "' is absent and no pre-tokenizer fallback is registered)");
        }
        Function<GGUF, Splitter> splitterFactory =
                registries.splitters.get(resolveAlias(registries.aliases, preKey));
        if (splitterFactory == null) {
            throw new UnsupportedPreTokenizerException(
                    "Unsupported GGUF pre-tokenizer key '"
                            + preKey
                            + "' for model '"
                            + modelKey
                            + "'. Quick fix without a rebuild: "
                            + OVERRIDE_USAGE.replace("<name>", preKey)
                            + ". Known schemes: "
                            + sortedKeys(registries.splitters.keySet())
                            + ". To fix in code: registerPreTokenizer(\""
                            + preKey
                            + "\", ...)");
        }
        Splitter splitter = splitterFactory.apply(gguf);

        if (normalizerKey == null) {
            throw new IllegalArgumentException(
                    "No normalizer key resolved for model '"
                            + modelKey
                            + "' (explicit pre-tokenizer key '"
                            + GGUFMetadataKeys.PRE
                            + "' is absent and no normalizer fallback is registered)");
        }
        Function<GGUF, Normalizer> normalizerFactory =
                registries.normalizers.get(resolveAlias(registries.aliases, normalizerKey));
        if (normalizerFactory == null) {
            throw new IllegalArgumentException(
                    "Unsupported GGUF normalizer key '"
                            + normalizerKey
                            + "' for model '"
                            + modelKey
                            + "'. Register it via registerNormalizer(...)");
        }
        Normalizer normalizer = normalizerFactory.apply(gguf);

        Tokenizer tokenizer = Toknroll.pipeline(normalizer, splitter, model);

        // SPM models with metaspace normalization need decode wrapping.
        if (PRE_DEFAULT.equals(preKey) && MODEL_LLAMA.equals(modelKey)) {
            tokenizer = wrapSentencePieceDecode(tokenizer);
        }

        return tokenizer;
    }

    /**
     * An unknown {@code tokenizer.ggml.pre} name - either at resolution time or as an alias target.
     * A distinct type so callers can attach their own remedy without matching on message text; the
     * message already carries the no-rebuild remedy, the {@code toknroll.gguf.pre.*} system
     * property.
     */
    public static final class UnsupportedPreTokenizerException extends IllegalArgumentException {
        UnsupportedPreTokenizerException(String message) {
            super(message);
        }
    }

    /**
     * Follows the alias chain from {@code key} to its terminal name. Cycles are normally caught at
     * {@link Builder#build()}; the check here keeps a corrupted registry from looping forever.
     */
    private static String resolveAlias(Map<String, String> aliases, String key) {
        String current = key;
        LinkedHashSet<String> seen = null; // lazy: most names are not aliases
        while (true) {
            String target = aliases.get(current);
            if (target == null) {
                return current;
            }
            if (seen == null) {
                seen = new LinkedHashSet<>();
            }
            if (!seen.add(current)) {
                StringBuilder chain = new StringBuilder();
                for (String step : seen) {
                    chain.append('\'').append(step).append("' -> ");
                }
                throw new IllegalArgumentException(
                        "pre-tokenizer alias cycle: " + chain + "'" + current + "'");
            }
            current = target;
        }
    }

    /** Every alias must resolve to a registered splitter - checked eagerly, even if unselected. */
    private static void validateAliases(Map<String, String> aliases, Map<String, ?> splitters) {
        for (String name : aliases.keySet()) {
            String terminal = resolveAlias(aliases, name);
            if (!splitters.containsKey(terminal)) {
                throw new UnsupportedPreTokenizerException(
                        "pre-tokenizer alias '"
                                + name
                                + "' points at unknown pre-tokenizer '"
                                + terminal
                                + "' (supported: "
                                + sortedKeys(splitters.keySet())
                                + ")");
            }
        }
    }

    private static List<String> readPatterns(String key, String fileValue) {
        Path path = Path.of(fileValue);
        List<String> patterns;
        try {
            patterns =
                    Files.readAllLines(path).stream()
                            .filter(line -> !line.isBlank() && !line.startsWith("#"))
                            .collect(Collectors.toList());
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

    /**
     * Compiles every pattern now, so a typo fails the build even when the GGUF never selects it.
     */
    private static void validatePatterns(String key, List<String> patterns) {
        for (String pattern : patterns) {
            try {
                Pattern.compile(pattern, Pattern.UNICODE_CHARACTER_CLASS);
            } catch (PatternSyntaxException e) {
                throw new IllegalArgumentException(
                        "-D" + key + ": invalid regex '" + pattern + "': " + e.getMessage(), e);
            }
        }
    }

    /**
     * The factory captures pattern STRINGS, never compiled {@link Pattern}s - like the builtin
     * table: patterns compile once per load, and a native image bakes only strings into its heap.
     */
    private static Splitter stagedSplitter(String[] patterns) {
        Splitter[] stages = new Splitter[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            stages[i] =
                    Splitter.regex(Pattern.compile(patterns[i], Pattern.UNICODE_CHARACTER_CLASS));
        }
        return Splitter.sequence(stages);
    }

    private static String sortedKeys(Collection<String> keys) {
        return String.join(", ", new TreeSet<>(keys));
    }

    private static Tokenizer wrapSentencePieceDecode(Tokenizer base) {
        return ImplAccessor.sentencePieceDecodeWrapper(base, true);
    }

    private String resolvePreTokenizerKey(GGUF gguf, String modelKey) {
        String explicitPre = GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.PRE);
        if (explicitPre != null) {
            return explicitPre;
        }
        return registries.preFallbackByModel.get(modelKey);
    }

    private String resolveNormalizerKey(GGUF gguf, String modelKey) {
        String explicitPre = GGUFMetadataKeys.key(gguf, GGUFMetadataKeys.PRE);
        if (explicitPre != null) {
            return explicitPre;
        }
        return registries.normalizerFallbackByModel.get(modelKey);
    }
}
