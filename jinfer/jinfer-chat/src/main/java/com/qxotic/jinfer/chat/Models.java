package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.jinfer.telemetry.ModelLoadEvent;
import com.qxotic.jinfer.telemetry.Telemetry;
import com.qxotic.toknroll.Tokenizer;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Loads any generative model, dispatching on {@code general.architecture} to the matching port via
 * {@link ModelProvider} services - the ports on the classpath define what is loadable. The one
 * "path to model" entry every consumer (server, CLI, benches) shares.
 */
public final class Models {

    private static final System.Logger LOG = System.getLogger("jinfer.models");

    private Models() {}

    // The in-repo provider classes, for the SHADING fallback below and its drift test. Absent
    // classes are silently skipped, so this list is a superset of any given classpath.
    private static final List<String> KNOWN_PROVIDER_CLASSES =
            List.of(
                    "com.qxotic.jinfer.models.llama.LlamaProvider",
                    "com.qxotic.jinfer.models.llama.GraniteProvider",
                    "com.qxotic.jinfer.models.gemma4.Gemma4Provider",
                    "com.qxotic.jinfer.models.lfm2.Lfm2Provider",
                    "com.qxotic.jinfer.models.qwen35.Qwen35Provider",
                    "com.qxotic.jinfer.models.nemotronh.NemotronHProvider",
                    "com.qxotic.jinfer.models.gptoss.GptOssProvider",
                    "com.qxotic.jinfer.models.inflect2.Inflect2Provider");

    private static final List<ModelProvider> PROVIDERS = discover();

    /**
     * ServiceLoader first - the contract. When it finds NOTHING, the likeliest cause is a shaded
     * jar built without merging {@code META-INF/services} (the provider CLASSES survive shading;
     * only the registration files get dropped), so fall back to the known class names and warn: the
     * setup works, but the build should add the transformer before relocation breaks it too.
     */
    private static List<ModelProvider> discover() {
        List<ModelProvider> loaded =
                ServiceLoader.load(ModelProvider.class).stream()
                        .map(ServiceLoader.Provider::get)
                        .toList();
        if (!loaded.isEmpty()) return loaded;
        List<ModelProvider> recovered = new java.util.ArrayList<>();
        for (String name : KNOWN_PROVIDER_CLASSES) {
            try {
                recovered.add(
                        (ModelProvider) Class.forName(name).getDeclaredConstructor().newInstance());
            } catch (ReflectiveOperationException | LinkageError absent) {
                // not on this classpath - fine, the list is a superset
            }
        }
        if (!recovered.isEmpty()) {
            LOG.log(
                    System.Logger.Level.WARNING,
                    "ServiceLoader found no model providers but {0} provider class(es) are"
                            + " present - your build likely shades jinfer without merging"
                            + " META-INF/services (Maven Shade: ServicesResourceTransformer)."
                            + " Recovered them reflectively; fix the build, this fallback cannot"
                            + " see third-party providers.",
                    recovered.size());
        }
        return List.copyOf(recovered);
    }

    /** Package-visible for the drift test: the fallback list must cover every classpath port. */
    static List<String> knownProviderClasses() {
        return KNOWN_PROVIDER_CLASSES;
    }

    /**
     * Loads {@code path}. Nothing here is sized by context - a state's size is chosen when the
     * state is allocated, not when the weights are mapped. Weights map into {@code arena}: who
     * provides the arena owns the weights' lifetime ({@code ofAuto} = GC-managed, {@code global} =
     * process, a scoped arena = deterministic - it must outlive every model sharing the weights,
     * and closing it while any computation runs is a crash, not an exception).
     */
    public static LoadedModel<?> load(Path path, Arena arena) throws IOException {
        return load(path, arena, Map.of());
    }

    /**
     * The one-argument bliss form: GC-managed weights ({@code Arena.ofAuto()} - freed when the
     * model becomes unreachable). Reach for the explicit overload when you need a deterministic
     * weight lifetime.
     */
    public static LoadedModel<?> load(Path path) throws IOException {
        return load(path, Arena.ofAuto());
    }

    /**
     * Load with COMPANIONS: auxiliary files that give the model a capability it does not have on
     * its own, keyed by capability name - {@code "media"} for a projector, {@code "speculation"}
     * for a draft head. What each architecture accepts is {@link #companionFiles(Path)}.
     *
     * <p>Values are PATHS to single FILES, never references: resolving a reference is a
     * downloader's job, done before this call, so no library load ever touches the network.
     */
    public static LoadedModel<?> load(Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        return load(path, arena, companions, null);
    }

    /**
     * As {@link #load(Path, Arena, Map)} with the caller's OWN tokenizer instead of the one the
     * GGUF describes - a patched pre-tokenizer, a custom implementation, or one built ahead of time
     * so the load skips the vocab/merges/regex build (the AOT preload). Null means the GGUF's own,
     * which is the norm; a supplied tokenizer must keep the GGUF's token-id space ({@link
     * Tokenizers#requireSameIdSpace}), because the embedding table and the stop-token ids are
     * indexed by id.
     */
    public static LoadedModel<?> load(
            Path path, Arena arena, Map<String, Path> companions, Tokenizer tokenizer)
            throws IOException {
        return open(path, (fc, gguf) -> load(fc, gguf, arena, companions, tokenizer));
    }

    /**
     * The capabilities {@code path}'s architecture can gain from a companion, and the filename that
     * carries each - the GGUF header only, no weights. A caller uses it to reject a capability this
     * architecture does not have BEFORE fetching anything for it, and to say what the file is
     * usually called when it does.
     */
    public static Map<String, String> companionFiles(Path path) throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).companionFiles());
    }

    /** A companion this architecture does not have is a mistake, not something to ignore. */
    private static void requireAccepted(
            ModelProvider provider, GGUF gguf, java.util.Set<String> capabilities) {
        Map<String, String> accepted = provider.companionFiles();
        for (String capability : capabilities) {
            if (!accepted.containsKey(capability)) {
                throw new IllegalArgumentException(
                        "'"
                                + gguf.getString("general.architecture")
                                + "' has no '"
                                + capability
                                + "' capability. It offers: "
                                + (accepted.isEmpty()
                                        ? "none"
                                        : new java.util.TreeSet<>(accepted.keySet())));
            }
        }
    }

    /**
     * Re-roots the cache seed with EVERY ATTACHED COMPANION, plus the image decoder and the model's
     * preprocessing plan. Media blocks are content-keyed by their SOURCE bytes (see {@code
     * Batch.embeddings}), so everything standing between those bytes and the stored KV must be part
     * of the key space: a different projector producing different rows for the same image must
     * never be served blocks cached under the old one.
     */
    /**
     * Attaches the effective sampling recommendations: the GGUF's {@code general.sampling.*} where
     * present, falling back to the port's model-author recommendation, if it declared one.
     */
    private static <S extends com.qxotic.jinfer.RuntimeState> LoadedModel<S> sampled(
            LoadedModel<S> loaded, GGUF gguf) {
        return loaded.withSamplingDefaults(
                LoadedModel.SamplingDefaults.fromGGUF(gguf)
                        .withFallback(loaded.samplingDefaults()));
    }

    static <S extends com.qxotic.jinfer.RuntimeState> LoadedModel<S> companionSeeded(
            LoadedModel<S> loaded, Map<String, Path> companions) {
        if (companions.isEmpty()) {
            return loaded;
        }
        try {
            java.security.MessageDigest sha = java.security.MessageDigest.getInstance("SHA-256");
            sha.update(loaded.seed());
            // sorted, so the seed does not depend on the order a caller listed them in
            for (var companion : new java.util.TreeMap<>(companions).entrySet()) {
                sha.update(companion.getKey().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                sha.update(com.qxotic.jinfer.cache.PromptCache.modelSeed(companion.getValue()));
            }
            sha.update(
                    com.qxotic.jinfer.media.ImageCodec.decoder()
                            .name()
                            .getBytes(java.nio.charset.StandardCharsets.UTF_8));
            if (loaded.model() instanceof com.qxotic.jinfer.MultiModal mm) {
                sha.update(mm.encodePlanId().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
            return new LoadedModel<>(
                    loaded.model(),
                    loaded.tokenizer(),
                    loaded.chatTemplateSource(),
                    loaded.stopTokens(),
                    sha.digest(),
                    loaded.template(),
                    loaded.samplingDefaults());
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * As {@link #load(Path, Arena)} but reusing an already-parsed {@code gguf} (the header is not
     * re-read) - used by AOT preload. {@code fileChannel} supplies the tensor data to mmap.
     */
    public static LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, Arena arena)
            throws IOException {
        return load(fileChannel, gguf, arena, null);
    }

    /**
     * As {@link #load(FileChannel, GGUF, Arena)} with the caller's own tokenizer - see {@link
     * #load(Path, Arena, Map, Tokenizer)} for the contract. The AOT preload's entry: header and
     * tokenizer come baked, only the tensor data is read.
     */
    public static LoadedModel<?> load(
            FileChannel fileChannel, GGUF gguf, Arena arena, Tokenizer tokenizer)
            throws IOException {
        return load(fileChannel, gguf, arena, Map.of(), tokenizer);
    }

    /**
     * As {@link #load(FileChannel, GGUF, Arena, Tokenizer)} with COMPANIONS - the preload's entry
     * when a parsed header and companions are both in hand. Companion files join the cache seed
     * exactly as on the path entry; each port parses its own companion's header (~10 ms), which is
     * the port's business and nobody else's.
     */
    public static LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        // copyOf: immutable for the port's lifetime, and it rejects a null capability or path at
        // the boundary rather than inside a loader that cannot say which entry was wrong
        Map<String, Path> attached = Map.copyOf(companions);
        ModelProvider provider = provider(gguf);
        requireAccepted(provider, gguf, attached.keySet());
        if (tokenizer != null) {
            Tokenizers.requireSameIdSpace(gguf, tokenizer);
        }
        return companionSeeded(
                sampled(provider.load(fileChannel, gguf, arena, attached, tokenizer), gguf),
                attached);
    }

    /**
     * Loads an embedding model (an {@code EmbeddingModel} port, e.g. Qwen3-Embedding); same
     * architecture dispatch as {@link #load}. Generative-only architectures fail with a clear
     * {@link UnsupportedOperationException}.
     */
    public static LoadedEmbedder<?> loadEmbedder(Path path, Arena arena) throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).loadEmbedder(fc, gguf, path, arena));
    }

    /**
     * Loads a RERANKER (a cross-encoder recipe over a port's backbone, e.g. Qwen3-Reranker); same
     * architecture dispatch as {@link #load}. Architectures with no reranker recipe fail with a
     * clear {@link UnsupportedOperationException}. The GGUF must be the reranker of its family - an
     * architecture cannot tell its reranker and embedder apart, so a wrong file scores rather than
     * refuses.
     */
    public static LoadedReranker<?> loadReranker(Path path, Arena arena) throws IOException {
        return open(path, (fc, gguf) -> provider(gguf).loadReranker(fc, gguf, path, arena));
    }

    /**
     * Loads a SPEECH model (a {@code SpeechModel} port, e.g. Inflect2); same architecture dispatch
     * as {@link #load}. Non-speech architectures fail with a clear {@link
     * UnsupportedOperationException}.
     *
     * <p>Defaults only. Anything a port lets you tune - a lexicon, a language, a family's own knobs
     * - lives on that port's own loader, typed; this is the entry for a caller that must not name
     * the port.
     */
    public static com.qxotic.jinfer.SpeechModel<?, ?, ?> loadSpeech(Path path, Arena arena)
            throws IOException {
        return loadSpeech(path, arena, Map.of());
    }

    /**
     * As {@link #loadSpeech(Path, Arena)} with COMPANIONS - {@code "phonemes"} for a pronunciation
     * lexicon. A port's own discovery (a lexicon beside the GGUF, then the classpath, then an
     * external tool) remains the DEFAULT; naming one here overrides that ladder rather than
     * extending it.
     */
    public static com.qxotic.jinfer.SpeechModel<?, ?, ?> loadSpeech(
            Path path, Arena arena, Map<String, Path> companions) throws IOException {
        Map<String, Path> attached = Map.copyOf(companions);
        return open(
                path,
                (fc, gguf) -> {
                    ModelProvider provider = provider(gguf);
                    requireAccepted(provider, gguf, attached.keySet());
                    return provider.loadSpeech(fc, gguf, path, arena, attached);
                });
    }

    private interface Load<T> {
        T apply(FileChannel fc, GGUF gguf) throws IOException;
    }

    /**
     * Opens {@code path}, reads the GGUF header, and hands both to {@code load}. The single
     * chokepoint for every architecture and every model kind, so it is also where {@code
     * jinfer.ModelLoad} is emitted and the sampled events are installed. A port loaded directly
     * (e.g. {@code Gemma4.loadModel}) bypasses this and reports nothing.
     */
    private static <T> T open(Path path, Load<T> load) throws IOException {
        Telemetry.install();
        if (!Files.exists(path)) {
            throw new java.nio.file.NoSuchFileException(
                    path.toString(), null, "model file not found");
        }
        ModelLoadEvent event = new ModelLoadEvent();
        event.begin();
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            fc.position(0L);
            GGUF gguf;
            try {
                gguf =
                        GGUF.read(
                                Channels.newChannel(
                                        new BufferedInputStream(
                                                Channels.newInputStream(fc), 1 << 20)));
            } catch (com.qxotic.format.gguf.GGUFFormatException e) {
                throw new IllegalArgumentException(
                        path
                                + " is not a GGUF model file ("
                                + e.getMessage()
                                + "). If this is a HuggingFace checkpoint (safetensors/pytorch),"
                                + " convert it with llama.cpp's convert_hf_to_gguf.py",
                        e);
            }
            // a SPLIT part carries only its own slice of the tensors; loading one alone would
            // build a silently WRONG model (missing weights) - refuse with the remedy instead
            long splitCount = metadataLong(gguf, "split.count");
            if (splitCount > 1) {
                throw new UnsupportedOperationException(
                        path.getFileName()
                                + " is part "
                                + (metadataLong(gguf, "split.no") + 1)
                                + " of a "
                                + splitCount
                                + "-file split GGUF - split models are not supported yet; merge the"
                                + " parts first: llama.cpp's llama-gguf-split --merge <part1>"
                                + " <out>");
            }
            T loaded = load.apply(fc, gguf);
            if (event.isEnabled()) {
                String arch = gguf.getString("general.architecture");
                event.model = path.getFileName().toString();
                event.architecture = arch;
                event.contextLength =
                        gguf.getValueOrDefault(int.class, arch + ".context_length", 0);
                event.dimensions = gguf.getValueOrDefault(int.class, arch + ".embedding_length", 0);
                event.weightsBytes = Files.size(path);
                event.mapped = true; // jinfer maps tensor data; it never reads it into the heap
            }
            return loaded;
        } finally {
            event.end();
            event.commit();
        }
    }

    /**
     * The architectures the classpath's ports claim (sorted) - what {@link #load} can dispatch. For
     * tooling and startup banners; a port that does not enumerate (default {@link
     * ModelProvider#architectures()}) dispatches fine but is absent here.
     */
    public static java.util.SortedSet<String> supportedArchitectures() {
        java.util.TreeSet<String> archs = new java.util.TreeSet<>();
        for (ModelProvider p : PROVIDERS) archs.addAll(p.architectures());
        return archs;
    }

    // arch (or prefix, for pattern-matching ports) -> the Maven artifact that provides it.
    // DIAGNOSTICS ONLY - dispatch never reads this; it exists so "unsupported architecture"
    // can name the jar to add. Kept in sync with the in-repo ports by hand; an entry for a
    // port the user has is harmless (dispatch already succeeded).
    private static final java.util.Map<String, String> PORT_ARTIFACTS =
            java.util.Map.ofEntries(
                    java.util.Map.entry("gemma4", "com.qxotic:jinfer-gemma4"),
                    java.util.Map.entry("gpt-oss", "com.qxotic:jinfer-gptoss"),
                    java.util.Map.entry("lfm", "com.qxotic:jinfer-lfm2"),
                    java.util.Map.entry("llama", "com.qxotic:jinfer-llama"),
                    java.util.Map.entry("granite", "com.qxotic:jinfer-llama"),
                    java.util.Map.entry("minicpm", "com.qxotic:jinfer-llama"),
                    java.util.Map.entry("mistral3", "com.qxotic:jinfer-llama"),
                    java.util.Map.entry("smollm3", "com.qxotic:jinfer-llama"),
                    java.util.Map.entry("nemotron_h", "com.qxotic:jinfer-nemotronh"),
                    java.util.Map.entry("qwen35", "com.qxotic:jinfer-qwen35"),
                    java.util.Map.entry("inflect", "com.qxotic:jinfer-inflect2"));

    /**
     * The provider for {@code arch} among {@code providers}, or null: highest {@link
     * ModelProvider#priority()} wins (a third-party override REPLACES a bundled port by declaring a
     * higher value); equal priorities resolve deterministically by class name - never by classpath
     * order - and warn, so an accidental duplicate is visible without breaking the
     * deliberate-override case. Package-visible for its unit test.
     */
    static ModelProvider select(List<ModelProvider> providers, String arch) {
        ModelProvider best = null;
        ModelProvider contender = null; // an equal-priority rival of best, for the warning
        for (ModelProvider p : providers) {
            if (!p.supports(arch)) continue;
            if (best == null) {
                best = p;
            } else if (p.priority() > best.priority()) {
                best = p;
                contender = null;
            } else if (p.priority() == best.priority()) {
                // deterministic tie-break: class name order, not jar order
                if (p.getClass().getName().compareTo(best.getClass().getName()) < 0) {
                    contender = best;
                    best = p;
                } else {
                    contender = p;
                }
            }
        }
        if (contender != null) {
            LOG.log(
                    System.Logger.Level.WARNING,
                    "architecture ''{0}'' is claimed by both {1} (selected, deterministic by"
                            + " class name) and {2} at equal priority - override"
                            + " ModelProvider.priority() on the one that should win",
                    arch,
                    best.getClass().getName(),
                    contender.getClass().getName());
        }
        return best;
    }

    /**
     * The diagnostics table's answer for {@code arch}, or null - package-visible for the drift test
     * (every classpath port's architecture must resolve here, so a new port cannot land without its
     * remedy entry).
     */
    static String artifactFor(String arch) {
        for (var e : PORT_ARTIFACTS.entrySet()) {
            if (arch.equals(e.getKey()) || arch.startsWith(e.getKey())) return e.getValue();
        }
        return null;
    }

    /** A numeric metadata value whatever its GGUF width (split.* is UINT16 in the wild). */
    private static long metadataLong(GGUF gguf, String key) {
        if (!gguf.containsKey(key)) return 0;
        Object v = gguf.getValue(Object.class, key);
        return v instanceof Number n ? n.longValue() : 0;
    }

    /** The port claiming the GGUF's architecture; throws a REMEDY-naming error when none does. */
    private static ModelProvider provider(GGUF gguf) {
        String arch = gguf.getString("general.architecture");
        for (ModelProvider p : PROVIDERS) {
            if (p.supports(arch)) return p;
        }
        String artifact = artifactFor(arch);
        if (PROVIDERS.isEmpty()) {
            throw new IllegalArgumentException(
                    "no model providers on the classpath. This GGUF needs architecture '"
                            + arch
                            + "': add "
                            + (artifact != null
                                    ? artifact
                                    : "the com.qxotic:jinfer-* artifact that supports it")
                            + ", or com.qxotic:jinfer-models-all for everything. (Shading jinfer"
                            + " into one jar? merge META-INF/services - Maven Shade's"
                            + " ServicesResourceTransformer - or ServiceLoader finds nothing.)");
        }
        java.util.SortedSet<String> here = supportedArchitectures();
        throw new IllegalArgumentException(
                "no provider for architecture '"
                        + arch
                        + "'"
                        + (artifact != null
                                ? " on the classpath - add " + artifact
                                : " on the classpath, and none ships with this jinfer version"
                                        + " (a newer jinfer, or a third-party ModelProvider,"
                                        + " may support it)")
                        + ". Supported architectures: "
                        + (here.isEmpty()
                                ? PROVIDERS.size() + " provider(s), none enumerated"
                                : here));
    }
}
