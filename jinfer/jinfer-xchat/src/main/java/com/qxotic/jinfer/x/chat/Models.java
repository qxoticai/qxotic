package com.qxotic.jinfer.x.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.GGUFFormatException;
import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.boundary.ContextState;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jinfer.x.boundary.Multimodal;
import com.qxotic.jinfer.x.boundary.media.ImageCodec;
import com.qxotic.toknroll.Tokenizer;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

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
                    "com.qxotic.jinfer.x.models.llama.LlamaProvider",
                    "com.qxotic.jinfer.x.models.llama.GraniteProvider",
                    "com.qxotic.jinfer.x.models.gemma4.Gemma4Provider",
                    "com.qxotic.jinfer.x.models.lfm2.Lfm2Provider",
                    "com.qxotic.jinfer.x.models.maple.MapleProvider",
                    "com.qxotic.jinfer.x.models.qwen3.Qwen3Provider",
                    "com.qxotic.jinfer.x.models.qwen35.Qwen35Provider",
                    "com.qxotic.jinfer.x.models.nemotronh.NemotronHProvider",
                    "com.qxotic.jinfer.x.models.gptoss.GptOssProvider",
                    "com.qxotic.jinfer.x.models.inflect2.Inflect2Provider");

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
        List<ModelProvider> recovered = new ArrayList<>();
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
     * Loads {@code path}. Weights map into {@code arena}: who provides the arena owns the weights'
     * lifetime ({@code ofAuto} = GC-managed, {@code global} = process, a scoped arena =
     * deterministic - it must outlive every model sharing the weights).
     */
    public static LoadedModel<?> load(Path path, Arena arena) throws IOException {
        return load(path, arena, Map.of(), null);
    }

    /**
     * Load with COMPANIONS, keyed by capability name - {@code "media"} for a projector. What a
     * companion IS - the four laws - lives on {@link ModelProvider#companionFiles()}. Values are
     * PATHS to single FILES, never references: resolving a reference is a downloader's job, done
     * before this call, so no library load ever touches the network.
     */
    public static LoadedModel<?> load(Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        return load(path, arena, companions, null);
    }

    /**
     * As {@link #load(Path, Arena, Map)} with the caller's OWN tokenizer instead of the one the
     * GGUF describes - a patched pre-tokenizer, or one built ahead of time so the load skips the
     * vocab/merges/regex build. Null means the GGUF's own; a supplied tokenizer must keep the
     * GGUF's token-id space ({@link #requireSameIdSpace}), because the embedding table and the
     * stop-token ids are indexed by id.
     */
    public static LoadedModel<?> load(
            Path path, Arena arena, Map<String, Path> companions, Tokenizer tokenizer)
            throws IOException {
        Map<String, Path> attached = Map.copyOf(companions);
        return open(
                path,
                (fc, gguf) -> {
                    ModelProvider provider = provider(gguf);
                    requireAccepted(provider, gguf, attached.keySet());
                    if (tokenizer != null) {
                        requireSameIdSpace(gguf, tokenizer);
                    }
                    return companionSeeded(
                            sampled(provider.load(fc, gguf, arena, attached, tokenizer), gguf),
                            attached);
                });
    }

    /**
     * Loads an EMBEDDING model (a retrieval checkpoint, told apart from a generative one by its own
     * metadata) through the same architecture dispatch as {@link #load}.
     */
    public static LoadedEmbedder<?> loadEmbedder(Path path, Arena arena) throws IOException {
        return loadEmbedder(path, arena, null);
    }

    /**
     * As {@link #loadEmbedder(Path, Arena)} with a caller-supplied tokenizer. Null means the GGUF's
     * own; a supplied tokenizer must keep its token-id space.
     */
    public static LoadedEmbedder<?> loadEmbedder(Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        return open(
                path,
                (fc, gguf) -> {
                    ModelProvider provider = provider(gguf);
                    if (tokenizer != null) requireSameIdSpace(gguf, tokenizer);
                    return provider.loadEmbedder(fc, gguf, path, arena, tokenizer);
                });
    }

    /** Loads a RERANKER - the backbone plus the family's scoring recipe. */
    public static LoadedReranker<?> loadReranker(Path path, Arena arena) throws IOException {
        return loadReranker(path, arena, null);
    }

    /**
     * As {@link #loadReranker(Path, Arena)} with a caller-supplied tokenizer. Null means the GGUF's
     * own; a supplied tokenizer must keep its token-id space.
     */
    public static LoadedReranker<?> loadReranker(Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        return open(
                path,
                (fc, gguf) -> {
                    ModelProvider provider = provider(gguf);
                    if (tokenizer != null) requireSameIdSpace(gguf, tokenizer);
                    return provider.loadReranker(fc, gguf, path, arena, tokenizer);
                });
    }

    /** Loads a SPEECH model at the port's own defaults. */
    public static com.qxotic.jinfer.x.boundary.SpeechSynthesisModel<?, ?, ?> loadSpeech(
            Path path, Arena arena) throws IOException {
        return loadSpeech(path, arena, Map.of());
    }

    /**
     * As {@link #loadSpeech(Path, Arena)} with COMPANIONS - {@code "phonemes"} for a pronunciation
     * lexicon. A port's own discovery (a lexicon beside the GGUF, then the classpath, then an
     * external tool) remains the DEFAULT; naming one here overrides that ladder rather than
     * extending it.
     */
    public static com.qxotic.jinfer.x.boundary.SpeechSynthesisModel<?, ?, ?> loadSpeech(
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

    /**
     * The capabilities {@code path}'s architecture can gain from a companion, and the filename that
     * carries each - the GGUF header only, no weights. A caller uses it to reject a capability this
     * architecture does not have BEFORE fetching anything for it.
     */
    public static Map<String, String> companionFiles(Path path) throws IOException {
        return open(path, (fc, gguf) -> companionFiles(gguf));
    }

    /**
     * As {@link #companionFiles(Path)} over an ALREADY-PARSED header - the AOT preload's bypass, so
     * a baked model is not re-parsed just to validate a {@code --with} flag.
     */
    public static Map<String, String> companionFiles(GGUF gguf) {
        return provider(gguf).companionFiles();
    }

    /**
     * The architectures the classpath's ports claim (sorted) - what {@link #load} can dispatch. For
     * tooling and startup banners.
     */
    public static SortedSet<String> supportedArchitectures() {
        TreeSet<String> archs = new TreeSet<>();
        for (ModelProvider p : PROVIDERS) archs.addAll(p.architectures());
        return archs;
    }

    private interface Load<T> {
        T apply(FileChannel fc, GGUF gguf) throws IOException;
    }

    /** Opens {@code path}, reads the GGUF header, and hands both to {@code load}. */
    private static <T> T open(Path path, Load<T> load) throws IOException {
        if (!Files.exists(path)) {
            throw new NoSuchFileException(path.toString(), null, "model file not found");
        }
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            fc.position(0L);
            GGUF gguf;
            try {
                gguf =
                        GGUF.read(
                                Channels.newChannel(
                                        new BufferedInputStream(
                                                Channels.newInputStream(fc), 1 << 20)));
            } catch (GGUFFormatException e) {
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
            return load.apply(fc, gguf);
        }
    }

    // arch (or prefix) -> the Maven artifact that provides it. DIAGNOSTICS ONLY - dispatch never
    // reads this; it exists so "unsupported architecture" can name the jar to add.
    private static final Map<String, String> PORT_ARTIFACTS =
            Map.ofEntries(
                    Map.entry("gemma4", "com.qxotic:jinfer-xgemma4"),
                    Map.entry("gpt-oss", "com.qxotic:jinfer-xgptoss"),
                    Map.entry("lfm", "com.qxotic:jinfer-xlfm2"),
                    Map.entry("llama", "com.qxotic:jinfer-xllama"),
                    Map.entry("granite", "com.qxotic:jinfer-xllama"),
                    Map.entry("minicpm", "com.qxotic:jinfer-xllama"),
                    Map.entry("mistral3", "com.qxotic:jinfer-xllama"),
                    Map.entry("smollm3", "com.qxotic:jinfer-xllama"),
                    Map.entry("maple", "com.qxotic:jinfer-xmaple"),
                    Map.entry("nemotron_h", "com.qxotic:jinfer-xnemotronh"),
                    Map.entry("qwen3", "com.qxotic:jinfer-xqwen3"),
                    Map.entry("qwen35", "com.qxotic:jinfer-xqwen35"),
                    Map.entry("inflect", "com.qxotic:jinfer-xinflect2"));

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

    /** The diagnostics table's answer for {@code arch}, or null. */
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
        ModelProvider selected = select(PROVIDERS, arch);
        if (selected != null) return selected;
        String artifact = artifactFor(arch);
        if (PROVIDERS.isEmpty()) {
            throw new IllegalArgumentException(
                    "no model providers on the classpath. This GGUF needs architecture '"
                            + arch
                            + "': add "
                            + (artifact != null
                                    ? artifact
                                    : "the com.qxotic:jinfer-x* artifact that supports it")
                            + ". (Shading jinfer into one jar? merge META-INF/services - Maven"
                            + " Shade's ServicesResourceTransformer - or ServiceLoader finds"
                            + " nothing.)");
        }
        SortedSet<String> here = supportedArchitectures();
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
                                + (accepted.isEmpty() ? "none" : new TreeSet<>(accepted.keySet())));
            }
        }
    }

    /**
     * A supplied tokenizer must cover the GGUF's exact token-id space: token ids index the
     * embedding table and the stop-token ids.
     */
    private static void requireSameIdSpace(GGUF gguf, Tokenizer tokenizer) {
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
     * Attaches the effective sampling recommendations: the GGUF's {@code general.sampling.*} where
     * present, falling back to the port's model-author recommendation, if it declared one.
     */
    private static <S extends ContextState> LoadedModel<S> sampled(
            LoadedModel<S> loaded, GGUF gguf) {
        return new LoadedModel<>(
                loaded.model(),
                loaded.tokenizer(),
                loaded.chatTemplateSource(),
                loaded.stopTokens(),
                loaded.seed(),
                loaded.template(),
                LoadedModel.SamplingDefaults.fromGGUF(gguf)
                        .withFallback(loaded.samplingDefaults()));
    }

    /**
     * Re-roots the cache seed with EVERY ATTACHED COMPANION, plus the image decoder and the model's
     * preprocessing plan. Media blocks are content-keyed by their SOURCE bytes, so everything
     * standing between those bytes and the stored KV must be part of the key space: a different
     * projector producing different rows for the same image must never be served blocks cached
     * under the old one.
     */
    static <S extends ContextState> LoadedModel<S> companionSeeded(
            LoadedModel<S> loaded, Map<String, Path> companions) {
        if (companions.isEmpty()) {
            return loaded;
        }
        MessageDigest sha = sha256();
        sha.update(loaded.seed().value().getBytes(StandardCharsets.UTF_8));
        // sorted, so the seed does not depend on the order a caller listed them in
        for (var companion : new TreeMap<>(companions).entrySet()) {
            sha.update(companion.getKey().getBytes(StandardCharsets.UTF_8));
            sha.update(modelSeed(companion.getValue()).value().getBytes(StandardCharsets.UTF_8));
        }
        sha.update(ImageCodec.decoder().name().getBytes(StandardCharsets.UTF_8));
        if (loaded.model() instanceof Multimodal mm) {
            mm.projector(Media.Image.class)
                    .ifPresent(
                            projector ->
                                    sha.update(
                                            projector.planId().getBytes(StandardCharsets.UTF_8)));
            mm.projector(Media.Audio.class)
                    .ifPresent(
                            projector ->
                                    sha.update(
                                            projector.planId().getBytes(StandardCharsets.UTF_8)));
        }
        return new LoadedModel<>(
                loaded.model(),
                loaded.tokenizer(),
                loaded.chatTemplateSource(),
                loaded.stopTokens(),
                new ContentKey("sha256:" + HexFormat.of().formatHex(sha.digest())),
                loaded.template(),
                loaded.samplingDefaults());
    }

    /**
     * A fast, stable identity for a model file: length + first and last MiB, hashed (full-content
     * hashing of multi-GB weights is not worth it - length + head/tail covers metadata, tensor
     * table and data edges).
     */
    public static ContentKey modelSeed(Path gguf) {
        try (var ch = FileChannel.open(gguf, StandardOpenOption.READ)) {
            return modelSeed(ch);
        } catch (IOException e) {
            throw new IllegalStateException("modelSeed(" + gguf + ")", e);
        }
    }

    /** As {@link #modelSeed(Path)} on an already-open channel (positional reads). */
    public static ContentKey modelSeed(FileChannel ch) {
        try {
            MessageDigest d = sha256();
            long size = ch.size();
            ByteBuffer len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(0, size);
            d.update(len);
            ByteBuffer buf = ByteBuffer.allocate((int) Math.min(1 << 20, size));
            ch.read(buf, 0);
            buf.flip();
            d.update(buf);
            if (size > (1 << 20)) {
                buf.clear();
                ch.read(buf, size - buf.capacity());
                buf.flip();
                d.update(buf);
            }
            return new ContentKey("sha256:" + HexFormat.of().formatHex(d.digest()));
        } catch (IOException e) {
            throw new IllegalStateException("modelSeed(channel)", e);
        }
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }
}
