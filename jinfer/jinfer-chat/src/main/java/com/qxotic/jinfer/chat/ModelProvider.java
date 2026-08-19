package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * One port's entry in the architecture dispatch: a {@link java.util.ServiceLoader} service each
 * port module registers (META-INF/services), so {@link Models#load} finds exactly the ports on the
 * classpath - no hand-maintained arch table in every consumer.
 */
public interface ModelProvider {

    /** Whether this port loads GGUFs with the given {@code general.architecture}. */
    boolean supports(String architecture);

    /**
     * Wins ties when several providers claim one architecture: highest priority is selected. The
     * bundled providers all sit at the default 0, so a third-party override declares a higher value
     * to REPLACE one (equal priorities resolve deterministically by class name, with a warning
     * naming this knob).
     */
    default int priority() {
        return 0;
    }

    /**
     * The architectures this port claims, for DIAGNOSTICS: error messages and {@link
     * Models#supportedArchitectures()} enumerate these so a failed load can say what the classpath
     * actually provides. {@link #supports} stays the dispatch authority. Default empty: the port
     * still works, it just cannot introduce itself in error messages.
     */
    default Set<String> architectures() {
        return Set.of();
    }

    /**
     * Loads the model from an already-parsed GGUF; {@code fileChannel} supplies the tensor data,
     * mapped into {@code arena} (who provides the arena owns the weights' lifetime; it must outlive
     * every model sharing them). Nothing here is sized by context: a state's size is chosen at
     * {@code newState}, and the model's own context length comes from the GGUF.
     *
     * <p>{@code companions} arrive already validated against {@link #companionFiles()} by {@code
     * Models.load}; a port that takes none may ignore the map. {@code tokenizer} is a
     * caller-supplied override - null means build the GGUF's own - and a supplied one must keep the
     * GGUF's token-id space (checked by {@code Models.load} before this is called).
     */
    LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException;

    /**
     * The COMPANION FILES this architecture can take: capability name to the filename that carries
     * it, e.g. {@code Map.of("media", "mmproj", "speculation", "mtp")}. This is what the
     * architecture OFFERS; what a caller ATTACHES is the capability-to-{@link Path} map on {@code
     * load}. Declaring does not attach.
     *
     * <p>WHAT A COMPANION IS - the whole concept, in four laws the implementation follows:
     *
     * <ol>
     *   <li>ONE FILE that gives THIS architecture a capability its base model lacks - a media
     *       projector, a draft head, a pronunciation lexicon. It has no meaning without its model
     *       and is not independently loadable.
     *   <li>Named by CAPABILITY, attached EXPLICITLY by the caller - never discovered, never
     *       guessed. The capability is what a user asks for; the filename is this port's business
     *       (and how a downloader finds it).
     *   <li>Loaded BY THE PORT, into the model's own arena; how is the port's business, and it is
     *       not cached or preloaded - a companion header parse costs ~10 ms.
     *   <li>Its BYTES JOIN THE CACHE SEED ({@code Models.load} does this), because a companion
     *       changes what the model computes - cached KV must be keyed by it.
     * </ol>
     *
     * <p>And what a companion is NOT: not a tokenizer (that is the text-to-ids codec OUTSIDE the
     * computation, passed as the {@code load} tokenizer argument), and not a model.
     */
    default Map<String, String> companionFiles() {
        return Map.of();
    }

    /**
     * Loads an EMBEDDING model from an already-parsed GGUF ({@link Models#loadEmbedder}); {@code
     * path} names the loaded file, which is the identity its telemetry reports. Ports whose
     * architectures are generative-only keep this default. {@code tokenizer} is the validated
     * caller override, or null for the GGUF's own.
     */
    default LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        throw new UnsupportedOperationException(
                "'"
                        + gguf.getString("general.architecture")
                        + "' is not an embedding architecture");
    }

    /**
     * Loads a RERANKER from an already-parsed GGUF ({@link Models#loadReranker}): the backbone plus
     * this family's {@link com.qxotic.jinfer.Reranker} recipe. Ports with no reranker in the family
     * keep this default. {@code tokenizer} is the validated caller override, or null for the GGUF's
     * own.
     */
    default LoadedReranker<?> loadReranker(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        throw new UnsupportedOperationException(
                "'" + gguf.getString("general.architecture") + "' is not a reranker architecture");
    }

    /**
     * Loads a SPEECH model from an already-parsed GGUF ({@link Models#loadSpeech}) at the port's
     * own defaults. Ports whose architectures do not synthesize speech keep this default.
     *
     * <p>{@code path} is where the GGUF lives, passed because a speech front end has companions the
     * container does not carry - a phoneme port looks for its pronunciation lexicon beside the
     * model before falling back.
     */
    default com.qxotic.jinfer.SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) throws IOException {
        throw new UnsupportedOperationException(
                "'" + gguf.getString("general.architecture") + "' is not a speech architecture");
    }

    /**
     * As {@link #loadSpeech(FileChannel, GGUF, Path, Arena)} plus the companions the caller
     * attached, keyed by the capability names this port {@link #companionFiles() declares} - a
     * pronunciation lexicon, for a port that turns text into phonemes.
     *
     * <p>The default serves every port that takes none, so a speech port with no companions needs
     * no change: an empty map is the plain load, and a non-empty one is a caller asking for
     * something this architecture does not have.
     */
    default com.qxotic.jinfer.SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions)
            throws IOException {
        if (companions.isEmpty()) {
            return loadSpeech(fileChannel, gguf, path, arena);
        }
        throw new UnsupportedOperationException(
                "'"
                        + gguf.getString("general.architecture")
                        + "' takes no companions, but "
                        + companions.keySet()
                        + " were attached");
    }
}
