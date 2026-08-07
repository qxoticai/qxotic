package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;

/**
 * One port's entry in the architecture dispatch: a {@link java.util.ServiceLoader} service each
 * port module registers (META-INF/services), so {@link Models#load} finds exactly the ports on the
 * classpath - no hand-maintained arch table in every consumer.
 */
public interface ModelProvider {

    /**
     * A companion FILE as it reaches a port: its path, and - when a preload already parsed it - its
     * header, so the port skips re-reading it. A null {@code header} means "parse it yourself"; a
     * non-GGUF companion (a lexicon) simply never has one.
     */
    record Companion(Path path, GGUF header) {
        public static Companion of(Path path) {
            return new Companion(path, null);
        }
    }

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
     * actually provides. {@link #supports} stays the dispatch authority (a port matching by prefix
     * lists representative names here). Default empty: the port still works, it just cannot
     * introduce itself in error messages.
     */
    default java.util.Set<String> architectures() {
        return java.util.Set.of();
    }

    /**
     * Loads a GENERATIVE model from an already-parsed GGUF; {@code fileChannel} supplies the tensor
     * data, mapped into {@code arena} (who provides the arena owns the weights' lifetime; it must
     * outlive every model sharing them). Nothing here is sized by context: a state's size is chosen
     * at {@code newState}, and the model's own context length comes from the GGUF.
     *
     * <p>Every capability here is optional and {@link #supports} is the only requirement: a port
     * overrides the loads its architecture actually has, and a speech-only or embedding-only port
     * keeps the rest of these defaults.
     *
     * <p>{@code companions} arrive already validated against {@link #companionFiles()} by {@code
     * Models.load}; a port that takes none may ignore the map. {@code tokenizer} is a
     * caller-supplied override - null means build the GGUF's own - and a supplied one must keep the
     * GGUF's token-id space (checked by {@code Models.load} before this is called).
     */
    default LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Companion> companions,
            Tokenizer tokenizer)
            throws IOException {
        throw new UnsupportedOperationException(
                "'"
                        + gguf.getString("general.architecture")
                        + "' is not a generative architecture");
    }

    /**
     * The COMPANION FILES this architecture can take: capability name to the filename that carries
     * it. Distinct from the {@code companions} map a caller ATTACHES, which is capability name to a
     * {@link Path} - this one is what the architecture OFFERS. A companion is an auxiliary file
     * that has no meaning without this model and is loaded by this port into the same arena - a
     * media projector, a draft head, a pronunciation lexicon.
     *
     * <pre>
     *   Map.of("media", "mmproj", "speculation", "mtp")
     * </pre>
     *
     * <p>The CAPABILITY is what a user asks for; the filename is this port's business, and how a
     * downloader finds it in the model's repository. Naming the capability rather than the file is
     * what lets a second implementation of the same capability arrive without renaming anything a
     * user types (speculation is MTP here and Eagle3 elsewhere).
     *
     * <p>Declaring one does NOT attach it: every companion costs memory or changes behaviour, so
     * attaching stays the caller's explicit act.
     */
    default Map<String, String> companionFiles() {
        return Map.of();
    }

    /**
     * Loads an EMBEDDING model from an already-parsed GGUF ({@link Models#loadEmbedder}); {@code
     * path} names the loaded file, which is the identity its telemetry reports. Ports whose
     * architectures are generative-only keep this default.
     */
    default LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) throws IOException {
        throw new UnsupportedOperationException(
                "'"
                        + gguf.getString("general.architecture")
                        + "' is not an embedding architecture");
    }

    /**
     * Loads a SPEECH model from an already-parsed GGUF ({@link Models#loadSpeech}) at the port's
     * own defaults. Ports whose architectures do not synthesize speech keep this default.
     *
     * <p>{@code path} is where the GGUF lives, passed because a speech front end has companions the
     * container does not carry - a phoneme port looks for its pronunciation lexicon beside the
     * model before falling back.
     */
    default com.qxotic.jinfer.SpeechModel<?, ?, ?> loadSpeech(
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
    default com.qxotic.jinfer.SpeechModel<?, ?, ?> loadSpeech(
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

    /**
     * Loads a RERANKER from an already-parsed GGUF ({@link Models#loadReranker}): the backbone plus
     * this family's {@link Reranker} recipe. Ports with no reranker in the family keep this
     * default.
     */
    default LoadedReranker<?> loadReranker(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) throws IOException {
        throw new UnsupportedOperationException(
                "'" + gguf.getString("general.architecture") + "' is not a reranker architecture");
    }
}
