package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * One port's entry in the architecture dispatch: a {@link java.util.ServiceLoader} service each
 * port module registers (META-INF/services), so {@link Models} finds exactly the ports on the
 * classpath - no hand-maintained arch table in every consumer.
 *
 * <p>A port declares its {@link #architectures()} and overrides the {@code load*} method of each
 * kind it can produce - language, embedding, reranker, speech, transcription. The others keep their
 * default, which refuses with the architecture's name. All five take the same picture of the model:
 * the {@code channel} holding the bytes, the parsed {@code gguf} (whose tensor offsets are absolute
 * in that channel - an embedded GGUF arrives as {@link GGUF#at(long)}), the {@code path} of the
 * file on disk (the GGUF itself, or the archive holding it: the loaded model's name, and where a
 * port looks for a sibling file), and the {@code arena} the weights are mapped into (who provides
 * the arena owns the weights' lifetime; it must outlive every model sharing them). Nothing here is
 * sized by context: a state's size is chosen at {@code newState}, and the model's own context
 * length comes from the GGUF.
 *
 * <p><b>WARNING: confined arenas MUST NOT be supplied to any loading method. Misuse can corrupt
 * memory or crash the JVM; a confined arena is refused at load.</b> Even one worker may run on a
 * custom pool's thread or native pthread other than the arena's owner. Raw-address kernels bypass
 * JDK confinement checks. See {@link com.qxotic.jinfer.Arenas} for the memory contract.
 */
public interface ModelProvider {

    /** The {@code general.architecture} values this port loads - the dispatch key. */
    Set<String> architectures();

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
     * The COMPANION FILES this architecture can take: capability name to the filename that carries
     * it, e.g. {@code Map.of("media", "mmproj", "speculation", "mtp")}. This is what the
     * architecture OFFERS; what a caller ATTACHES is the capability-to-{@link Path} map on {@code
     * loadLanguage} or {@code loadSpeech}. Declaring does not attach.
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
     * computation, passed as the {@code tokenizer} argument), and not a model.
     */
    default Map<String, String> companionFiles() {
        return Map.of();
    }

    /**
     * Loads a LANGUAGE model ({@link Models#load}). {@code companions} arrive already validated
     * against {@link #companionFiles()}; a port that takes none may ignore the map. {@code
     * tokenizer} is a caller-supplied override - null means build the GGUF's own - and a supplied
     * one must keep the GGUF's token-id space (checked before this is called).
     */
    default LoadedModel<?> loadLanguage(
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        throw notA(gguf, "a language");
    }

    /**
     * Loads an EMBEDDING model ({@link Models#loadEmbedder}). {@code tokenizer} is the validated
     * caller override, or null for the GGUF's own.
     */
    default LoadedEmbedder<?> loadEmbedder(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        throw notA(gguf, "an embedding");
    }

    /**
     * Loads a RERANKER ({@link Models#loadReranker}). {@code tokenizer} is the validated caller
     * override, or null for the GGUF's own.
     */
    default LoadedReranker<?> loadReranker(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        throw notA(gguf, "a reranker");
    }

    /**
     * Loads a SPEECH SYNTHESIS model ({@link Models#loadSpeech}). {@code companions} arrive already
     * validated against {@link #companionFiles()} - a voice, a pronunciation lexicon.
     */
    default SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        throw notA(gguf, "a speech");
    }

    /**
     * Loads a TRANSCRIPTION (speech-to-text) model ({@link Models#loadTranscription}). {@code
     * companions} arrive already validated against {@link #companionFiles()}.
     */
    default TranscriptionModel<?, ?, ?> loadTranscription(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        throw notA(gguf, "a transcription");
    }

    private static IncompatibleModelException notA(GGUF gguf, String kind) {
        return new IncompatibleModelException(
                "'"
                        + gguf.getString("general.architecture")
                        + "' is not "
                        + kind
                        + " architecture");
    }

    /**
     * The model is known, but not as the kind that was asked for. A port claimed the file's
     * architecture and could load it, only not through this {@code load*} method: {@link
     * Models#loadSpeech} on a language model, {@link Models#load} on a transcription model. Every
     * {@code load*} default a port leaves in place throws it, before any weights are touched.
     *
     * <p>It is distinct from the other refusals a load can raise, and callers rely on the
     * difference: an architecture no port claims, a file that is not a GGUF, or one part of a split
     * model are refused with other exceptions, because in each of those cases no loader would have
     * worked. Here another one would. So a caller that offers several kinds (the CLI server, which
     * serves language or transcription models) catches this to try the next loader, and a caller
     * with one kind reports it as "wrong model for this command" rather than "broken model".
     *
     * <p>The message names the architecture and the kind refused, in the form {@code 'lfm2' is not
     * a speech architecture}. A port that overrides a {@code load*} method only to say more (what
     * the architecture IS, and the loader that takes it) throws this type with its own message, so
     * callers that try the next loader still can.
     */
    final class IncompatibleModelException extends IllegalArgumentException {
        public IncompatibleModelException(String message) {
            super(message);
        }
    }
}
