// Text-to-speech: text in, waveform out. The Config/Weights/State triple like any other model -
// it just does not ingest a token stream, so it stands beside Model rather than under it.
package com.qxotic.jinfer;

import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * A speech model - named for what it PRODUCES, as {@link EmbeddingModel} is. It carries {@link
 * #config}, {@link #weights} and caller-owned {@link #newState state}, the same triple {@link
 * Model} is built on, and stands BESIDE it rather than under it: two of {@code Model}'s five
 * members would fit, but {@code ingest(Batch)} has no batch to ingest and {@link RuntimeState} is a
 * POSITIONAL CURSOR over ingested tokens, which a text-to-waveform pass has nothing to be
 * positioned in. Its state is scratch.
 *
 * <p>The port owns everything between text and samples: normalization, phonemization, how long text
 * is split, the pauses between clips and the joins. Weights are immutable and shared. A state is
 * ONE SERIAL PIPELINE and a LONG-LIVED object - mint one per pipeline at startup, because minting
 * one per utterance repays every sizing allocation and closes a shared arena, which is a JVM-wide
 * handshake. Nothing here is synchronized and nothing here pools: a layer that serves many callers
 * from one model owns that decision, and this one owns none of it.
 *
 * <p>Lifetime, the two laws the code cannot enforce: an arena must outlive every read from it
 * (kernels read raw addresses, so a violation is a crash, not an exception), and the weights arena
 * must outlive every model sharing those weights.
 */
public interface SpeechModel<C extends Config, W, S extends SpeechState> {

    C config();

    /**
     * Captured at load, so {@link #speak} never threads them, and exposed so a model can be cheaply
     * re-wrapped over shared weights ({@code new Impl(config(), weights())}).
     */
    W weights();

    /**
     * Allocate scratch from {@code arena}. {@code adopt} declares who frees it: false BORROWS - the
     * state's close never touches it and you close it yourself, after your last {@link #speak}
     * returns; true ADOPTS - the state's close frees it, co-tenants like weights included, so adopt
     * only when nothing in that arena outlives the state. A non-closeable arena ({@code ofAuto},
     * {@code global}) may be adopted: it manages itself, and close stays a valid no-op.
     *
     * <p>Use {@code ofShared}, never {@code ofConfined}: a port may run parts of one synthesis on
     * worker threads, and a confined arena fails loudly there.
     */
    S newState(Arena arena, boolean adopt);

    /** BORROWED scratch from {@code arena} - you own its lifetime. */
    default S newState(Arena arena) {
        return newState(arena, false);
    }

    /**
     * Scratch the state OWNS: an internal cross-thread arena that {@code state.close()} frees (an
     * {@code ofShared} on the JVM, degrading to {@code ofAuto} in a native image - see {@link
     * Arenas}). Warm scratch - reuse it across utterances; the first call sizes it and later calls
     * allocate nothing.
     */
    default S newState() {
        Arena arena = Arenas.newShared();
        try {
            return newState(arena, true);
        } catch (RuntimeException | Error e) {
            try {
                arena.close(); // a leaked ofShared arena has no backstop: free before failing
            } catch (UnsupportedOperationException ignored) {
                // ofAuto (native image) frees at GC
            }
            throw e;
        }
    }

    /**
     * The one synthesis method: each clip goes to {@code sink} as it is produced, so a consumer can
     * play or encode before the whole text is done.
     *
     * <p>{@code sink} returns false to CANCEL: nothing further is synthesized. Granularity is one
     * clip, not one sample - a clip is tens of milliseconds of compute, and finer would mean
     * threading a flag through the vocoder. Nothing is returned because there is nothing to report:
     * a caller that cancelled is the one that said so.
     *
     * <p>A clip is a HEAP COPY and does not alias state memory, so it stays valid after the state
     * is reused, closed, or its arena freed. (The opposite of {@link EmbeddingModel}'s sink, whose
     * tensor is a reused per-state buffer.)
     *
     * <p>{@code state} must come from this model, and options this model does not recognise are
     * rejected.
     */
    void speak(S state, String text, SpeechOptions options, Predicate<Media.Audio> sink);

    /**
     * One utterance, however long: the clips of {@link #speak(SpeechState, String, SpeechOptions,
     * Predicate)} concatenated. Defined in terms of the streaming pass so the two cannot drift
     * apart.
     */
    default Media.Audio speak(S state, String text, SpeechOptions options) {
        List<Media.Audio> clips = new ArrayList<>();
        speak(state, text, options, clips::add); // List::add returns true: never cancels
        return Media.Audio.concat(clips);
    }
}
