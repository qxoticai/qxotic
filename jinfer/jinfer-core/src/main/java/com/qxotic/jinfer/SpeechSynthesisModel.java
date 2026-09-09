package com.qxotic.jinfer;

import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.function.Predicate;

/**
 * A phoneme-to-waveform model with reusable runtime state: phonemes in at the low level, text in at
 * the high level, the way a language model takes tokens and a chat engine takes messages.
 *
 * <pre>
 *   speak:   normalize  ->  chunk  ->  phonemize   ->  synthesize  ->  join
 *            port           port        Phonemizer      model           port
 * </pre>
 *
 * <p>The model is shared; a state is one serial pipeline, used by one call at a time.
 */
public interface SpeechSynthesisModel<C, W, S extends RuntimeState> extends Model<C, W, S> {

    /** Output sample rate, in Hz. */
    int sampleRate();

    /**
     * This model's grapheme-to-phoneme step, bound at load. Text policy - normalization, sentence
     * chunking, a terminating mark - is {@link #speak}'s, not the phonemizer's: feed it what you
     * would feed a tokenizer.
     */
    Phonemizer phonemizer();

    /** Creates state that owns its memory. */
    S newState();

    /** Creates state that borrows caller-owned memory. */
    S newState(MemoryArena<MemorySegment> arena);

    /**
     * LOW LEVEL. One utterance, verbatim: ids in this model's vocabulary, synthesized as given. A
     * sequence over the family's ceiling, or an id off its table, is REFUSED - never split, never
     * substituted. Splitting is a text decision and belongs to whoever still has the text.
     */
    Media.Audio synthesize(S state, int[] phonemes, SpeechOptions options);

    /**
     * HIGH LEVEL. The family's text policy over {@link #synthesize}: normalized, cut into
     * utterances, phonemized, joined. Streams clips; the sink returns false to cancel after the
     * current clip. Blank text is refused.
     */
    void speak(S state, String text, SpeechOptions options, Predicate<Media.Audio> sink);

    /** Synthesizes and concatenates the complete waveform. */
    default Media.Audio speak(S state, String text, SpeechOptions options) {
        var clips = new ArrayList<Media.Audio>();
        speak(state, text, options, clips::add);
        return Media.Audio.concat(clips);
    }
}
