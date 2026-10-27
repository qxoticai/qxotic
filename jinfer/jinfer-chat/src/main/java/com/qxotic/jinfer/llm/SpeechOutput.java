package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContextState;
import java.util.random.RandomGenerator;

/**
 * A model that can answer in speech: its hidden state becomes audio frames, one context position
 * each, beside the text tokens the caller samples. The caller drives one {@link Speaker} per reply
 * - where text comes next it samples, ingests and shows the token; otherwise it ingests the frame
 * the speaker hands back - so ingestion, budgets, cancellation and caching stay the caller's, and
 * the lane schedule (how text and speech alternate) stays the family's.
 *
 * <p>Deliberately narrow, like {@link SpeculativeDecoding}: one capability check and one per-reply
 * object.
 *
 * @param <S> the implementing model's state type
 */
public interface SpeechOutput<S extends ContextState> {

    /** Whether this load can speak (its speech head is attached). */
    boolean speaks();

    /** The speech's sample rate, in Hz; mono. */
    int speechRate();

    /**
     * The token a spoken reply opens with, which the caller ingests as the end of the prompt:
     * speech with its transcript interleaved, or speech alone.
     */
    int opener(boolean transcript);

    /** The token id a frame stands for in a reply's token record. */
    int framePlaceholder();

    /** One spoken reply from the state's frontier, its {@link #opener} already ingested. */
    Speaker speaker(S state, boolean transcript, RandomGenerator random);

    /** One reply's lanes. Not thread-safe; close it when the reply ends. */
    interface Speaker extends AutoCloseable {

        /** Whether the next position is a text token the caller samples; else a {@link #frame}. */
        boolean textNext();

        /** What the caller's sampled text token means to the reply, before it is ingested. */
        Lane text(int token);

        /**
         * The next frame, from the state's last position. Its {@code row} goes in next, at once:
         * the batch borrows the speaker's memory until the next call.
         */
        Frame frame();

        /** The audio still held back once the reply is over. */
        float[] flush();

        @Override
        void close();
    }

    /** A sampled text token's role. */
    enum Lane {
        /** Transcript text: ingested and shown. */
        TEXT,
        /** A lane marker (the end of the transcript): ingested, never shown. */
        CONTROL,
        /** The end of the reply: neither ingested nor shown. */
        END
    }

    /**
     * One frame: its codes (the caller's to keep), the PCM it decodes to (possibly empty while the
     * decoder fills), and the one-row batch the caller ingests at the next position. The end frame
     * is a frame too: it is ingested, and text follows.
     */
    record Frame(int[] codes, float[] pcm, Batch row) {}
}
