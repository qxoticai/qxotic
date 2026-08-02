// langchain4j TextToSpeechModel backed by jinfer: in-process CPU synthesis over a local GGUF, no
// server. Names no port - either you pass a path and architecture dispatch finds one, or you pass
// a model you loaded and tuned yourself.
package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.SpeechModel;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechState;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.media.AudioCodec;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.model.audio.TextToSpeechModel;
import dev.langchain4j.model.audio.TextToSpeechRequest;
import dev.langchain4j.model.audio.TextToSpeechResponse;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe and shared, like every other {@code TextToSpeechModel}: concurrent requests run in
 * PARALLEL, each on a state of its own.
 *
 * <p>A jinfer speech state is ONE SERIAL PIPELINE and cannot be shared - so this does not share
 * one. Minting per call costs a measured +3.5% against reusing one (~1 ms on a short utterance),
 * which is a small price for not having to reconcile an unshareable state with a shared bean by
 * serializing every caller. Serializing would have hidden the capacity limit; rejecting past a
 * timeout would have failed only under load, which is worse.
 *
 * <p>The one thing that must still be coordinated is the WEIGHTS arena, which every synthesis
 * reads: {@link #close()} takes a write lock and therefore waits for every in-flight request before
 * freeing it. Requests take the read lock and never block each other.
 */
public final class JinferSpeechModel implements TextToSpeechModel, AutoCloseable {

    /** OpenAI's TTS limit, so a caller porting from it meets the same boundary here. */
    private static final int DEFAULT_MAX_INPUT_CHARS = 4096;

    private final SpeechModel<?, ?, SpeechState> model;
    private final Arena owned; // null unless this instance loaded the weights
    // Requests take the READ lock and run in PARALLEL - a state is per-call, so there is nothing
    // to serialize. close() takes the WRITE lock, which is what makes it wait for every in-flight
    // synthesis before freeing the weights arena those syntheses are reading.
    private final ReentrantReadWriteLock lifecycle = new ReentrantReadWriteLock();
    private final SpeechOptions defaults;
    private final int maxInputChars;
    private volatile boolean closed;

    @SuppressWarnings("unchecked") // the state below comes from this very model, so it IS S
    private JinferSpeechModel(Builder b) {
        this.defaults = b.speed == null ? SpeechOptions.NONE : SpeechOptions.speed(b.speed);
        this.maxInputChars = b.maxInputChars;
        // an arena this instance creates is this instance's to free on EVERY path out of here,
        // including a state allocation that fails after the weights are already mapped
        Arena created = b.model == null && b.arena == null ? Arena.ofShared() : null;
        try {
            this.model =
                    (SpeechModel<?, ?, SpeechState>)
                            (b.model != null
                                    ? b.model
                                    : Models.loadSpeech(
                                            b.modelPath, created != null ? created : b.arena));
        } catch (IOException e) {
            closeQuietly(created); // a leaked ofShared arena has no backstop: free before failing
            throw new UncheckedIOException("failed to load " + b.modelPath, e);
        } catch (RuntimeException | Error e) {
            closeQuietly(created);
            throw e;
        }
        this.owned = created; // a caller's arena stays the caller's
    }

    private static void closeQuietly(Arena arena) {
        if (arena != null) arena.close();
    }

    @Override
    public TextToSpeechResponse synthesize(TextToSpeechRequest request) {
        // langchain4j's voice is free text with no validation upstream; a caller who asks for one
        // and silently gets this model's only voice has been lied to
        if (request.voice() != null && !request.voice().isBlank())
            throw new UnsupportedOperationException(
                    "this model has one voice; requested '" + request.voice() + "'");
        String text = request.text();
        if (text.length() > maxInputChars)
            throw new IllegalArgumentException(
                    "text is "
                            + text.length()
                            + " characters, over the "
                            + maxInputChars
                            + " limit - raise maxInputChars(...) or split it");
        lifecycle.readLock().lock(); // shared: concurrent requests proceed in parallel
        try {
            if (closed) throw new IllegalStateException("this model is closed");
            // ONE STATE PER CALL. A jinfer speech state cannot be shared - that contract is the
            // port's, and the honest way to meet it is not to share one. Measured cost of minting
            // per call rather than reusing: +3.5% (about 1 ms on a short utterance), which buys
            // a bean that is thread-safe the way every other TextToSpeechModel is.
            try (SpeechState state = model.newState()) {
                Media.Audio audio = model.speak(state, text, defaults);
                return TextToSpeechResponse.from(
                        Audio.builder()
                                .binaryData(AudioCodec.wav(audio))
                                .mimeType("audio/wav")
                                .build());
            }
        } finally {
            lifecycle.readLock().unlock();
        }
    }

    /**
     * Idempotent, BLOCKING close: returns only after the in-flight synthesis (if any) has finished,
     * so its returning is the caller's quiescence certificate - the only thing standing between a
     * shutdown and a kernel reading freed memory. Frees the synthesis state, and the weights arena
     * IFF this instance created it: a model or an arena you passed in stays yours, so close yours
     * after this one, never before. Requests after this fail loudly.
     */
    @Override
    public void close() {
        lifecycle.writeLock().lock(); // BLOCKS until every in-flight synthesis has returned
        try {
            if (closed) return; // Arena.close is one-shot; this makes the adapter idempotent
            closed = true;
            closeQuietly(owned);
        } finally {
            lifecycle.writeLock().unlock();
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {

        private SpeechModel<?, ?, ?> model;
        private Path modelPath;
        private Arena arena;
        private Double speed;
        private int maxInputChars = DEFAULT_MAX_INPUT_CHARS;

        /**
         * A model you loaded yourself - the typed path, where a port's own knobs are expressible
         * ({@code InflectTTS.load(gguf, weights).variation(0.5)}). Its weights arena stays yours.
         * Mutually exclusive with {@link #modelPath}.
         */
        public Builder model(SpeechModel<?, ?, ?> model) {
            this.model = model;
            return this;
        }

        /** The GGUF to load, at the port's own defaults, through architecture dispatch. */
        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        /**
         * Where the weights map, with {@link #modelPath} only; default an arena closed with this.
         */
        public Builder arena(Arena arena) {
            this.arena = arena;
            return this;
        }

        /** Rate multiplier for every request, 1.0 = the model's natural rate. */
        public Builder speed(double speed) {
            this.speed = speed;
            return this;
        }

        /**
         * Longest accepted request, default {@value #DEFAULT_MAX_INPUT_CHARS}. Bounds chunk count,
         * and so compute AND output, since the port caps each chunk - which is what stops one
         * adversarial request from holding this instance's only pipeline indefinitely. Rejected
         * before any synthesis begins.
         */
        public Builder maxInputChars(int maxInputChars) {
            if (maxInputChars < 1)
                throw new IllegalArgumentException("maxInputChars " + maxInputChars);
            this.maxInputChars = maxInputChars;
            return this;
        }

        public JinferSpeechModel build() {
            if ((model == null) == (modelPath == null))
                throw new IllegalArgumentException(
                        "exactly one of model(...) or modelPath(...) is required");
            if (model != null && arena != null)
                throw new IllegalArgumentException(
                        "arena(...) is where modelPath(...) loads the weights; a model you built"
                                + " already has its own");
            return new JinferSpeechModel(this);
        }
    }
}
