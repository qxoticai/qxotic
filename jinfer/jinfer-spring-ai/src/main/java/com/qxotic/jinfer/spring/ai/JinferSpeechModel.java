// Spring AI TextToSpeechModel backed by jinfer: in-process CPU synthesis over a local GGUF, no
// server. Names no port - either you pass a path and architecture dispatch finds one, or you pass
// a model you loaded and tuned yourself.
package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.SpeechModel;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechState;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.media.AudioCodec;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;
import org.springframework.ai.audio.tts.Speech;
import org.springframework.ai.audio.tts.TextToSpeechModel;
import org.springframework.ai.audio.tts.TextToSpeechOptions;
import org.springframework.ai.audio.tts.TextToSpeechPrompt;
import org.springframework.ai.audio.tts.TextToSpeechResponse;
import reactor.core.publisher.Flux;

/**
 * One model, one synthesis state, one lock. A jinfer speech state is ONE SERIAL PIPELINE, so
 * concurrent requests queue on a fair lock rather than being refused - the port already fans out
 * across cores inside a single synthesis. When it does profile as the bottleneck, declare a second
 * bean over the same model: weights are immutable and shared, so that is a second pipeline for no
 * new code.
 */
public final class JinferSpeechModel implements TextToSpeechModel, AutoCloseable {

    /** OpenAI's TTS limit, so a caller porting from it meets the same boundary here. */
    private static final int DEFAULT_MAX_INPUT_CHARS = 4096;

    private final SpeechModel<?, ?, SpeechState> model;
    private final SpeechState state;
    private final Arena owned; // null unless this instance loaded the weights
    private final ReentrantLock lock = new ReentrantLock(true); // fair: no request starves
    private final SpeechOptions defaults;
    private final int maxInputChars;

    @SuppressWarnings("unchecked") // the state below comes from this very model, so it IS S
    private JinferSpeechModel(Builder b) {
        this.defaults = b.speed == null ? SpeechOptions.NONE : SpeechOptions.speed(b.speed);
        this.maxInputChars = b.maxInputChars;
        if (b.model != null) {
            this.owned = null;
            this.model = (SpeechModel<?, ?, SpeechState>) b.model;
        } else {
            Arena weights = b.arena == null ? Arena.ofShared() : b.arena;
            try {
                this.model = (SpeechModel<?, ?, SpeechState>) Models.loadSpeech(b.modelPath, weights);
            } catch (IOException e) {
                if (b.arena == null) weights.close();
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            } catch (RuntimeException | Error e) {
                if (b.arena == null) weights.close();
                throw e;
            }
            this.owned = b.arena == null ? weights : null; // a caller's arena stays the caller's
        }
        this.state = model.newState();
    }

    @Override
    public TextToSpeechResponse call(TextToSpeechPrompt prompt) {
        String text = text(prompt);
        SpeechOptions options = options(prompt);
        lock.lock(); // the port fails fast on concurrent use of one state; this queues instead
        try {
            Media.Audio audio = model.speak(state, text, options);
            return new TextToSpeechResponse(List.of(new Speech(AudioCodec.wav(audio))));
        } finally {
            lock.unlock();
        }
    }

    /**
     * One element per clip, so a caller plays or writes before the whole text is done. Cancelling
     * the subscription cancels the synthesis - the sink's false return is the port's cancel
     * signal, so no further clip is computed.
     *
     * <p>The pipeline is held for the whole emission: a state is one serial pipeline, and a second
     * request must wait rather than interleave into the same scratch.
     */
    @Override
    public Flux<TextToSpeechResponse> stream(TextToSpeechPrompt prompt) {
        String text = text(prompt);
        SpeechOptions options = options(prompt);
        return Flux.create(
                emitter -> {
                    lock.lock();
                    try {
                        model.speak(
                                state,
                                text,
                                options,
                                clip -> {
                                    if (emitter.isCancelled()) return false;
                                    emitter.next(
                                            new TextToSpeechResponse(
                                                    List.of(new Speech(AudioCodec.pcm16(clip)))));
                                    return true;
                                });
                        emitter.complete();
                    } catch (RuntimeException e) {
                        emitter.error(e);
                    } finally {
                        lock.unlock();
                    }
                });
    }

    private String text(TextToSpeechPrompt prompt) {
        String text = prompt.getInstructions().getText();
        if (text.length() > maxInputChars)
            throw new IllegalArgumentException(
                    "text is "
                            + text.length()
                            + " characters, over the "
                            + maxInputChars
                            + " limit - raise maxInputChars(...) or split it");
        return text;
    }

    /**
     * The request's knobs, of which exactly one survives translation. {@code voice}, {@code model}
     * and {@code format} name choices this instance does not have, and a caller who set one and
     * silently got the default has been lied to.
     */
    private SpeechOptions options(TextToSpeechPrompt prompt) {
        TextToSpeechOptions requested = prompt.getOptions();
        if (requested == null) return defaults;
        reject("voice", requested.getVoice(), "this model has one voice");
        reject("model", requested.getModel(), "the GGUF this instance loaded is the model");
        reject("format", requested.getFormat(), "output is WAV");
        return requested.getSpeed() == null ? defaults : SpeechOptions.speed(requested.getSpeed());
    }

    private static void reject(String knob, String value, String why) {
        if (value != null && !value.isBlank())
            throw new UnsupportedOperationException(
                    knob + " '" + value + "' is not supported: " + why);
    }

    /**
     * Closes the synthesis state, and the weights arena IFF this instance created it - a model or
     * an arena you passed in stays yours, so close yours after this one, never before. Call once
     * every synthesis has returned.
     */
    @Override
    public void close() {
        state.close();
        if (owned != null) owned.close();
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

        /** Where the weights map, with {@link #modelPath} only; default an arena closed with this. */
        public Builder arena(Arena arena) {
            this.arena = arena;
            return this;
        }

        /** Rate multiplier for requests that do not carry one, 1.0 = the model's natural rate. */
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
            if (maxInputChars < 1) throw new IllegalArgumentException("maxInputChars " + maxInputChars);
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
