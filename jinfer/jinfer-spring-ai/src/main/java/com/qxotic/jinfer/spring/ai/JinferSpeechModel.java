// Spring AI TextToSpeechModel backed by jinfer: in-process CPU synthesis over a local GGUF, no
// server. Names no port - either you pass a path and architecture dispatch finds one, or you pass
// a model you loaded and tuned yourself.
package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.springframework.ai.audio.tts.Speech;
import org.springframework.ai.audio.tts.TextToSpeechModel;
import org.springframework.ai.audio.tts.TextToSpeechOptions;
import org.springframework.ai.audio.tts.TextToSpeechPrompt;
import org.springframework.ai.audio.tts.TextToSpeechResponse;
import org.springframework.ai.audio.tts.TextToSpeechResponseMetadata;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

/**
 * Spring AI {@code TextToSpeechModel} backed by jinfer: in-process CPU synthesis over a local GGUF.
 * {@link #call} returns WAV audio; {@link #stream} emits successive PCM16 clips.
 *
 * <p>Thread-safe: concurrent requests use independent synthesis states and may run in parallel.
 * {@link #close()} waits for in-flight requests before freeing owned weights.
 */
public final class JinferSpeechModel implements TextToSpeechModel, AutoCloseable {

    /** OpenAI's TTS limit, so a caller porting from it meets the same boundary here. */
    private static final int DEFAULT_MAX_INPUT_CHARS = 4096;

    /** Metadata keys on every response: how to play the bytes. Integers, Hz and a count. */
    public static final String SAMPLE_RATE = "sampleRate";

    public static final String CHANNELS = "channels";

    private final SpeechSynthesisModel<?, ?, RuntimeState> model;
    private final Arena owned; // null unless this instance loaded the weights
    // Requests take the READ lock and run in PARALLEL - a state is per-call, so there is nothing
    // to serialize. close() takes the WRITE lock, which is what makes it wait for every in-flight
    // synthesis before freeing the weights arena those syntheses are reading.
    private final ReentrantReadWriteLock lifecycle = new ReentrantReadWriteLock();
    private final TextToSpeechOptions defaultOptions;
    private final int maxInputChars;
    private volatile boolean closed;

    @SuppressWarnings("unchecked") // the state below comes from this very model, so it IS S
    private JinferSpeechModel(Builder b) {
        this.defaultOptions = TextToSpeechOptions.builder().speed(b.speed).build();
        this.maxInputChars = b.maxInputChars;
        // an arena this instance creates is this instance's to free on EVERY path out of here,
        // including a state allocation that fails after the weights are already mapped
        Arena created = b.model == null ? Arenas.newCrossThread() : null;
        try {
            this.model =
                    (SpeechSynthesisModel<?, ?, RuntimeState>)
                            (b.model != null
                                    ? b.model
                                    : Models.loadSpeech(b.modelPath, created, b.companionPaths));
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
        if (arena != null) Arenas.close(arena);
    }

    @Override
    public TextToSpeechResponse call(TextToSpeechPrompt prompt) {
        String text = text(prompt);
        SpeechOptions options = resolveOptions(prompt.getOptions());
        lifecycle.readLock().lock(); // shared: concurrent requests proceed in parallel
        try {
            checkOpen();
            // ONE STATE PER CALL - a jinfer speech state cannot be shared, so this does not share
            // one. Measured at +3.5% against reusing a state, which is what a thread-safe bean is
            // worth.
            try (RuntimeState state = model.newState()) {
                Media.Audio audio = model.speak(state, text, options);
                return response(AudioCodec.wav(audio), audio);
            }
        } finally {
            lifecycle.readLock().unlock();
        }
    }

    /**
     * One element per clip, so a caller plays or writes before the whole text is done. Cancelling
     * the subscription cancels the synthesis - the sink's false return is the port's cancel signal,
     * so no further clip is computed.
     *
     * <p>The pipeline is held for the whole emission: a state is one serial pipeline, and a second
     * request must wait rather than interleave into the same scratch.
     */
    @Override
    public Flux<TextToSpeechResponse> stream(TextToSpeechPrompt prompt) {
        String text = text(prompt);
        SpeechOptions options = resolveOptions(prompt.getOptions());
        // The state is scoped to the SUBSCRIPTION, not to this method: a Flux may be subscribed
        // late, more than once, or never, and each subscription is its own synthesis.
        return Flux.<TextToSpeechResponse>create(
                        emitter -> {
                            lifecycle.readLock().lock();
                            try {
                                checkOpen();
                                try (RuntimeState state = model.newState()) {
                                    model.speak(
                                            state,
                                            text,
                                            options,
                                            clip -> {
                                                if (emitter.isCancelled()) return false;
                                                emitter.next(
                                                        response(AudioCodec.pcm16(clip), clip));
                                                return true;
                                            });
                                }
                                emitter.complete();
                            } catch (RuntimeException | Error e) {
                                // Errors too: a swallowed Error on the elastic thread would leave
                                // the subscriber waiting forever with nothing in the logs
                                emitter.error(e);
                            } finally {
                                lifecycle.readLock().unlock();
                            }
                        })
                // The synthesis is BLOCKING and holds the pipeline for its whole emission, so it
                // must not run on the subscriber's thread - in WebFlux that is an event-loop
                // thread, and parking one there stalls every other request on that loop. The chat
                // side solves the same problem with the engine's own driver thread.
                .subscribeOn(Schedulers.boundedElastic());
    }

    /**
     * WAV from {@link #call}, PCM16 from {@link #stream}: both carry what a player needs, read off
     * the clip itself. WAV describes itself already; the same shape from both doors costs nothing.
     */
    private static TextToSpeechResponse response(byte[] bytes, Media.Audio audio) {
        var metadata = new TextToSpeechResponseMetadata();
        metadata.put(SAMPLE_RATE, audio.sampleRate());
        metadata.put(CHANNELS, audio.channels());
        return new TextToSpeechResponse(List.of(new Speech(bytes)), metadata);
    }

    private String text(TextToSpeechPrompt prompt) {
        Objects.requireNonNull(prompt, "prompt must not be null");
        var instructions =
                Objects.requireNonNull(
                        prompt.getInstructions(), "prompt instructions must not be null");
        String text = instructions.getText();
        if (text == null || text.isBlank())
            throw new IllegalArgumentException("text cannot be null or blank");
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
    private SpeechOptions resolveOptions(TextToSpeechOptions requested) {
        Double speed = defaultOptions.getSpeed();
        if (requested != null) {
            reject(
                    "voice",
                    requested.getVoice(),
                    "the voice is fixed by the loaded GGUF; load the desired voice model instead");
            reject("model", requested.getModel(), "this instance is bound to the loaded GGUF");
            reject("format", requested.getFormat(), "call returns WAV and stream returns PCM16");
            if (requested.getSpeed() != null) speed = requested.getSpeed();
        }
        return speed == null ? SpeechOptions.NONE : SpeechOptions.speed(speed);
    }

    private static void reject(String knob, String value, String why) {
        if (value != null && !value.isBlank())
            throw new UnsupportedOperationException(
                    knob + " '" + value + "' is not supported: " + why);
    }

    private void checkOpen() {
        if (closed) throw new IllegalStateException("this model is closed");
    }

    @Override
    public TextToSpeechOptions getOptions() {
        return defaultOptions;
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

        private Object source; // Path | model-ref String | SpeechSynthesisModel: last setter wins
        private SpeechSynthesisModel<?, ?, ?> model; // derived from source at build()
        private Path modelPath; // derived from source at build()
        private Double speed;
        private int maxInputChars = DEFAULT_MAX_INPUT_CHARS;
        private Map<String, Path> companionPaths; // resolved at build()
        private final Map<String, String> companionRefs = new LinkedHashMap<>();
        private final Map<String, Path> localCompanions = new LinkedHashMap<>();

        /**
         * A model you loaded yourself - the typed path, where a port's own knobs are expressible
         * ({@code InflectTTS.load(gguf, weights).variation(0.5)}). Its weights arena stays yours.
         * The model source is the last setter called: this replaces any earlier {@link
         * #model(String)} or {@link #modelPath(Path)}.
         */
        public Builder model(SpeechSynthesisModel<?, ?, ?> model) {
            this.source = model;
            return this;
        }

        /** The GGUF to load, at the port's own defaults, through architecture dispatch. */
        public Builder modelPath(Path modelPath) {
            this.source = modelPath;
            return this;
        }

        /**
         * The model as a model ref, resolved - downloading to the local cache on first use - by
         * {@link #build()}.
         *
         * <pre>{@code
         * model("unsloth/gemma-4-E2B-it-GGUF:Q8_0");
         * }</pre>
         *
         * <p>The full grammar - the default quant, pinned revisions, a file inside a repository,
         * ModelScope - is documented once in {@link com.qxotic.jinfer.hub.ModelRef}. For a file
         * already on disk use {@link #modelPath(Path)}. A URL is not a model ref: download it
         * first, then pass the path.
         */
        public Builder model(String modelRef) {
            ModelStore.requireRef(modelRef);
            this.source = modelRef;
            return this;
        }

        /** Rate multiplier for requests that do not carry one, 1.0 = the model's natural rate. */
        /** Attaches a local companion file. This method never touches the network. */
        public Builder companionPath(String capability, Path companionPath) {
            Objects.requireNonNull(capability, "capability");
            Objects.requireNonNull(companionPath, "companionPath");
            companionRefs.remove(capability);
            localCompanions.put(capability, companionPath);
            return this;
        }

        /**
         * Attaches a companion from a supported model repository - {@code "lexicon"} for an Inflect
         * model's pronunciation lexicon, as {@code owner/repo/lexicon.bin}. The reference is
         * resolved at {@link #build()}.
         */
        public Builder companion(String capability, String companionRef) {
            Objects.requireNonNull(capability, "capability");
            if (!ModelStore.isRef(companionRef)) {
                throw new IllegalArgumentException(
                        "'"
                                + companionRef
                                + "' is not a companion model ref. Use companionPath(...) for a"
                                + " local file; download plain URLs first.");
            }
            localCompanions.remove(capability);
            companionRefs.put(capability, companionRef);
            return this;
        }

        public Builder speed(double speed) {
            if (!(speed > 0) || Double.isInfinite(speed))
                throw new IllegalArgumentException("speed must be > 0: " + speed);
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
            model = null;
            modelPath = null;
            companionPaths = Map.of();
            switch (source) {
                case SpeechSynthesisModel<?, ?, ?> m -> {
                    if (!companionRefs.isEmpty() || !localCompanions.isEmpty())
                        throw new IllegalArgumentException(
                                "companions are load-time settings; apply them when you load the"
                                        + " model passed to model(...)");
                    model = m;
                    return new JinferSpeechModel(this);
                }
                case Path path -> modelPath = path;
                case String ref -> {} // resolved below, in one batch with the companions
                case null, default ->
                        throw new IllegalArgumentException(
                                "a model is required: model(\"owner/repo:Q4_K_M\"),"
                                        + " modelPath(...) or model(SpeechSynthesisModel)");
            }
            // the model (when it is a ref) and the companions resolve in ONE batch, so a cold
            // start pays the slowest download, not the sum
            List<String> wanted = new ArrayList<>();
            if (source instanceof String ref) wanted.add(ref);
            wanted.addAll(companionRefs.values());
            List<Path> resolved =
                    wanted.isEmpty() ? List.of() : ModelStore.standard().resolveAll(wanted);
            int at = 0;
            if (modelPath == null) modelPath = resolved.get(at++);
            var attached = new LinkedHashMap<>(localCompanions);
            for (String capability : companionRefs.keySet()) {
                attached.put(capability, resolved.get(at++));
            }
            companionPaths = Collections.unmodifiableMap(attached);
            return new JinferSpeechModel(this);
        }
    }
}
