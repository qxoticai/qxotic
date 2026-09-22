// langchain4j AudioTranscriptionModel backed by jinfer: in-process CPU speech recognition over a
// local GGUF, no server. Names no port - either you pass a path and architecture dispatch finds
// one, or you pass a model you loaded yourself.
package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.media.Media;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.audio.AudioTranscriptionModel;
import dev.langchain4j.model.audio.AudioTranscriptionRequest;
import dev.langchain4j.model.audio.AudioTranscriptionResponse;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * langchain4j {@code AudioTranscriptionModel} backed by jinfer: in-process CPU speech recognition
 * over a local GGUF.
 *
 * <p>Thread-safe: concurrent requests use independent transcription states and may run in parallel.
 * {@link #close()} waits for in-flight requests before freeing owned weights.
 */
public final class JinferTranscriptionModel implements AudioTranscriptionModel, AutoCloseable {

    private final TranscriptionModel<?, ?, ?> model;
    private final Arena owned; // null unless this instance loaded the weights
    // Requests take the READ lock and run in PARALLEL - a state is per-call. close() takes the
    // WRITE lock, so it waits for every in-flight transcription before freeing the weights arena.
    private final ReentrantReadWriteLock lifecycle = new ReentrantReadWriteLock();
    private volatile boolean closed;

    private JinferTranscriptionModel(Builder b) {
        Arena created = b.model == null ? Arenas.newCrossThread() : null;
        try {
            this.model = b.model != null ? b.model : Models.loadTranscription(b.modelPath, created);
        } catch (IOException e) {
            closeQuietly(created); // a leaked ofShared arena has no backstop: free before failing
            throw new UncheckedIOException("failed to load " + b.modelPath, e);
        } catch (RuntimeException | Error e) {
            closeQuietly(created);
            throw e;
        }
        this.owned = created; // a caller's model stays the caller's
    }

    private static void closeQuietly(Arena arena) {
        if (arena != null) Arenas.close(arena);
    }

    @Override
    public AudioTranscriptionResponse transcribe(AudioTranscriptionRequest request) {
        Objects.requireNonNull(request, "request must not be null");
        // A silently ignored request parameter is a lie; this model decodes what it decodes.
        if (request.prompt() != null)
            throw new UnsupportedFeatureException("prompts are not supported by this model");
        if (request.temperature() != null)
            throw new UnsupportedFeatureException("temperature is not supported (greedy decoding)");
        if (request.language() != null)
            throw new UnsupportedFeatureException(
                    "this model detects the language itself; requested '"
                            + request.language()
                            + "'");
        byte[] bytes = audioBytes(request);
        return AudioTranscriptionResponse.from(transcribe(bytes).text());
    }

    private static byte[] audioBytes(AudioTranscriptionRequest request) {
        var audio = Objects.requireNonNull(request.audio(), "request audio must not be null");
        if (audio.binaryData() != null) return audio.binaryData();
        if (audio.base64Data() != null) return Base64.getDecoder().decode(audio.base64Data());
        throw new IllegalArgumentException(
                "audio must carry binaryData or base64Data - jinfer does not fetch media from a"
                        + " URL during inference; download it first");
    }

    /** Transcribes an audio file (any format {@code jinfer-codecs} decodes). */
    public Transcription transcribe(Path audioFile) {
        try {
            return transcribe(AudioCodec.load(audioFile));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to read " + audioFile, e);
        }
    }

    /** Transcribes encoded audio bytes (any format {@code jinfer-codecs} decodes). */
    public Transcription transcribe(byte[] audio) {
        try {
            return transcribe(AudioCodec.decode(audio));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to decode audio", e);
        }
    }

    /** Transcribes already-decoded audio: mono at the model's sample rate, refused otherwise. */
    public Transcription transcribe(Media.Audio audio) {
        lifecycle.readLock().lock(); // shared: concurrent requests proceed in parallel
        try {
            if (closed) throw new IllegalStateException("this model is closed");
            // ONE STATE PER CALL: a jinfer transcription state is a single serial pipeline, and
            // the honest way to meet that contract is not to share one. The core door also
            // validates the audio shape, refusing anything but mono at the model's rate.
            return model.transcribe(audio);
        } finally {
            lifecycle.readLock().unlock();
        }
    }

    /**
     * Idempotent, BLOCKING close: returns only after in-flight transcriptions finish, and frees the
     * weights arena IFF this instance created it - a model you passed in stays yours, so close
     * yours after this one, never before. Requests after this fail loudly.
     */
    @Override
    public void close() {
        lifecycle.writeLock().lock(); // BLOCKS until every in-flight transcription has returned
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

        private Object source; // Path | model-ref String | TranscriptionModel: last setter wins
        private TranscriptionModel<?, ?, ?> model; // derived from source at build()
        private Path modelPath; // derived from source at build()

        /**
         * A model you loaded yourself - the typed path, where a port's own knobs are expressible.
         * Its weights arena stays yours. The model source is the last setter called.
         */
        public Builder model(TranscriptionModel<?, ?, ?> model) {
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
         * model("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf");
         * }</pre>
         *
         * <p>The full grammar is documented once in {@link com.qxotic.jinfer.hub.ModelRef}. For a
         * file already on disk use {@link #modelPath(Path)}.
         */
        public Builder model(String modelRef) {
            ModelStore.requireRef(modelRef);
            this.source = modelRef;
            return this;
        }

        public JinferTranscriptionModel build() {
            model = null;
            modelPath = null;
            switch (source) {
                case TranscriptionModel<?, ?, ?> m -> model = m;
                case Path path -> modelPath = path;
                case String ref -> modelPath = ModelStore.standard().resolve(ref);
                case null, default ->
                        throw new IllegalArgumentException(
                                "a model is required: model(\"owner/repo/file.gguf\"),"
                                        + " modelPath(...) or model(TranscriptionModel)");
            }
            return new JinferTranscriptionModel(this);
        }
    }
}
