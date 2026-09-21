package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * NVIDIA Parakeet speech recognition ({@code general.architecture=parakeet}): the FastConformer
 * encoder and TDT transducer decoder behind the {@link TranscriptionModel} contract. Offline,
 * whole-utterance transcription; the TDT variant is required (a CTC-only conversion is refused at
 * load).
 */
public final class Parakeet
        implements TranscriptionModel<Parakeet.Configuration, Parakeet.Weights, Parakeet.State> {

    public static final String ARCHITECTURE = "parakeet";

    /** The load-time summary; {@code frameSeconds} is the unit of the decoder's token timing. */
    public record Configuration(
            int sampleRate, int dModel, int layers, int vocabSize, int subsamplingFactor, int hop) {
        public double frameSeconds() {
            return (double) hop * subsamplingFactor / sampleRate;
        }
    }

    public record Weights(ParakeetEncoder encoder, ParakeetTdt decoder) {}

    /**
     * A transcription pipeline's slot. Inference scratch is currently per-call, so the state holds
     * no memory yet; it exists for the contract's serial-pipeline law and will carry streaming
     * state when incremental input lands.
     */
    public static final class State extends RuntimeState {
        @Override
        protected void releaseResources() {}
    }

    private final Configuration configuration;
    private final Weights weights;
    private final String name; // for telemetry: the checkpoint's own name

    private Parakeet(Configuration configuration, Weights weights, String name) {
        this.configuration = configuration;
        this.weights = weights;
        this.name = name;
    }

    public static Parakeet load(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        Map<String, MemoryView<MemorySegment>> tensors =
                ModelLoader.loadTensors(channel, gguf, arena);
        ParakeetEncoder encoder = ParakeetEncoder.load(gguf, tensors, arena);
        ParakeetTdt decoder = ParakeetTdt.load(gguf, tensors);
        Configuration configuration =
                new Configuration(
                        gguf.getValue(int.class, "parakeet.preprocessor.sample_rate"),
                        encoder.config().dModel(),
                        encoder.config().layers(),
                        decoder.config().vocabSize(),
                        gguf.getValue(int.class, "parakeet.encoder.subsampling_factor"),
                        encoder.config().hop());
        return new Parakeet(
                configuration,
                new Weights(encoder, decoder),
                gguf.getStringOrDefault("general.name", ARCHITECTURE));
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }

    @Override
    public Weights weights() {
        return weights;
    }

    @Override
    public int sampleRate() {
        return configuration.sampleRate();
    }

    @Override
    public State newState() {
        return new State();
    }

    @Override
    public State newState(MemoryArena<MemorySegment> arena) {
        Objects.requireNonNull(arena, "arena");
        return new State(); // no borrowed buffers yet; see State
    }

    /**
     * Full-context attention is quadratic in audio length, so audio is windowed: a window commits
     * once audio beyond it arrives (its kept tokens can no longer change), neighbors overlap and
     * split the overlap at its midpoint, and the buffered tail re-decodes on demand. One window or
     * less is a single exact pass. {@code -Djinfer.parakeet.chunkSeconds} sizes the window.
     */
    private static final int OVERLAP_SECONDS = 5;

    // A window commits only once this much audio exists beyond it, and the consensus preview is
    // cut to exactly overlap+lookahead - so the anchor, and therefore the transcript, is
    // identical for every feeding pattern (0.5 s live chunks or the whole file at once).
    private static final int COMMIT_LOOKAHEAD_SECONDS = 5;

    // One window size for offline and streaming: 60 s measures within 0.1 WER of full context
    // (consensus seams + shared mel statistics + the decode watchdog), while smaller windows
    // re-expose the model's short-excerpt fragility - whole windows can normalize into silence.
    // The streaming reader thread absorbs the larger per-partial decode, so live partials still
    // pace at a few seconds.
    private static final int WINDOW_SECONDS = 60;

    private int chunkSamples(int defaultSeconds) {
        int seconds = Integer.getInteger("jinfer.parakeet.chunkSeconds", defaultSeconds);
        return Math.max(2 * OVERLAP_SECONDS, seconds) * sampleRate();
    }

    /** Offline transcription is the stream fed whole: one windowing implementation, not two. */
    @Override
    public Transcription transcribe(State state, float[] pcm) {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(pcm, "pcm");
        TranscriptionStream stream = new Stream(state, chunkSamples(WINDOW_SECONDS));
        stream.feed(pcm);
        return stream.finish();
    }

    /**
     * The one windowing implementation ({@link #transcribe} is this stream fed whole). A
     * cache-aware streaming encoder can later replace the tail re-decode behind this interface.
     */
    @Override
    public TranscriptionStream stream(State state) {
        Objects.requireNonNull(state, "state");
        return new Stream(state, chunkSamples(WINDOW_SECONDS));
    }

    private final class Stream implements TranscriptionStream {
        private final State state;
        private final int chunk;
        private final ParakeetEncoder.MelStats melStats = new ParakeetEncoder.MelStats();
        private float[] buffer = new float[sampleRate() * 8];
        private int buffered;
        private long windowStart; // absolute sample index of buffer[0]
        private double committedUpTo; // absolute seconds: tokens starting before this are final
        private final List<ParakeetTdt.Emission> committed = new ArrayList<>();
        private final List<Transcription.Token> committedTokens = new ArrayList<>();
        private boolean done;
        // One utterance = one jinfer.Inference event, committed at finish. The event spans the
        // stream's life - for offline transcribe that IS the call - while decodeTime carries only
        // compute, so a live stream's waiting for audio is not billed as inference.
        private final InferenceEvent event;
        private long totalFed; // samples
        private long computeNanos;

        private Stream(State state, int chunk) {
            this.state = state;
            this.chunk = chunk;
            this.event =
                    InferenceEvent.started(name, InferenceEvent.TRANSCRIPTION, InferenceEvent.TEXT);
        }

        /** Every encoder+decoder pass goes through here so the event sees all of the compute. */
        private Window timed(
                float[] pcm,
                int from,
                int length,
                ParakeetEncoder.MelStats stats,
                boolean persist) {
            long started = System.nanoTime();
            try {
                return window(pcm, from, length, stats, persist);
            } finally {
                computeNanos += System.nanoTime() - started;
            }
        }

        @Override
        public void feed(float[] pcm, int offset, int length) {
            Objects.requireNonNull(pcm, "pcm");
            Objects.checkFromIndexSize(offset, length, pcm.length);
            if (done) throw new IllegalStateException("stream is finished");
            state.exclusively(
                    () -> {
                        if (buffered + length > buffer.length) {
                            int grown = buffer.length;
                            while (grown < buffered + length) grown *= 2;
                            float[] wider = new float[grown];
                            System.arraycopy(buffer, 0, wider, 0, buffered);
                            buffer = wider;
                        }
                        System.arraycopy(pcm, offset, buffer, buffered, length);
                        buffered += length;
                        totalFed += length;
                        int step = chunk - OVERLAP_SECONDS * sampleRate();
                        int lookahead = COMMIT_LOOKAHEAD_SECONDS * sampleRate();
                        while (buffered >= chunk + lookahead) {
                            commitWindow(step);
                        }
                    });
        }

        /**
         * Commits the oldest window by CONSENSUS: the old window and the new window's prefix both
         * transcribe the overlap, and the commit horizon lands in a gap inside their longest
         * agreeing token run, so both sides attribute every boundary word identically. Measured
         * seam cost: ~0.3 errors per seam, from ~2.5 under a midpoint time split. Midpoint remains
         * the fallback for an overlap the decoders cannot agree on (silence, music).
         */
        private void commitWindow(int step) {
            Window old = timed(buffer, 0, chunk, melStats, true);
            // fixed-length preview: determinism requires the same lookahead every time
            int preview = (OVERLAP_SECONDS + COMMIT_LOOKAHEAD_SECONDS) * sampleRate();
            Window next = timed(buffer, step, Math.min(buffered - step, preview), melStats, false);
            double anchor = consensusAnchor(old, next, step);
            keep(old, 0, committedUpTo, anchor, committed, committedTokens);
            committedUpTo = anchor;
            System.arraycopy(buffer, step, buffer, 0, buffered - step);
            buffered -= step;
            windowStart += step;
        }

        /**
         * The cut inside the overlap {@code [windowStart+step, windowStart+chunk]}: the midpoint of
         * the middle gap of the longest run of tokens both decodes agree on (same id, frames within
         * 3), or the overlap midpoint when no two-token run agrees.
         */
        private double consensusAnchor(Window old, Window next, int step) {
            double overlapFrom =
                    Math.max(committedUpTo, (windowStart + step) / (double) sampleRate());
            double overlapTo = (windowStart + chunk) / (double) sampleRate();
            List<double[]> gaps = new ArrayList<>(); // start times of agreed consecutive pairs
            int j = 0;
            int runLength = 0;
            double previousStart = 0;
            for (int i = 0; i < old.emissions().size(); i++) {
                double start = old.tokens().get(i).start() + windowStart / (double) sampleRate();
                if (start < overlapFrom || start >= overlapTo) continue;
                // two-pointer scan for the same token at nearly the same absolute time
                boolean matched = false;
                double matchedStart = 0;
                for (; j < next.emissions().size(); j++) {
                    double nextStart =
                            next.tokens().get(j).start()
                                    + (windowStart + step) / (double) sampleRate();
                    if (nextStart < start - 0.25) continue;
                    if (nextStart > start + 0.25) break;
                    if (next.emissions().get(j).token() == old.emissions().get(i).token()) {
                        matched = true;
                        matchedStart = start;
                        j++;
                        break;
                    }
                }
                if (matched) {
                    if (runLength > 0) gaps.add(new double[] {previousStart, matchedStart});
                    runLength++;
                    previousStart = matchedStart;
                } else {
                    runLength = 0;
                }
            }
            if (gaps.isEmpty())
                return Math.max(committedUpTo, (overlapFrom + overlapTo) / 2); // fallback
            double[] middle = gaps.get(gaps.size() / 2);
            return (middle[0] + middle[1]) / 2;
        }

        /** Appends the window's tokens that start inside {@code [keepFrom, keepTo)}, retimed. */
        private void keep(
                Window window,
                long windowOffsetSamples,
                double keepFrom,
                double keepTo,
                List<ParakeetTdt.Emission> emissions,
                List<Transcription.Token> tokens) {
            double offset = (windowStart + windowOffsetSamples) / (double) sampleRate();
            for (int i = 0; i < window.emissions().size(); i++) {
                Transcription.Token token = window.tokens().get(i);
                double start = token.start() + offset;
                if (start < keepFrom || start >= keepTo) continue;
                emissions.add(window.emissions().get(i));
                tokens.add(
                        new Transcription.Token(
                                token.text(), start, token.end() + offset, token.confidence()));
            }
        }

        @Override
        public Transcription committed() {
            if (done) throw new IllegalStateException("stream is finished");
            return state.exclusively(
                    () ->
                            new Transcription(
                                    weights.decoder().text(committed),
                                    List.copyOf(committedTokens)));
        }

        @Override
        public Transcription partial() {
            if (done) throw new IllegalStateException("stream is finished");
            return state.exclusively(this::assemble);
        }

        @Override
        public Transcription finish() {
            if (done) throw new IllegalStateException("stream is finished");
            return state.exclusively(
                    () -> {
                        Transcription last = assemble();
                        done = true;
                        buffer = new float[0];
                        // encoder frames of unique audio: the transcription analog of input tokens
                        long samplesPerFrame =
                                (long) configuration.hop() * configuration.subsamplingFactor();
                        event.inputTokens =
                                (int) Math.min(Integer.MAX_VALUE, totalFed / samplesPerFrame);
                        event.outputTokens = last.tokens().size();
                        event.decodeTime = computeNanos;
                        event.finishReason = "stop";
                        event.end();
                        event.commit();
                        return last;
                    });
        }

        /** Committed windows plus a decode of the buffered tail - the offline last-window rule. */
        private Transcription assemble() {
            List<ParakeetTdt.Emission> emissions = new ArrayList<>(committed);
            List<Transcription.Token> tokens = new ArrayList<>(committedTokens);
            if (buffered > 0)
                keep(
                        timed(buffer, 0, buffered, melStats, false),
                        0,
                        committedUpTo,
                        Double.MAX_VALUE,
                        emissions,
                        tokens);
            return new Transcription(weights.decoder().text(emissions), tokens);
        }

        @Override
        public void close() {
            done = true;
            buffer = new float[0];
        }
    }

    private record Window(List<ParakeetTdt.Emission> emissions, List<Transcription.Token> tokens) {}

    /** One window transcribed whole; tokens are timed relative to the window start. */
    private Window window(
            float[] pcm, int from, int length, ParakeetEncoder.MelStats stats, boolean persist) {
        float[] slice = pcm;
        if (from != 0 || length != pcm.length) {
            slice = new float[length];
            System.arraycopy(pcm, from, slice, 0, length);
        }
        ParakeetEncoder.Output encoded = weights.encoder().forward(slice, stats, persist);
        ParakeetTdt decoder = weights.decoder();
        float[] projected = decoder.encProjection(encoded.data(), encoded.frames());
        List<ParakeetTdt.Emission> emissions = decoder.decode(projected, encoded.frames());
        double frameSeconds = configuration.frameSeconds();
        String[] pieces = decoder.config().pieces();
        List<ParakeetTdt.Emission> textual = new ArrayList<>(emissions.size());
        List<Transcription.Token> tokens = new ArrayList<>(emissions.size());
        for (ParakeetTdt.Emission emission : emissions) {
            String piece = pieces[emission.token()];
            if (!piece.isEmpty()
                    && ((piece.startsWith("<") && piece.endsWith(">"))
                            || (piece.startsWith("[") && piece.endsWith("]"))))
                continue; // bracketed specials carry no text
            textual.add(emission);
            tokens.add(
                    new Transcription.Token(
                            piece.replace('▁', ' '),
                            emission.frame() * frameSeconds,
                            (emission.frame() + emission.duration()) * frameSeconds,
                            emission.confidence()));
        }
        return new Window(textual, tokens);
    }
}
