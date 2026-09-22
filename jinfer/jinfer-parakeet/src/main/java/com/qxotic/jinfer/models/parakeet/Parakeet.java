package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.LeakWatch;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * NVIDIA Parakeet speech recognition ({@code general.architecture=parakeet}): the FastConformer
 * encoder and the TDT transducer decoder. CTC-only checkpoints are refused at load.
 */
public final class Parakeet
        implements TranscriptionModel<Parakeet.Configuration, Parakeet.Weights, Parakeet.State> {

    public static final String ARCHITECTURE = "parakeet";

    public record Configuration(
            int sampleRate, int dModel, int layers, int vocabSize, int subsamplingFactor, int hop) {
        /** One encoder frame, the unit of token timing. */
        public Duration frame() {
            return Duration.ofNanos(1_000_000_000L * hop * subsamplingFactor / sampleRate);
        }
    }

    public record Weights(ParakeetEncoder encoder, ParakeetTdt decoder) {}

    /**
     * Owns the {@link Workspace} that every window draws its scratch from, so a warm state
     * transcribes without allocating.
     */
    public static final class State extends RuntimeState {
        private final MemoryArena<MemorySegment> owned; // null when the arena is borrowed
        private final MemoryArena<MemorySegment> allocator;
        private final Workspace workspace;
        private final Runnable disarm;

        private State(MemoryArena<MemorySegment> allocator, MemoryArena<MemorySegment> owned) {
            Arenas.requireCrossThread(allocator);
            this.owned = owned;
            this.allocator = allocator;
            this.workspace = new Workspace(allocator);
            this.disarm = LeakWatch.arm(this, "Parakeet.State");
        }

        /** Native plus heap buffers allocated so far: flat once the state is warm. */
        int scratchAllocations() {
            return workspace.backingAllocations() + workspace.heapAllocations();
        }

        @Override
        protected void checkResourcesAlive() {
            if (!allocator.isAlive())
                throw new IllegalStateException("the transcription state's arena has been closed");
        }

        @Override
        protected void releaseResources() {
            disarm.run();
            if (owned != null) Arenas.close(owned);
        }
    }

    // Windows are decoded as NeMo streams Parakeet: left context, a chunk and right context, of
    // which only the chunk's tokens become final, while the decoder carries its state from chunk to
    // chunk so words simply continue across them. Sizes in encoder frames (80 ms). A live stream
    // uses NeMo's 10-2-2 s, so final text trails the audio by about 4 s; offline transcription
    // uses 46 s chunks, keeping windows at 60 s. -Djinfer.parakeet.chunkSeconds overrides both.
    // A partial, provisional and asked for often, keeps only 3 s of left context: its window is
    // a third as long, and far cheaper, since attention is quadratic in it.
    private static final int LEFT_FRAMES = 125, PARTIAL_LEFT_FRAMES = 38;
    private static final int STREAM_CHUNK_FRAMES = 25, STREAM_RIGHT_FRAMES = 25;
    private static final int OFFLINE_CHUNK_FRAMES = 575, OFFLINE_RIGHT_FRAMES = 50;

    private final Configuration configuration;
    private final Weights weights;
    private final String name; // checkpoint name, for telemetry

    private Parakeet(Configuration configuration, Weights weights, String name) {
        this.configuration = configuration;
        this.weights = weights;
        this.name = name;
    }

    public static Parakeet load(FileChannel channel, GGUF gguf, Arena arena) throws IOException {
        String architecture = gguf.getValue(String.class, "general.architecture");
        if (!ARCHITECTURE.equals(architecture))
            throw new IllegalArgumentException(
                    "expected general.architecture=parakeet but was '" + architecture + "'");
        Tensors tensors = new Tensors(ModelLoader.loadTensors(channel, gguf, arena));
        ParakeetEncoder encoder = ParakeetEncoder.load(gguf, tensors);
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
        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            return new State(arena, arena);
        } catch (RuntimeException | Error e) {
            Arenas.close(arena);
            throw e;
        }
    }

    @Override
    public State newState(MemoryArena<MemorySegment> arena) {
        Objects.requireNonNull(arena, "arena");
        return new State(arena, null);
    }

    @Override
    public Transcription transcribe(State state, float[] pcm) {
        Stream stream = stream(state, OFFLINE_CHUNK_FRAMES, OFFLINE_RIGHT_FRAMES);
        Transcription head = stream.feed(pcm), tail = stream.finish();
        List<Transcription.Token> tokens = new ArrayList<>(head.tokens());
        tokens.addAll(tail.tokens());
        return new Transcription(head.text() + tail.text(), tokens);
    }

    @Override
    public TranscriptionStream stream(State state) {
        return stream(state, STREAM_CHUNK_FRAMES, STREAM_RIGHT_FRAMES);
    }

    private Stream stream(State state, int chunkFrames, int rightFrames) {
        Objects.requireNonNull(state, "state");
        Integer seconds = Integer.getInteger("jinfer.parakeet.chunkSeconds");
        if (seconds != null) chunkFrames = seconds * sampleRate() / frameSamples();
        return new Stream(state, chunkFrames, rightFrames);
    }

    /**
     * Windowed decoding, since full-context attention is quadratic in length. A chunk commits once
     * the audio reaches its end plus the right context; the buffer then keeps just the next
     * window's left context. {@link #partial()} and {@link #finish()} decode what is buffered from
     * a copy of the decoder's cursor, or the cursor itself.
     */
    private final class Stream implements TranscriptionStream {
        private final State state;
        private final int chunk, right, left; // samples
        private final ParakeetTdt.Cursor cursor;
        private float[] buffer = new float[sampleRate() * 8];
        private int buffered;
        private long bufferStart; // absolute sample of buffer[0], where the next window starts
        private long chunkEnd; // absolute sample where the next chunk to commit ends
        private int finalTokens; // emitted in final pieces so far
        private boolean done;
        // One inference event per stream, committed at finish. decodeTime counts compute only, so
        // time spent waiting for live audio is not billed.
        private final InferenceEvent event;
        private long computeNanos;

        private Stream(State state, int chunkFrames, int rightFrames) {
            this.state = state;
            this.chunk = chunkFrames * frameSamples();
            this.right = rightFrames * frameSamples();
            this.left = LEFT_FRAMES * frameSamples();
            this.chunkEnd = chunk;
            this.cursor = new ParakeetTdt.Cursor(weights.decoder().config());
            this.event =
                    InferenceEvent.started(name, InferenceEvent.TRANSCRIPTION, InferenceEvent.TEXT);
        }

        @Override
        public int sampleRate() {
            return Parakeet.this.sampleRate();
        }

        @Override
        public Transcription feed(float[] pcm, int offset, int length) {
            Objects.requireNonNull(pcm, "pcm");
            Objects.checkFromIndexSize(offset, length, pcm.length);
            requireOpen();
            return state.exclusively(
                    () -> {
                        if (buffered + length > buffer.length) {
                            int grown = buffer.length;
                            while (grown < buffered + length) grown *= 2;
                            buffer = Arrays.copyOf(buffer, grown);
                        }
                        System.arraycopy(pcm, offset, buffer, buffered, length);
                        buffered += length;
                        List<Transcription.Token> piece = new ArrayList<>();
                        while (bufferStart + buffered >= chunkEnd + right) {
                            piece.addAll(decode(bufferStart, chunkEnd + right, chunkEnd, cursor));
                            chunkEnd += chunk;
                            int dropped =
                                    (int) (Math.max(0, chunkEnd - chunk - left) - bufferStart);
                            System.arraycopy(buffer, dropped, buffer, 0, buffered - dropped);
                            buffered -= dropped;
                            bufferStart += dropped;
                        }
                        return finalPiece(piece);
                    });
        }

        @Override
        public Transcription partial() {
            requireOpen();
            long from =
                    Math.max(bufferStart, (cursor.frame() - PARTIAL_LEFT_FRAMES) * frameSamples());
            return state.exclusively(
                    () -> transcription(decode(from, end(), Long.MAX_VALUE, cursor.copy())));
        }

        @Override
        public Transcription finish() {
            requireOpen();
            return state.exclusively(
                    () -> {
                        long fed = end();
                        Transcription last =
                                finalPiece(decode(bufferStart, fed, Long.MAX_VALUE, cursor));
                        close();
                        // input tokens: encoder frames of audio fed
                        event.inputTokens = (int) Math.min(Integer.MAX_VALUE, fed / frameSamples());
                        event.outputTokens = finalTokens;
                        event.decodeTime = computeNanos;
                        event.finishReason = "stop";
                        event.end();
                        event.commit();
                        return last;
                    });
        }

        private long end() {
            return bufferStart + buffered;
        }

        /**
         * Decodes the buffered audio between absolute samples {@code from}, a frame boundary at or
         * before the cursor, and {@code to}, running the decoder from {@code cursor} up to the
         * frame at absolute sample {@code until}. Token times are from the start of the stream. The
         * state's workspace is rewound first.
         */
        private List<Transcription.Token> decode(
                long from, long to, long until, ParakeetTdt.Cursor cursor) {
            if (to <= from) return List.of();
            long started = System.nanoTime();
            try {
                Workspace workspace = state.workspace;
                workspace.rewind();
                int offset = (int) (from - bufferStart), length = (int) (to - from);
                ParakeetEncoder encoder = weights.encoder();
                MemoryView<MemorySegment> encoded =
                        encoder.encode(buffer, offset, length, null, workspace);
                int frames = encoder.frames(length), first = (int) (from / frameSamples());
                int last = (int) Math.min(first + frames, until / frameSamples());
                ParakeetTdt decoder = weights.decoder();
                float[] projected = decoder.encProjection(encoded, frames, workspace);
                List<ParakeetTdt.Emission> emissions =
                        decoder.decode(projected, first, last, cursor, workspace);
                Duration frame = configuration.frame();
                // a duration predicted near the end can overrun the audio: tokens end inside it
                Duration audioEnd = duration(to);
                String[] pieces = decoder.config().pieces();
                List<Transcription.Token> tokens = new ArrayList<>(emissions.size());
                for (ParakeetTdt.Emission emission : emissions) {
                    String piece = pieces[emission.token()];
                    if (ParakeetTdt.isSpecial(piece)) continue;
                    Duration end = frame.multipliedBy(emission.frame() + emission.duration());
                    tokens.add(
                            new Transcription.Token(
                                    piece.replace('▁', ' '),
                                    frame.multipliedBy(emission.frame()),
                                    end.compareTo(audioEnd) < 0 ? end : audioEnd,
                                    emission.confidence()));
                }
                return tokens;
            } finally {
                computeNanos += System.nanoTime() - started;
            }
        }

        private Transcription finalPiece(List<Transcription.Token> tokens) {
            Transcription piece = transcription(tokens);
            finalTokens += tokens.size();
            return piece;
        }

        /** The tokens' text keeps its leading space, except where it opens the transcript. */
        private Transcription transcription(List<Transcription.Token> tokens) {
            if (tokens.isEmpty()) return Transcription.empty();
            StringBuilder text = new StringBuilder();
            for (Transcription.Token token : tokens) text.append(token.text());
            if (finalTokens == 0 && text.charAt(0) == ' ') text.deleteCharAt(0);
            return new Transcription(text.toString(), tokens);
        }

        @Override
        public void close() {
            done = true;
            buffer = new float[0];
        }

        private void requireOpen() {
            if (done) throw new IllegalStateException("the stream is finished or closed");
        }
    }

    private int frameSamples() {
        return configuration.hop() * configuration.subsamplingFactor();
    }

    private Duration duration(long samples) {
        return Duration.ofNanos(samples * 1_000_000_000L / sampleRate());
    }
}
