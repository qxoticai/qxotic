package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.testkit.SystemProperty;
import com.qxotic.jota.memory.MemoryArena;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.Function;

/**
 * Shared test support: the parakeet.cpp-generated parity fixtures in {@code
 * test-fixtures/parakeet/}, checkpoint loading, scratch workspaces, and streaming helpers.
 */
final class Fixtures {

    private Fixtures() {}

    /** Every ported model tag with parity fixtures; the parity tests iterate these. */
    static final String[] MODELS = {"tdt-0.6b-v3", "tdt-1.1b"};

    /** The named fixture GGUF, walking up from the working directory to the checkout root. */
    static Optional<Path> fixture(String name) {
        for (Path directory = Path.of("").toAbsolutePath();
                directory != null;
                directory = directory.getParent()) {
            Path candidate = directory.resolve("test-fixtures").resolve("parakeet").resolve(name);
            if (Files.isRegularFile(candidate)) return Optional.of(candidate);
        }
        return Optional.empty();
    }

    /** Loads a Parakeet checkpoint into {@code arena}. */
    static Parakeet load(Path model, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            return Parakeet.load(channel, ModelLoader.readGguf(channel, model.toString()), arena);
        }
    }

    /** Runs {@code body} on a workspace that lives for this call only. */
    static <T> T onScratch(Function<Workspace, T> body) {
        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            return body.apply(new Workspace(arena));
        } finally {
            Arenas.close(arena);
        }
    }

    static float[] floats(FileChannel channel, GGUF gguf, String name) throws IOException {
        ByteBuffer bytes = raw(channel, gguf, name);
        float[] values = new float[bytes.remaining() / Float.BYTES];
        bytes.asFloatBuffer().get(values);
        return values;
    }

    static int[] ints(FileChannel channel, GGUF gguf, String name) throws IOException {
        ByteBuffer bytes = raw(channel, gguf, name);
        int[] values = new int[bytes.remaining() / Integer.BYTES];
        bytes.asIntBuffer().get(values);
        return values;
    }

    private static ByteBuffer raw(FileChannel channel, GGUF gguf, String name) throws IOException {
        TensorEntry tensor =
                gguf.getTensors().stream()
                        .filter(entry -> entry.name().equals(name))
                        .findFirst()
                        .orElseThrow(() -> new IllegalStateException("fixture misses " + name));
        ByteBuffer bytes =
                ByteBuffer.allocate(Math.toIntExact(tensor.byteSize()))
                        .order(ByteOrder.LITTLE_ENDIAN);
        channel.read(bytes, gguf.getTensorDataOffset() + tensor.offset());
        bytes.flip();
        return bytes;
    }

    /** A stream's final pieces, joined into the whole transcript. */
    static final class Pieces {
        private final StringBuilder text = new StringBuilder();
        private final List<Transcription.Token> tokens = new ArrayList<>();

        void add(Transcription piece) {
            text.append(piece.text());
            tokens.addAll(piece.tokens());
        }

        Transcription joined() {
            return new Transcription(text.toString(), tokens);
        }
    }

    /** Chunks of {@code seconds}, offline and streamed, until the returned override is closed. */
    static SystemProperty chunkSeconds(int seconds) {
        return SystemProperty.override("jinfer.parakeet.chunkSeconds", Integer.toString(seconds));
    }

    /**
     * {@code pcm} streamed in random chunks of up to {@code maxChunk} samples, polling a partial
     * after one feed in {@code partialOdds} (0 never); the final pieces joined.
     */
    static Transcription streamed(
            Parakeet parakeet, float[] pcm, Random random, int maxChunk, int partialOdds) {
        try (Parakeet.State state = parakeet.newState();
                TranscriptionStream stream = parakeet.stream(state)) {
            Pieces pieces = new Pieces();
            for (int at = 0; at < pcm.length; ) {
                int chunk = Math.min(1 + random.nextInt(maxChunk), pcm.length - at);
                pieces.add(stream.feed(pcm, at, chunk));
                at += chunk;
                if (partialOdds > 0 && random.nextInt(partialOdds) == 0) stream.partial();
            }
            pieces.add(stream.finish());
            return pieces.joined();
        }
    }

    /** {@code pcm} fed to a stream at once; the final pieces joined. */
    static Transcription streamedWhole(Parakeet parakeet, float[] pcm) {
        try (Parakeet.State state = parakeet.newState();
                TranscriptionStream stream = parakeet.stream(state)) {
            Pieces pieces = new Pieces();
            pieces.add(stream.feed(pcm));
            pieces.add(stream.finish());
            return pieces.joined();
        }
    }

    /** Text and timing exactly; confidence to 1e-6, since JIT tiers round differently. */
    static void assertSameTranscript(Transcription expected, Transcription actual, String what) {
        assertEquals(expected.text(), actual.text(), what);
        assertEquals(expected.tokens().size(), actual.tokens().size(), what);
        for (int i = 0; i < actual.tokens().size(); i++) {
            Transcription.Token e = expected.tokens().get(i), a = actual.tokens().get(i);
            assertEquals(e.text(), a.text(), what);
            assertEquals(e.start(), a.start(), what);
            assertEquals(e.end(), a.end(), what);
            assertEquals(e.confidence(), a.confidence(), 1e-6, what);
        }
    }
}
