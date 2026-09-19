package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/** The parakeet.cpp-generated parity fixtures in {@code test-fixtures/parakeet/}. */
final class Fixtures {

    private Fixtures() {}

    /** Every ported model tag with committed fixtures; parity tests iterate these. */
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
}
