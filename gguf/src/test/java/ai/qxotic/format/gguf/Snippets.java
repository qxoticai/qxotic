package ai.qxotic.format.gguf;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

/**
 * Code snippets for GGUF documentation. Snippet markers use pymdownx.snippets format: [start:name]
 * / [end:name]
 *
 * <p>These methods are not tests - they exist to ensure snippets compile.
 */
@SuppressWarnings("unused")
class Snippets {

    void quickExample() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:quick-example]
        // Access metadata
        String name = gguf.getValue(String.class, "general.name");
        int contextLength = gguf.getValue(int.class, "llama.context_length");

        // List tensors
        for (TensorEntry tensor : gguf.getTensors()) {
            System.out.println(tensor.name() + ": " + tensor.ggmlType());
        }
        // --8<-- [end:quick-example]
    }

    void readPath() throws IOException {
        // --8<-- [start:read-path]
        Path path = Path.of("model.gguf");
        GGUF gguf = GGUF.read(path);
        // --8<-- [end:read-path]
    }

    void readChannel() throws IOException {
        // --8<-- [start:read-channel]
        try (var channel = Files.newByteChannel(Path.of("model.gguf"))) {
            GGUF gguf = GGUF.read(channel);
            System.out.println(gguf);
        }
        // --8<-- [end:read-channel]
    }

    void readUrl() throws Exception {
        // --8<-- [start:read-url]
        URL url = new URL("https://huggingface.co/user/repo/resolve/main/model.gguf");
        try (var channel = Channels.newChannel(new BufferedInputStream(url.openStream()))) {
            GGUF gguf = GGUF.read(channel);
        }
        // --8<-- [end:read-url]
    }

    void readFromHuggingFace() {
        // This demonstrates a reusable helper method pattern
    }

    // --8<-- [start:read-from-huggingface]
    static GGUF readFromHuggingFace(String user, String repo, String filename) throws IOException {
        URL url =
                new URL(
                        "https://huggingface.co/%s/%s/resolve/main/%s"
                                .formatted(user, repo, filename));
        try (var channel = Channels.newChannel(new BufferedInputStream(url.openStream()))) {
            return GGUF.read(channel);
        }
    }

    // --8<-- [end:read-from-huggingface]

    void basicInfo() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:basic-info]
        int version = gguf.getVersion();
        int alignment = gguf.getAlignment();
        long tensorDataOffset = gguf.getTensorDataOffset();
        // --8<-- [end:basic-info]
    }

    void metadataKeys() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:metadata-keys]
        for (String key : gguf.getMetadataKeys()) {
            MetadataValueType type = gguf.getType(key);
            System.out.println(key + ": " + type);
        }
        // --8<-- [end:metadata-keys]
    }

    void metadataAccess() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:metadata-access]
        String name = gguf.getValue(String.class, "general.name");
        int contextLength = gguf.getValue(int.class, "llama.context_length");
        float ropeTheta = gguf.getValue(float.class, "llama.rope.freq_base");
        // --8<-- [end:metadata-access]
    }

    void metadataArrays() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:metadata-arrays]
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        float[] embedding = gguf.getValue(float[].class, "embedding");
        int[] layerSizes = gguf.getValue(int[].class, "layer_sizes");
        // --8<-- [end:metadata-arrays]
    }

    void metadataCheck() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:metadata-check]
        if (gguf.containsKey("llama.context_length")) {
            int ctxLen = gguf.getValue(int.class, "llama.context_length");
        }
        // --8<-- [end:metadata-check]
    }

    void tensorAccess() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:tensor-access]
        for (TensorEntry tensor : gguf.getTensors()) {
            System.out.println(
                    tensor.name()
                            + ": "
                            + tensor.ggmlType()
                            + " "
                            + Arrays.toString(tensor.shape()));
        }
        // --8<-- [end:tensor-access]
    }

    void tensorInfo() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:tensor-info]
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        GGMLType type = tensor.ggmlType();
        long[] shape = tensor.shape();
        long offset = tensor.offset();
        long byteSize = tensor.byteSize();
        // --8<-- [end:tensor-info]
    }

    void readTensorMmap() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        // --8<-- [start:read-tensor-mmap]
        long absoluteOffset = gguf.getTensorDataOffset() + tensor.offset();
        long byteSize = tensor.byteSize();

        try (var raf = new RandomAccessFile("model.gguf", "r");
                var channel = raf.getChannel()) {
            var buffer = channel.map(FileChannel.MapMode.READ_ONLY, absoluteOffset, byteSize);
            buffer.order(ByteOrder.nativeOrder());
            // buffer now contains the raw tensor data
            // For quantized types, you'll need to decode the data
        }
        // --8<-- [end:read-tensor-mmap]
    }

    void readTensorChannel() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        // --8<-- [start:read-tensor-channel]
        long absoluteOffset = gguf.getTensorDataOffset() + tensor.offset();
        long byteSize = tensor.byteSize();

        ByteBuffer buffer = ByteBuffer.allocate((int) byteSize).order(ByteOrder.nativeOrder());

        try (var channel = Files.newByteChannel(Path.of("model.gguf"))) {
            channel.position(absoluteOffset);
            channel.read(buffer);
            buffer.flip();
            // buffer now contains the raw tensor data
        }
        // --8<-- [end:read-tensor-channel]
    }

    void writeFile() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:write-file]
        GGUF.write(gguf, Path.of("output.gguf"));
        // --8<-- [end:write-file]
    }

    void writeChannel() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:write-channel]
        try (WritableByteChannel channel =
                Files.newByteChannel(
                        Path.of("output.gguf"),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE)) {
            GGUF.write(gguf, channel);
        }
        // --8<-- [end:write-channel]
    }

    void builderCreate() throws IOException {
        // --8<-- [start:builder-create]
        Builder builder =
                Builder.newBuilder()
                        .putString("general.name", "my-model")
                        .putString("general.architecture", "llama")
                        .putInteger("llama.context_length", 4096)
                        .putFloat("llama.rope.freq_base", 10000.0f)
                        .putTensor(
                                TensorEntry.create(
                                        "token_embd.weight",
                                        new long[] {32000, 4096},
                                        GGMLType.F16,
                                        0));

        GGUF newGguf = builder.build();
        GGUF.write(newGguf, Path.of("model.gguf"));
        // --8<-- [end:builder-create]
    }

    void builderModify() throws IOException {
        // --8<-- [start:builder-modify]
        GGUF existing = GGUF.read(Path.of("model.gguf"));
        Builder builder =
                Builder.newBuilder(existing)
                        .putString("general.description", "Modified model")
                        .removeKey("old_key");

        GGUF modified = builder.build();
        // --8<-- [end:builder-modify]
    }

    void builderAlignment() {
        // --8<-- [start:builder-alignment]
        Builder builder =
                Builder.newBuilder().setAlignment(64).putString("general.name", "aligned-model");
        // --8<-- [end:builder-alignment]
    }
}
