package com.qxotic.format.safetensors;

import java.io.BufferedInputStream;
import java.io.IOException;
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
import java.util.Map;

/**
 * Code snippets for Safetensors documentation.
 *
 * <p>Snippet markers use pymdownx.snippets format: {@code [start:name]} / {@code [end:name]}.
 */
@SuppressWarnings("unused")
class Snippets {

    void readPath() throws IOException {
        // --8<-- [start:read-path]
        Path path = Path.of("model.safetensors");
        Safetensors st = Safetensors.read(path);
        // --8<-- [end:read-path]
    }

    void readChannel() throws IOException {
        // --8<-- [start:read-channel]
        try (var channel = Files.newByteChannel(Path.of("model.safetensors"))) {
            Safetensors st = Safetensors.read(channel);
            System.out.println(st);
        }
        // --8<-- [end:read-channel]
    }

    void readUrl() throws Exception {
        // --8<-- [start:read-url]
        URL url = new URL("https://huggingface.co/user/repo/resolve/main/model.safetensors");
        try (var channel = Channels.newChannel(new BufferedInputStream(url.openStream()))) {
            Safetensors st = Safetensors.read(channel);
        }
        // --8<-- [end:read-url]
    }

    // --8<-- [start:read-from-huggingface]
    static Safetensors readFromHuggingFace(String user, String repo, String filename)
            throws IOException {
        URL url =
                new URL(
                        String.format("https://huggingface.co/%s/%s/resolve/main/%s", user, repo, filename));
        try (var channel = Channels.newChannel(new BufferedInputStream(url.openStream()))) {
            return Safetensors.read(channel);
        }
    }

    // --8<-- [end:read-from-huggingface]

    void basicInfo() throws IOException {
        Safetensors st = Safetensors.read(Path.of("model.safetensors"));
        // --8<-- [start:basic-info]
        long tensorDataOffset = st.getTensorDataOffset();
        int alignment = st.getAlignment();
        boolean hasTensor = st.containsTensor("model.embed_tokens.weight");
        // --8<-- [end:basic-info]
    }

    void metadata() throws IOException {
        Safetensors st = Safetensors.read(Path.of("model.safetensors"));
        // --8<-- [start:metadata]
        Map<String, String> metadata = st.getMetadata();
        String format = metadata.get("format");
        // --8<-- [end:metadata]
    }

    void metadataKeys() throws IOException {
        Safetensors st = Safetensors.read(Path.of("model.safetensors"));
        // --8<-- [start:metadata-keys]
        for (String key : st.getMetadata().keySet()) {
            String value = st.getMetadata().get(key);
            System.out.println(key + ": " + value);
        }
        // --8<-- [end:metadata-keys]
    }

    void tensors() throws IOException {
        Safetensors st = Safetensors.read(Path.of("model.safetensors"));
        // --8<-- [start:tensors]
        for (TensorEntry tensor : st.getTensors()) {
            System.out.println(
                    tensor.name()
                            + " "
                            + tensor.dtype()
                            + " "
                            + Arrays.toString(tensor.shape())
                            + " @ "
                            + tensor.byteOffset());
        }
        // --8<-- [end:tensors]
    }

    void tensorOne() throws IOException {
        Safetensors st = Safetensors.read(Path.of("model.safetensors"));
        // --8<-- [start:tensor-one]
        TensorEntry tensor = st.getTensor("model.embed_tokens.weight");
        if (tensor != null) {
            long byteSize = tensor.byteSize();
            long absoluteOffset = st.getTensorDataOffset() + tensor.byteOffset();
        }
        // --8<-- [end:tensor-one]
    }

    void builderCreate() {
        // --8<-- [start:builder-create]
        Safetensors st =
                Builder.newBuilder()
                        .putMetadataKey("format", "pt")
                        .putTensor(TensorEntry.create("weight", DType.F32, new long[] {2, 2}, 0))
                        .build();
        // --8<-- [end:builder-create]
    }

    void builderModify() throws IOException {
        Safetensors existing = Safetensors.read(Path.of("model.safetensors"));
        // --8<-- [start:builder-modify]
        Safetensors modified =
                Builder.newBuilder(existing)
                        .putMetadataKey("format", "pt")
                        .removeMetadataKey("old_key")
                        .build(false);
        // --8<-- [end:builder-modify]
    }

    void builderAlignment() {
        // --8<-- [start:builder-alignment]
        Builder builder = Builder.newBuilder().setAlignment(64);
        int alignment = builder.getAlignment();
        // --8<-- [end:builder-alignment]
    }

    void writeFile() throws IOException {
        Safetensors st = Builder.newBuilder().build();
        // --8<-- [start:write-file]
        Safetensors.write(st, Path.of("output.safetensors"));
        // --8<-- [end:write-file]
    }

    void writeChannel() throws IOException {
        Safetensors st = Builder.newBuilder().build();
        // --8<-- [start:write-channel]
        try (WritableByteChannel channel =
                Files.newByteChannel(
                        Path.of("output.safetensors"),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
            Safetensors.write(st, channel);
        }
        // --8<-- [end:write-channel]
    }

    void indexLoad() throws IOException {
        // --8<-- [start:index-load]
        SafetensorsIndex index = SafetensorsIndex.load(Path.of("/path/to/model"));
        Path shard = index.getSafetensorsPath("model.layers.0.self_attn.q_proj.weight");
        // --8<-- [end:index-load]
    }

    void errorHandling() {
        // --8<-- [start:error-handling]
        try {
            Safetensors st = Safetensors.read(Path.of("corrupted.safetensors"));
        } catch (SafetensorsFormatException e) {
            System.err.println("Invalid format: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("I/O error: " + e.getMessage());
        }
        // --8<-- [end:error-handling]
    }

    void readTensorByteBuffer() throws IOException {
        Safetensors st = Safetensors.read(Path.of("model.safetensors"));
        TensorEntry tensor = st.getTensor("weight");
        // --8<-- [start:read-tensor-bytebuffer]
        long absoluteOffset = st.absoluteOffset(tensor);
        long byteSize = tensor.byteSize();

        try (var channel =
                FileChannel.open(Path.of("model.safetensors"), StandardOpenOption.READ)) {
            channel.position(absoluteOffset);
            ByteBuffer buffer = ByteBuffer.allocate((int) byteSize);
            channel.read(buffer);
            buffer.flip();
            // buffer now contains raw tensor data
        }
        // --8<-- [end:read-tensor-bytebuffer]
    }

    void readTensorMmapBuffer() throws IOException {
        Safetensors st = Safetensors.read(Path.of("model.safetensors"));
        TensorEntry tensor = st.getTensor("weight");
        // --8<-- [start:read-tensor-mmap-buffer]
        long absoluteOffset = st.absoluteOffset(tensor);
        long byteSize = tensor.byteSize();

        try (var channel =
                FileChannel.open(Path.of("model.safetensors"), StandardOpenOption.READ)) {
            ByteBuffer buffer =
                    channel.map(FileChannel.MapMode.READ_ONLY, absoluteOffset, byteSize);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            // buffer provides direct access to tensor data (zero-copy)
        }
        // --8<-- [end:read-tensor-mmap-buffer]
    }

    void writeTensorBuffer() throws IOException {
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("weight", DType.F32, new long[] {1024}, 0))
                        .build();
        TensorEntry tensor = st.getTensor("weight");
        // --8<-- [start:write-tensor-buffer]
        ByteBuffer tensorData = ByteBuffer.allocate((int) tensor.byteSize());
        tensorData.order(ByteOrder.LITTLE_ENDIAN);
        // ... fill tensorData with your values ...

        try (var channel =
                FileChannel.open(
                        Path.of("output.safetensors"),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
            Safetensors.write(st, channel);
            channel.position(st.absoluteOffset(tensor));
            channel.write(tensorData);
        }
        // --8<-- [end:write-tensor-buffer]
    }
}
