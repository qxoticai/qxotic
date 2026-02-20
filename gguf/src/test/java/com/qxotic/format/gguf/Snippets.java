package com.qxotic.format.gguf;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.*;
import java.nio.channels.FileChannel.MapMode;
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
        Path modelPath = Path.of("model.gguf");
        GGUF gguf = GGUF.read(modelPath);
        // --8<-- [end:read-path]
    }

    void readChannel() throws IOException {
        Path modelPath = Path.of("model.gguf");
        // --8<-- [start:read-channel]
        try (var channel = Files.newByteChannel(modelPath, StandardOpenOption.READ)) {
            GGUF gguf = GGUF.read(channel);
            System.out.println(gguf.toString(true, true));
        }
        // --8<-- [end:read-channel]
    }

    void readUrl() throws Exception {
        // --8<-- [start:read-url]
        GGUF gguf = readFromHuggingFace("unsloth", "Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf");
        // --8<-- [end:read-url]
    }

    // --8<-- [start:read-from-huggingface]

    static GGUF readFromHuggingFace(String user, String repo, String filename) throws IOException {
        URL url = new URL(String.format("https://hf.co/%s/%s/resolve/main/%s", user, repo, filename));
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
            if (type == MetadataValueType.ARRAY) {
                MetadataValueType componentType = gguf.getComponentType(key);
                System.out.println(key + ": ARRAY<" + componentType + ">");
            } else {
                System.out.println(key + ": " + type);
            }
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

    void metadataOrDefault() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        // --8<-- [start:metadata-or-default]
        // Use getValueOrDefault to provide a fallback when key is missing
        int contextLength = gguf.getValueOrDefault(int.class, "llama.context_length", 4096);
        String architecture = gguf.getValueOrDefault(String.class, "general.architecture", "llama");
        
        // Convenience method for strings
        String name = gguf.getStringOrDefault("general.name", "unknown-model");
        // --8<-- [end:metadata-or-default]
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
                    tensor.name() + ": " + tensor.ggmlType() + " " + Arrays.toString(tensor.shape()));
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

    void tensorOffset() throws IOException {
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        // --8<-- [start:tensor-offset]
        // Compute absolute file position where tensor data begins
        long position = gguf.getTensorDataOffset() + tensor.offset();
        // Or using the convenience method:
        long position2 = tensor.absoluteOffset(gguf);
        // --8<-- [end:tensor-offset]
    }

    void readTensorMmap() throws IOException {
        Path modelPath = Path.of("model.gguf");
        GGUF gguf = GGUF.read(modelPath);
        // --8<-- [start:read-tensor-mmap]
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        long absoluteOffset = tensor.absoluteOffset(gguf);
        long byteSize = tensor.byteSize();

        try (var channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            var buffer = channel.map(MapMode.READ_ONLY, absoluteOffset, byteSize);
            buffer.order(ByteOrder.nativeOrder());
            // buffer now contains the raw tensor data
            // For quantized types, you'll need to decode the data
        }
        // --8<-- [end:read-tensor-mmap]
    }

    void readTensorChannel() throws IOException {
        Path modelPath = Path.of("model.gguf");
        GGUF gguf = GGUF.read(modelPath);
        // --8<-- [start:read-tensor-channel]
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        long absoluteOffset = tensor.absoluteOffset(gguf);
        long byteSize = tensor.byteSize();

        ByteBuffer buffer = ByteBuffer.allocate(Math.toIntExact(byteSize)).order(ByteOrder.nativeOrder());

        try (var channel = Files.newByteChannel(modelPath, StandardOpenOption.READ)) {
            channel.position(absoluteOffset);
            channel.read(buffer);
            buffer.flip();
            // buffer now contains the raw tensor data
        }
        // --8<-- [end:read-tensor-channel]
    }

    void readTensorByteBuffer() throws IOException {
        Path modelPath = Path.of("model.gguf");
        GGUF gguf = GGUF.read(modelPath);
        // --8<-- [start:read-tensor-bytebuffer]
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        long absoluteOffset = tensor.absoluteOffset(gguf);
        long byteSize = tensor.byteSize();

        // Allocate a direct ByteBuffer for better performance
        ByteBuffer buffer = ByteBuffer.allocateDirect(Math.toIntExact(byteSize));
        buffer.order(ByteOrder.nativeOrder());

        try (var channel = Files.newByteChannel(modelPath, StandardOpenOption.READ)) {
            channel.position(absoluteOffset);
            int bytesRead = 0;
            while (bytesRead < byteSize) {
                int read = channel.read(buffer);
                if (read < 0) break;
                bytesRead += read;
            }
        }
        buffer.flip();
        // buffer now contains the raw tensor data
        // --8<-- [end:read-tensor-bytebuffer]
    }

    void readTensorMemoryMapped() throws IOException {
        Path modelPath = Path.of("model.gguf");
        GGUF gguf = GGUF.read(modelPath);
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        // --8<-- [start:read-tensor-mmap-buffer]
        long absoluteOffset = tensor.absoluteOffset(gguf);
        long byteSize = tensor.byteSize();

        try (var channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            // Create a memory-mapped ByteBuffer
            ByteBuffer buffer = channel.map(MapMode.READ_ONLY, absoluteOffset, byteSize);
            buffer.order(ByteOrder.nativeOrder());
            // buffer is memory-mapped and data is loaded on-demand by the OS
        }
        // --8<-- [end:read-tensor-mmap-buffer]
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

    void writeTensorDataByteBuffer() throws IOException {
        // --8<-- [start:write-tensor-buffer]
        // Assume tensorData is a ByteBuffer containing the tensor bytes
        ByteBuffer tensorData = ByteBuffer.allocateDirect(1024);

        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        TensorEntry tensor = gguf.getTensor("token_embd.weight");

        try (FileChannel channel = FileChannel.open(Path.of("output.gguf"),
                StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            // Write GGUF metadata first
            GGUF.write(gguf, channel);

            // Position to tensor offset and write data
            channel.position(tensor.absoluteOffset(gguf));
            channel.write(tensorData);
        }
        // --8<-- [end:write-tensor-buffer]
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
        Builder builder = Builder.newBuilder(existing)
                .putString("general.description", "Modified model")
                .removeKey("old_key");

        GGUF modified = builder.build();
        // --8<-- [end:builder-modify]
    }

    void builderAlignment() throws IOException {
        // --8<-- [start:builder-alignment]
        GGUF gguf = GGUF.read(Path.of("model.gguf"));
        int alignment = gguf.getAlignment();
        Builder builder = Builder.newBuilder()
                .setAlignment(16 * (1 << 10)) // 16KB
                .putString("general.name", "aligned-model");
        // --8<-- [end:builder-alignment]
    }
}