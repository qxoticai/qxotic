package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/** Snippets for reading tensor data. */
@SuppressWarnings("unused")
public class TensorDataSnippets {

    // --8<-- [start:read-tensor-bytebuffer]
    void readWithByteBuffer() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        TensorEntry tensor = gguf.getTensor("weights");
        ByteBuffer buffer = ByteBuffer.allocate((int) tensor.byteSize());

        try (FileChannel channel =
                FileChannel.open(Paths.get("model.gguf"), StandardOpenOption.READ)) {
            channel.position(gguf.absoluteOffset(tensor));
            channel.read(buffer);
            buffer.flip();
        }
    }

    // --8<-- [end:read-tensor-bytebuffer]

    // --8<-- [start:read-tensor-mmap]
    void readWithMemoryMap() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        TensorEntry tensor = gguf.getTensor("weights");

        try (RandomAccessFile raf = new RandomAccessFile("model.gguf", "r");
                FileChannel channel = raf.getChannel()) {
            MappedByteBuffer buffer =
                    channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            gguf.absoluteOffset(tensor),
                            tensor.byteSize());
        }
    }

    // --8<-- [end:read-tensor-mmap]

    // --8<-- [start:read-all-tensors]
    void readAllTensors() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        Path path = Paths.get("model.gguf");

        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            for (TensorEntry tensor : gguf.getTensors()) {
                ByteBuffer buffer = ByteBuffer.allocate((int) tensor.byteSize());
                channel.position(gguf.absoluteOffset(tensor));
                channel.read(buffer);
                buffer.flip();
                process(tensor.name(), buffer);
            }
        }
    }

    void process(String name, ByteBuffer data) {
        // Process tensor data
    }
    // --8<-- [end:read-all-tensors]
}
