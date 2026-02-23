package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/** Snippets for writing GGUF files. */
@SuppressWarnings("unused")
public class WritingSnippets {

    // --8<-- [start:write-file]
    void writeToFile() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        GGUF.write(gguf, Paths.get("output.gguf"));
    }

    // --8<-- [end:write-file]

    // --8<-- [start:write-channel]
    void writeToChannel() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        try (WritableByteChannel channel =
                Files.newByteChannel(
                        Paths.get("output.gguf"),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
            GGUF.write(gguf, channel);
        }
    }

    // --8<-- [end:write-channel]

    // --8<-- [start:write-tensor-buffer]
    void writeTensorData() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        Path outputPath = Paths.get("output.gguf");

        try (FileChannel channel =
                FileChannel.open(
                        outputPath,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
            GGUF.write(gguf, channel);

            for (TensorEntry tensor : gguf.getTensors()) {
                ByteBuffer data = ByteBuffer.allocate((int) tensor.byteSize());
                channel.position(tensor.absoluteOffset(gguf));
                channel.write(data);
            }
        }
    }
    // --8<-- [end:write-tensor-buffer]
}
