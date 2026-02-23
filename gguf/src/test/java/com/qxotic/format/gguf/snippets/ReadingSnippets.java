package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.GGUF;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.SeekableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;

/** Snippets for reading GGUF files. */
@SuppressWarnings("unused")
public class ReadingSnippets {

    // --8<-- [start:read-path]
    void readFromPath() throws IOException {
        Path path = Path.of("model.gguf");
        GGUF gguf = GGUF.read(path);
    }

    // --8<-- [end:read-path]

    // --8<-- [start:read-channel]
    void readFromChannel() throws IOException {
        Path path = Path.of("model.gguf");
        try (SeekableByteChannel channel = Files.newByteChannel(path)) {
            GGUF gguf = GGUF.read(channel);
        }
    }

    // --8<-- [end:read-channel]

    // --8<-- [start:read-url]
    void readFromUrl() throws IOException {
        URL url = new URL("https://huggingface.co/user/repo/resolve/main/model.gguf");
        try (ReadableByteChannel channel =
                Channels.newChannel(new BufferedInputStream(url.openStream()))) {
            GGUF gguf = GGUF.read(channel);
        }
    }
    // --8<-- [end:read-url]
}
