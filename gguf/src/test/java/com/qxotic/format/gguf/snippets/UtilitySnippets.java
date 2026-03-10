package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Utility snippets and helper methods. */
@SuppressWarnings("unused")
public class UtilitySnippets {

    // --8<-- [start:read-from-huggingface]
    public static GGUF readFromHuggingFace(String user, String repo, String filename)
            throws IOException {
        URL url =
                new URL(
                        String.format("https://huggingface.co/%s/%s/resolve/main/%s", user, repo, filename));
        try (ReadableByteChannel channel =
                Channels.newChannel(new BufferedInputStream(url.openStream()))) {
            return GGUF.read(channel);
        }
    }

    // --8<-- [end:read-from-huggingface]

    // --8<-- [start:inspector-complete]
    public static class GGUFInspector {
        public static void main(String[] args) throws Exception {
            if (args.length < 1) {
                System.out.println("Usage: java GGUFInspector <model.gguf>");
                System.exit(1);
            }

            Path path = Paths.get(args[0]);
            GGUF gguf = GGUF.read(path);

            System.out.println("Name: " + gguf.getValue(String.class, "general.name"));
            System.out.println(
                    "Architecture: " + gguf.getValue(String.class, "general.architecture"));

            long totalBytes = 0;
            for (TensorEntry tensor : gguf.getTensors()) {
                totalBytes += tensor.byteSize();
            }
            System.out.printf(
                    "Tensors: %d (%.2f MB)%n",
                    gguf.getTensors().size(), totalBytes / (1024.0 * 1024.0));
        }
    }
    // --8<-- [end:inspector-complete]
}
