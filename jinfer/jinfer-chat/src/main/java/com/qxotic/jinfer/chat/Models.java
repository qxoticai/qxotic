package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.ServiceLoader;

/**
 * Loads any generative model, dispatching on {@code general.architecture} to the matching port via
 * {@link ModelProvider} services - the ports on the classpath define what is loadable. The one
 * "path to model" entry every consumer (server, CLI, benches) shares.
 */
public final class Models {

    private Models() {}

    private static final List<ModelProvider> PROVIDERS =
            ServiceLoader.load(ModelProvider.class).stream()
                    .map(ServiceLoader.Provider::get)
                    .toList();

    /** Loads {@code path} at context size {@code ctx} (-1 = the model's full context). */
    public static LoadedModel<?> load(Path path, int ctx) throws IOException {
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            fc.position(0L);
            GGUF gguf =
                    GGUF.read(
                            Channels.newChannel(
                                    new BufferedInputStream(Channels.newInputStream(fc), 1 << 20)));
            return load(fc, gguf, ctx);
        }
    }

    /**
     * As {@link #load(Path, int)} but reusing an already-parsed {@code gguf} (the header is not
     * re-read) - used by AOT preload. {@code fileChannel} supplies the tensor data to mmap.
     */
    public static LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int ctx)
            throws IOException {
        String arch = gguf.getString("general.architecture");
        for (ModelProvider p : PROVIDERS) {
            if (p.supports(arch)) return p.load(fileChannel, gguf, ctx);
        }
        throw new IllegalArgumentException(
                "unsupported architecture '"
                        + arch
                        + "' ("
                        + PROVIDERS.size()
                        + " ports on the classpath)");
    }
}
