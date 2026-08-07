package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Timer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

final class AOT {
    // The preloaded model's parsed GGUF (metadata + tensor descriptors), baked at class-init. In a
    // native image (AOT class initialized-at-build-time) this skips re-reading and re-parsing the
    // header at startup; the tensor data is still mmap'd at runtime. Arch-agnostic: any new-API
    // port loads from it via Models.load(fileChannel, ctx).
    //
    // Tradeoff vs the old per-model AOT: that one baked the fully materialized tokenizer + config
    // and only mmap'd weights at runtime. This generic version bakes the parsed GGUF and rebuilds
    // the tokenizer at runtime (Models.load re-materializes it), so the win is skipping the header
    // parse, not the tokenizer build. A fuller bake would need a per-port "attach weights to a
    // preloaded config-only model" method across all ports; deferred.
    record PartialModel(String modelFileName, GGUF gguf) {}

    private static final PartialModel PRELOADED_GGUF =
            preLoadGGUF(System.getProperty("jinfer.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, path.toString());
                return new PartialModel(path.getFileName().toString(), gguf);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The preloaded model when {@code modelPath} matches the baked one, else null (the caller falls
     * back to {@link Models#load(Path)}). Reuses the baked GGUF, so only the tensor data is read.
     */
    static LoadedModel<?> tryUsePreLoaded(Path modelPath) throws IOException {
        PartialModel preLoaded = PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        if (!Objects.equals(modelPath.getFileName().toString(), preLoaded.modelFileName())) {
            return null;
        }
        try (var timer = Timer.log("Load tensors from pre-loaded model");
                FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            return Models.load(fileChannel, preLoaded.gguf(), java.lang.foreign.Arena.global());
        }
    }
}
