package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Timer;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

final class AOT {
    // The preloaded model's baked artifacts: the parsed GGUF (metadata + tensor descriptors) and
    // the tokenizer built from it - everything derivable from the file's header, and nothing
    // else. Built at class-init, so in a native image (com.qxotic.* initializes at build time)
    // the whole record snapshots into the image heap: both pieces are pure heap, the tokenizer
    // because toknroll holds no segments, channels or arenas. The tensor data is still mmap'd at
    // runtime.
    //
    // What is NOT baked is the model object itself (config + weight wiring): that would need a
    // per-port "attach weights to a preloaded model" method across all ports; deferred.
    record PartialModel(String modelFileName, GGUF gguf, Tokenizer tokenizer) {}

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
                return new PartialModel(
                        path.getFileName().toString(), gguf, Tokenizers.fromGGUF(gguf));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The preloaded model when {@code modelPath} matches the baked one, else null (the caller falls
     * back to {@link Models#load(Path)}). Hands the baked tokenizer to {@link Tokenizers} first, so
     * the port's own {@code fromGGUF(gguf)} finds it by instance identity and only the tensor data
     * is read.
     */
    static LoadedModel<?> tryUsePreLoaded(Path modelPath) throws IOException {
        PartialModel preLoaded = PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        if (!Objects.equals(modelPath.getFileName().toString(), preLoaded.modelFileName())) {
            return null;
        }
        Tokenizers.preBaked(preLoaded.gguf(), preLoaded.tokenizer());
        try (var timer = Timer.log("Load tensors from pre-loaded model");
                FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            return Models.load(fileChannel, preLoaded.gguf(), java.lang.foreign.Arena.global());
        }
    }
}
