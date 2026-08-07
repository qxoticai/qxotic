package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Timer;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * The native image's model preload ({@code -Djinfer.PreloadGGUF} at BUILD time): everything
 * derivable from the file's header - the parsed GGUF and the tokenizer built from it - is
 * constructed at class-init and, since {@code com.qxotic.*} initializes at image build, snapshots
 * into the image heap as one {@link PreloadedModel} record. Both pieces are pure heap; the tensor
 * data is still mmap'd at run time. What is NOT preloaded is the model object itself (config +
 * weight wiring): that would need a per-port "attach weights to a preloaded model" method;
 * deferred.
 */
final class AOT {

    /** Matched at run time by FILENAME - the one identity a path carries. */
    record PreloadedModel(String fileName, GGUF gguf, Tokenizer tokenizer) {}

    private static final PreloadedModel PRELOADED =
            preload(System.getProperty("jinfer.PreloadGGUF"));

    private static PreloadedModel preload(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        Path path = Path.of(modelPath);
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("cannot pre-load model: " + path);
        }
        try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fileChannel, path.toString());
            return new PreloadedModel(
                    path.getFileName().toString(), gguf, Tokenizers.fromGGUF(gguf));
        } catch (IOException e) {
            throw new java.io.UncheckedIOException("cannot pre-load model: " + path, e);
        }
    }

    /**
     * The preloaded model when {@code modelPath} names it, else null (the caller falls back to
     * {@link Models#load(Path)}). The preloaded header and tokenizer are plain arguments, so only
     * the tensor data is read - unless a runtime {@code -Djinfer.preTokenizer.*} override is
     * present, in which case the tokenizer rebuilds so the override applies: the escape hatch
     * outranks the preload, and that precedence is this caller's policy, not the library's.
     */
    static LoadedModel<?> tryUsePreloaded(Path modelPath) throws IOException {
        PreloadedModel preloaded = PRELOADED;
        if (preloaded == null
                || !Objects.equals(modelPath.getFileName().toString(), preloaded.fileName())) {
            return null;
        }
        Tokenizer tokenizer = Tokenizers.hasPropertyOverrides() ? null : preloaded.tokenizer();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
                FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            return Models.load(fileChannel, preloaded.gguf(), Arena.global(), tokenizer);
        }
    }
}
