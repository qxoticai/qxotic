// Loads any new-API generative model, dispatching on general.architecture to the matching port
// (all implement LanguageModel). The single place that knows the full arch -> port mapping for the
// server; mirrors the dispatch in JinferBench.loadAny.
package com.qxotic.jinfer.server;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.*;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.models.gemma4.Gemma4;
import com.qxotic.jinfer.models.gptoss.GptOss;
import com.qxotic.jinfer.models.lfm2.Lfm2;
import com.qxotic.jinfer.models.llama.Granite;
import com.qxotic.jinfer.models.llama.Llama;
import com.qxotic.jinfer.models.nemotronh.NemotronH;
import com.qxotic.jinfer.models.qwen35.Qwen35;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public final class Models {

    private Models() {}

    /**
     * Loads {@code path} at context size {@code ctx} (-1 = the model's full context), dispatching
     * on {@code general.architecture} to the matching new-API port.
     */
    public static LanguageModel<?, ?, ?> load(Path path, int ctx) throws IOException {
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fc, path.toString());
            return load(fc, gguf, ctx);
        }
    }

    /**
     * As {@link #load(Path, int)} but reusing an already-parsed {@code gguf} (the header is not
     * re-read) - used by AOT preload. {@code fileChannel} supplies the tensor data to mmap.
     */
    public static LanguageModel<?, ?, ?> load(FileChannel fileChannel, GGUF gguf, int ctx)
            throws IOException {
        String arch = gguf.getString("general.architecture");
        return switch (arch) {
            case "gemma4" -> Gemma4.loadModel(fileChannel, gguf, ctx, true);
            case "gpt-oss" -> GptOss.loadModel(fileChannel, gguf, ctx, true);
            case "qwen35", "qwen35moe" -> Qwen35.loadModel(fileChannel, gguf, ctx, true);
            case "nemotron_h", "nemotron_h_moe" -> NemotronH.loadModel(fileChannel, gguf, ctx);
            case "llama", "minicpm", "mistral3", "smollm3" ->
                    Llama.loadModel(fileChannel, gguf, ctx, true); // same-graph Llama variants
            case "granite" -> Granite.loadModel(fileChannel, gguf, ctx, true);
            default -> {
                if (arch.startsWith("lfm")) yield Lfm2.loadModel(fileChannel, gguf, ctx, true);
                throw new IllegalArgumentException(
                        "Models.load: unsupported architecture '" + arch + "'");
            }
        };
    }
}
