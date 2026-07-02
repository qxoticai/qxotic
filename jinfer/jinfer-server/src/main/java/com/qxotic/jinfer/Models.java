// Loads any new-API generative model, dispatching on general.architecture to the matching port
// (all implement LanguageModel). The single place that knows the full arch -> port mapping for the
// server; mirrors the dispatch in JinferBench.loadAny.
package com.qxotic.jinfer;

import com.qxotic.llm.Gemma4;
import com.qxotic.llm.Granite;
import com.qxotic.llm.GptOss;
import com.qxotic.llm.Lfm2;
import com.qxotic.llm.Llama;
import com.qxotic.llm.NemotronH;
import com.qxotic.llm.Qwen35;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public final class Models {

    private Models() {
    }

    /** Loads {@code path} at context size {@code ctx} (-1 = the model's full context), dispatching on
     *  {@code general.architecture} to the matching new-API port. */
    public static LanguageModel<?, ?, ?> load(Path path, int ctx) throws IOException {
        String arch;
        try (FileChannel fc = FileChannel.open(path, StandardOpenOption.READ)) {
            arch = ModelLoader.readGguf(fc, path.toString()).getString("general.architecture");
        }
        return switch (arch) {
            case "gemma4" -> Gemma4.loadModel(path, ctx);
            case "gpt-oss" -> GptOss.loadModel(path, ctx);
            case "qwen35", "qwen35moe" -> Qwen35.loadModel(path, ctx);
            case "nemotron_h", "nemotron_h_moe" -> NemotronH.loadModel(path, ctx);
            case "llama", "minicpm", "mistral3", "smollm3" -> Llama.loadModel(path, ctx);   // same-graph Llama variants
            case "granite" -> Granite.loadModel(path, ctx);
            default -> {
                if (arch.startsWith("lfm")) yield Lfm2.loadModel(path, ctx);
                throw new IllegalArgumentException("Models.load: unsupported architecture '" + arch + "'");
            }
        };
    }
}
