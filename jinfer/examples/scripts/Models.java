// Shared model-path resolution for the example scripts. Override the root with
//     export JINFER_MODELS=/path/to/models        (default: ~/models)
// or pass an explicit .gguf path as the script's trailing argument.
import java.nio.file.Files;
import java.nio.file.Path;

public final class Models {
    static final Path ROOT = Path.of(System.getenv().getOrDefault(
            "JINFER_MODELS", System.getProperty("user.home") + "/models"));

    static Path chat(String[] args, int at)     { return pick(args, at, "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"); }
    static Path embed(String[] args, int at)    { return pick(args, at, "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"); }
    static Path rerank(String[] args, int at)   { return pick(args, at, "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf"); }
    static Path speech(String[] args, int at)   { return pick(args, at, "hf.co/remixerdec/Inflect-Nano-v2-GGUF/inflect_nano_v2_q8_0.gguf"); }

    static Path gemmaVision(String[] a, int at) { return pick(a, at, "hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"); }
    static Path gemmaMmproj(String[] a, int at) { return pick(a, at, "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf"); }
    static Path gemmaMtp(String[] a, int at)    { return pick(a, at, "hf.co/unsloth/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf"); }

    private static Path pick(String[] args, int at, String relative) {
        Path p = args.length > at ? Path.of(args[at]) : ROOT.resolve(relative);
        if (!Files.isReadable(p))
            throw new IllegalStateException("model not found: " + p
                    + "\nSet JINFER_MODELS or pass the .gguf path as an argument.");
        return p;
    }
}
