// Shared model-path resolution for the example scripts, matching scripts/download-models.sh and
// ModelFixture so a tree populated by the download script just works:
//     -Djinfer.models=/path > $JINFER_MODELS > ../models beside the checkout > ~/models
// or pass an explicit .gguf path as the script's trailing argument.
import java.nio.file.Files;
import java.nio.file.Path;

public final class Models {
    static final Path ROOT = resolveRoot();

    private static Path resolveRoot() {
        String prop = System.getProperty("jinfer.models");
        if (prop != null && !prop.isBlank()) return Path.of(prop);
        String env = System.getenv("JINFER_MODELS");
        if (env != null && !env.isBlank()) return Path.of(env);
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent()) {
            if (Files.exists(dir.resolve(".git"))) { // a FILE in worktrees, a directory otherwise
                return dir.getParent() == null ? dir.resolve("models")
                                               : dir.getParent().resolve("models");
            }
        }
        return Path.of(System.getProperty("user.home"), "models");
    }

    static Path chat(String[] args, int at)     { return pick(args, at, "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf"); }
    static Path embed(String[] args, int at)    { return pick(args, at, "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"); }
    static Path rerank(String[] args, int at)   { return pick(args, at, "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf"); }
    static Path speech(String[] args, int at)   { return pick(args, at, "hf.co/remixerdec/Inflect-Nano-v2-GGUF/inflect_nano_v2_q8_0.gguf"); }

    // Detection needs a bigger model than description does - see the note in Detect.java.
    static Path gemmaDetect(String[] a, int at)       { return pick(a, at, "hf.co/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf"); }
    static Path gemmaDetectMmproj(String[] a, int at) { return pick(a, at, "hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf"); }

    private static Path pick(String[] args, int at, String relative) {
        Path p = args.length > at ? Path.of(args[at]) : ROOT.resolve(relative);
        if (!Files.isReadable(p))
            throw new IllegalStateException("model not found: " + p
                    + "\nSet JINFER_MODELS or pass the .gguf path as an argument.");
        return p;
    }
}
