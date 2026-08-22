// Shared remote-ref defaults for the examples. Pass a trailing model ref to override one without
// changing the script; local files use the builders' modelPath/companionPath methods instead.

public final class Models {
    private Models() {}

    static String chat(String[] args, int at) {
        return pick(args, at, "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0");
    }

    static String embed(String[] args, int at) {
        return pick(args, at, "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0");
    }

    static String rerank(String[] args, int at) {
        return pick(args, at, "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0");
    }

    static String speech(String[] args, int at) {
        return pick(args, at, "hf.co/remixerdec/Inflect-Nano-v2-GGUF:Q8_0");
    }

    static String gemma(String[] args, int at) {
        return pick(args, at, "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0");
    }

    static String gemmaMmproj(String[] args, int at) {
        return pick(args, at, "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
    }

    // Detection needs a bigger model than description does - see the note in Detect.java.
    static String gemmaDetect(String[] args, int at) {
        return pick(args, at, "hf.co/unsloth/gemma-4-12b-it-GGUF:Q8_0");
    }

    static String gemmaDetectMmproj(String[] args, int at) {
        return pick(args, at, "hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf");
    }

    private static String pick(String[] args, int at, String fallback) {
        return args.length > at ? args[at] : fallback;
    }
}
