package com.llama4j;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Greedy-generation smoke/parity check for Gemma4: loads a GGUF, decodes a fixed prompt with
 *  argmax through the shared Engine loop, and prints the continuation for comparison against
 *  llama.cpp (./llama-cli ... --temp 0 --top-k 1 -no-cnv). Run:
 *    java ... com.llama4j.GemmaSmokeTest /path/to/gemma-4-E2B-it.gguf ["prompt"] [nTokens] */
public final class GemmaSmokeTest {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: GemmaSmokeTest <model.gguf> [prompt] [nTokens]");
            System.exit(2);
        }
        String prompt = args.length > 1 ? args[1] : "The capital of France is";
        int nTokens = args.length > 2 ? Integer.parseInt(args[2]) : 20;

        Gemma4 model = Gemma4.loadModel(Path.of(args[0]), 4096);
        System.err.printf("config: dim=%d layers=%d heads=%d headFull=%d headSwa=%d window=%d kvFromStart=%d plDim=%d moe=%b vocab=%d%n",
                model.configuration().embeddingLength, model.configuration().numberOfLayers,
                model.configuration().numberOfHeads, model.configuration().headSizeFull,
                model.configuration().headSizeSWA, model.configuration().slidingWindow,
                model.configuration().nLayerKvFromStart, model.configuration().embeddingLengthPerLayer,
                model.configuration().isMoE(), model.configuration().vocabularySize);

        boolean chat = System.getenv("CHAT") != null;
        List<Integer> promptTokens;
        if (chat) {
            // Gemma chat format: <bos><|turn>user\n{msg}<turn|>\n<|turn>model\n
            // THINK=1 mirrors llama.cpp's enable_thinking: a leading system turn with <|think|>.
            String think = System.getenv("THINK") != null ? "<|turn>system\n<|think|><turn|>\n" : "";
            String templated = "<bos>" + think + "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n";
            promptTokens = new ArrayList<>(model.tokenizer().encodeWithSpecialTokens(templated));
        } else {
            int bos = model.tokenizer().getSpecialTokens().getOrDefault("<bos>", 2);
            promptTokens = new ArrayList<>();
            promptTokens.add(bos);
            promptTokens.addAll(model.tokenizer().encode(prompt));
        }
        System.err.println("prompt tokens: " + promptTokens);

        InferenceState state = model.createNewState();
        long t0 = System.nanoTime();
        List<Integer> out = Engine.decodeLoop(model, state, 0, promptTokens, model.stopTokens(),
                promptTokens.size() + nTokens, Sampler.ARGMAX, null, GenerationHooks.NONE);
        double secs = (System.nanoTime() - t0) / 1e9;

        System.err.println("generated tokens: " + out);
        System.out.println("=== continuation ===");
        System.out.println(prompt + model.tokenizer().decode(out));
        System.err.printf("%n%.2f tok/s (%d tokens)%n", out.size() / secs, out.size());
    }
}
