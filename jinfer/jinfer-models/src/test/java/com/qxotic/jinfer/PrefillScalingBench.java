package com.qxotic.jinfer;

import java.nio.file.Path;

/**
 * Measures prefill throughput as the CONTEXT FILLS UP: ingest fixed-size chunks at increasing
 * start positions, timing each chunk. Chunk k attends over k*chunk prior keys, so this exposes
 * the O(n) per-token growth of attention and any super-linear pathology.
 *
 *   java ... com.qxotic.jinfer.PrefillScalingBench <model.gguf> [chunk=256] [nChunks=32] [warmupChunks=2]
 */
public final class PrefillScalingBench {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[0]);
        int chunk = args.length > 1 ? Integer.parseInt(args[1]) : 256;
        int nChunks = args.length > 2 ? Integer.parseInt(args[2]) : 32;
        int warmupChunks = args.length > 3 ? Integer.parseInt(args[3]) : 2;
        int ctx = chunk * (nChunks + 1) + 64;

        Model model = ModelLoader.loadModel(path, Math.max(8192, ctx));
        if (model instanceof Llama llama) { // LFM-only layer breakdown; other archs just print basics
            Llama.Configuration c = llama.configuration();
            int nConv = 0, nAttn = 0;
            for (int l = 0; l < c.numberOfLayers; l++) {
                if (c.isRecurrentLayer(l)) nConv++; else nAttn++;
            }
            System.out.printf("model: layers=%d conv=%d attn=%d dim=%d heads=%d ctx=%d chunk=%d%n",
                    c.numberOfLayers, nConv, nAttn, c.embeddingLength, c.numberOfHeads, c.contextLength, chunk);
        } else {
            System.out.printf("model: %s ctx=%d chunk=%d%n",
                    model.getClass().getSimpleName(), model.contextLength(), chunk);
        }

        // Build a long token stream.
        String para = "The history of the Roman empire is long and complex, spanning many centuries of "
                + "conquest, civil strife, cultural achievement and eventual decline across three continents. ";
        StringBuilder sb = new StringBuilder();
        int need = chunk * (nChunks + warmupChunks) + 16;
        // Encode the prose block ONCE and tile it (re-encoding a growing string would be O(n^2) and
        // unusable at long context). Token values are arbitrary but valid — this is a speed bench.
        for (int i = 0; i < 8; i++) sb.append(para);
        int[] block = model.tokenizer().encode(sb.toString()).stream().mapToInt(Integer::intValue).toArray();
        int[] stream = new int[need];
        for (int i = 0; i < need; i++) stream[i] = block[i % block.length];

        // Warmup at position 0 (JIT compile the kernels) on a throwaway state.
        InferenceState warm = model.createNewState();
        for (int i = 0; i < warmupChunks; i++) model.ingest(warm, stream, 0, 0, chunk);

        // Fresh state: ingest sequential chunks, timing each.
        InferenceState state = model.createNewState();
        System.out.printf("%-8s %-10s %-12s %-12s%n", "startPos", "ms", "tok/s", "attn_ms");
        for (int k = 0; k < nChunks; k++) {
            int startPos = k * chunk;
            Timing.reset();
            long t0 = System.nanoTime();
            model.ingest(state, stream, startPos, startPos, chunk);
            long t1 = System.nanoTime();
            double ms = (t1 - t0) / 1e6;
            System.out.printf("%-8d %-10.2f %-12.0f %-12.2f%n",
                    startPos, ms, chunk / (ms / 1000.0), Timing.attnMs());
        }
    }
}
