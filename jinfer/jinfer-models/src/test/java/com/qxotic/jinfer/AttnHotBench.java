package com.qxotic.jinfer;

import java.nio.file.Path;

/** Holds the context at a fixed high depth and re-ingests a chunk in a tight loop, so a sampling
 *  profiler (JFR) sees the attention-dominated steady state. */
public final class AttnHotBench {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[0]);
        int depth = args.length > 1 ? Integer.parseInt(args[1]) : 7000;
        int chunk = args.length > 2 ? Integer.parseInt(args[2]) : 256;
        int iters = args.length > 3 ? Integer.parseInt(args[3]) : 60;

        ModelLegacy model = ModelLoader.loadModel(path, depth + chunk + 256);
        String para = "The history of the Roman empire is long and complex, spanning many centuries of "
                + "conquest, civil strife, cultural achievement and eventual decline across three continents. ";
        // Encode the prose block once and tile it (re-encoding a growing string would be O(n^2)).
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 8; i++) sb.append(para);
        int[] block = model.tokenizer().encode(sb.toString()).stream().mapToInt(Integer::intValue).toArray();
        int need = depth + chunk + 16;
        int[] stream = new int[need];
        for (int i = 0; i < need; i++) stream[i] = block[i % block.length];

        InferenceState state = model.createNewState();
        for (int pos = 0; pos < depth; pos += chunk) {
            int c = Math.min(chunk, depth - pos);
            model.ingest(state, stream, pos, pos, c);
        }
        // Steady state: re-ingest a chunk at `depth` (attends over `depth` cached keys).
        for (int i = 0; i < 5; i++) model.ingest(state, stream, depth, depth, chunk);
        Timing.reset();
        long t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) model.ingest(state, stream, depth, depth, chunk);
        long t1 = System.nanoTime();
        double ms = (t1 - t0) / 1e6 / iters;
        System.out.printf("depth=%d chunk=%d  %.2f ms/chunk  attn=%.2f ms/chunk  (%.0f tok/s)%n",
                depth, chunk, ms, Timing.attnMs() / iters, chunk / (ms / 1000.0));
    }
}
