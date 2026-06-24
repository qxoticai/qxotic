package com.qxotic.jinfer;

import java.nio.file.Path;

/**
 * Decode (single-token generation) throughput as the context fills up: prefill to each target depth,
 * then time single-token steps there. Each step runs on the decode pool, exactly like production.
 *
 *   java ... com.qxotic.jinfer.DecodeScalingBench <model.gguf> [warmup=16] [measure=64] [depths...]
 */
public final class DecodeScalingBench {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[0]);
        int warmup = args.length > 1 ? Integer.parseInt(args[1]) : 16;
        int measure = args.length > 2 ? Integer.parseInt(args[2]) : 64;
        int[] depths = args.length > 3
                ? java.util.Arrays.stream(args, 3, args.length).mapToInt(Integer::parseInt).toArray()
                : new int[]{0, 2048, 4096, 8000};
        int maxDepth = 0; for (int d : depths) maxDepth = Math.max(maxDepth, d);
        int ctx = maxDepth + warmup + measure + 256;

        Model model = ModelLoader.loadModel(path, Math.max(8192, ctx));
        int vocab = model.vocabularySize();

        // filler token stream for prefill
        String para = "The history of the Roman empire is long and complex, spanning many centuries of "
                + "conquest, civil strife, cultural achievement and decline across three continents. ";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 8; i++) sb.append(para);
        int[] block = model.tokenizer().encode(sb.toString()).stream().mapToInt(Integer::intValue).toArray();
        int[] stream = new int[Math.max(1, maxDepth + 16)];
        for (int i = 0; i < stream.length; i++) stream[i] = block[i % block.length];

        // Global warmup: compile the decode path before ANY measured depth (otherwise the first
        // depth pays the JIT cost and reads artificially slow).
        {
            InferenceState w = model.createNewState();
            model.ingest(w, stream, 0, 0, Math.min(256, stream.length));
            int[] one = new int[1]; int[] wp = {Math.min(256, stream.length)};
            int t = Parallel.onDecodePool(() -> model.computeLogits(w)).argmax(0, vocab);
            for (int i = 0; i < 80; i++) {
                int prev = t;
                t = Parallel.onDecodePool(() -> { one[0] = prev; model.ingest(w, one, 0, wp[0]++, 1); return model.computeLogits(w).argmax(0, vocab); });
            }
        }

        System.out.printf("%-8s %-12s%n", "depth", "decode_tok/s");
        for (int depth : depths) {
            InferenceState state = model.createNewState();
            // prefill to `depth` in chunks
            int pos = 0;
            while (pos < depth) {
                int c = Math.min(1024, depth - pos);
                model.ingest(state, stream, pos, pos, c);
                pos += c;
            }
            int[] one = new int[1];
            int[] p = {Math.max(pos, 0)};
            int tok = depth > 0 ? Parallel.onDecodePool(() -> model.computeLogits(state)).argmax(0, vocab) : 1;
            java.util.function.IntUnaryOperator step = prev -> Parallel.onDecodePool(() -> {
                one[0] = prev; model.ingest(state, one, 0, p[0]++, 1);
                return model.computeLogits(state).argmax(0, vocab);
            });
            for (int i = 0; i < warmup; i++) tok = step.applyAsInt(tok);
            long t0 = System.nanoTime();
            for (int i = 0; i < measure; i++) tok = step.applyAsInt(tok);
            long t1 = System.nanoTime();
            double secs = (t1 - t0) / 1e9;
            System.out.printf("%-8d %-12.2f%n", depth, measure / secs);
        }
    }
}
