package com.llama4j;

import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.List;

/**
 * Measures decode-phase throughput (tok/s) and heap-allocation rate, isolated from model load and
 * prefill, after JIT warmup. Allocation is read exactly from {@code ThreadMXBean.getThreadAllocatedBytes}
 * summed over all live threads (main + ForkJoin workers).
 *
 *   java ... com.llama4j.DecodeAllocBench <model.gguf> [warmup=64] [measure=256]
 */
public final class DecodeAllocBench {

    static long totalAllocated() {
        var tb = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        long s = 0;
        for (long id : tb.getAllThreadIds()) {
            long b = tb.getThreadAllocatedBytes(id);
            if (b > 0) s += b;
        }
        return s;
    }

    static int argmax(FloatTensor logits, int n) {
        int best = 0;
        float bestVal = logits.getFloat(0);
        for (int i = 1; i < n; i++) {
            float v = logits.getFloat(i);
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[0]);
        int warmup = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        int measure = args.length > 2 ? Integer.parseInt(args[2]) : 256;

        Model model = ModelLoader.loadModel(path, 4096);
        int vocab = model.vocabularySize();
        InferenceState state = model.createNewState();

        List<Integer> enc = model.tokenizer().encode("The history of the Roman empire is");
        int[] prompt = enc.stream().mapToInt(Integer::intValue).toArray();
        model.ingest(state, prompt, 0, 0, prompt.length);
        int pos = prompt.length;
        int tok = argmax(model.computeLogits(state), vocab);

        int[] one = new int[1];
        // Mirror production: each decode step runs on Parallel.onDecodePool (the physical-core-width pool
        // Engine.decodeLoop uses), so this measures real decode throughput.
        final InferenceState fstate = state;
        final Model fmodel = model;
        final int[] fone = one;
        final int fvocab = vocab;
        final java.util.concurrent.atomic.AtomicInteger stepPos = new java.util.concurrent.atomic.AtomicInteger(pos);
        java.util.function.IntUnaryOperator run = prevTok -> Parallel.onDecodePool(() -> {
            int p = stepPos.getAndIncrement();
            fone[0] = prevTok; fmodel.ingest(fstate, fone, 0, p, 1);
            return argmax(fmodel.computeLogits(fstate), fvocab);
        });
        // warmup: let the JIT compile the decode path before measuring.
        for (int i = 0; i < warmup; i++) tok = run.applyAsInt(tok);

        jdk.jfr.Recording rec = new jdk.jfr.Recording();
        rec.enable("jdk.ObjectAllocationSample").with("throttle", "100000/s");
        rec.enable("jdk.ObjectAllocationInNewTLAB");   // carries the actual object size
        rec.start();
        long a0 = totalAllocated();
        long t0 = System.nanoTime();
        for (int i = 0; i < measure; i++) tok = run.applyAsInt(tok);
        long t1 = System.nanoTime();
        long a1 = totalAllocated();
        rec.stop();
        rec.dump(Path.of("/tmp/decode-alloc.jfr"));

        double secs = (t1 - t0) / 1e9;
        long bytes = a1 - a0;
        System.out.printf("%-34s decode %.1f tok/s | alloc %.2f MB/s | %.1f KB/token (%d tokens, %.2fs)%n",
                path.getFileName().toString(), measure / secs, bytes / 1048576.0 / secs,
                bytes / 1024.0 / measure, measure, secs);
    }
}
