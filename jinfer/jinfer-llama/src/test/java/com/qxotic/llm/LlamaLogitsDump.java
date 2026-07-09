// Dumps logits at several rows of a scored (ALL) prompt, to compare the lazy last-layer split
// against
// the pre-refactor impl (argmax must match; checksum diff quantifies the FP delta). Also probes
// any-order
// idempotency (querying a row twice / out of order must be identical - the read-only-s.x
// discipline).
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import java.nio.file.Path;

public final class LlamaLogitsDump {
    public static void main(String[] args) throws Exception {
        Llama model = Llama.loadModel(Path.of(args[0]), 256);
        int vocab = model.config().vocabularySize();
        int n = 48;
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = (i * 37 + 11) % vocab; // fixed synthetic prompt

        var s = model.newState(64, 64);
        model.ingest(
                s, Batch.score(ids)); // ALL -> every row's logits addressable via logits(s, row)

        int[] rows = {0, 12, 24, 47};
        for (int r : rows) print(model, s, r, vocab);

        // idempotency / any-order: re-query row 24 and a reverse sweep; must match the first pass
        // exactly.
        System.out.println("-- reverse re-query (idempotency) --");
        for (int r : new int[] {47, 24, 12, 0, 24}) print(model, s, r, vocab);
    }

    static void print(Llama model, Llama.State s, int row, int vocab) {
        FloatTensor lg = model.logits(s, row);
        int am = 0;
        float mx = lg.getFloat(0);
        double sum = 0;
        for (int i = 0; i < vocab; i++) {
            float v = lg.getFloat(i);
            sum += Math.abs(v);
            if (v > mx) {
                mx = v;
                am = i;
            }
        }
        System.out.printf("row %2d  argmax=%d  max=%.6f  |L1|=%.4f%n", row, am, mx, sum);
    }
}
