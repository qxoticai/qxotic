// Prefill correctness for the Qwen35 state surgery (advance/resumeAt via BaseState, LAST-row
// logits): (1) repeated batched prefills agree within the FP-parallel floor (the same-path
// determinism PrefillDeterminism gates), (2) batched prefill and token-by-token step prefill
// agree on the next token (argmax) - the two paths run different kernels (GEMM vs matvec), so
// their logits differ by kernel reassociation (~3e-4 pure-Java, larger under jam tiling), which
// is reported informationally.
//   java ... com.qxotic.llm.Qwen35PrefillCheck [model.gguf]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public final class Qwen35PrefillCheck {

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args.length > 0 ? args[0] : "/home/mukel/Desktop/playground/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf");
        if (!Files.exists(path)) {
            System.out.println("Qwen35PrefillCheck: model not found (" + path + "), skipping");
            return;
        }
        Qwen35 model = Qwen35.loadModel(path, 4096);
        int vocab = model.config().vocabularySize();
        List<Integer> prompt = model.tokenizer().encode(
                "The expedition logged river depth, canopy density and soil acidity at every station; "
                + "readings were nominal and the weather held clear. Summarize the day in one sentence, "
                + "then estimate how many stations a four-person team could cover before dusk.");
        int[] ids = prompt.stream().mapToInt(Integer::intValue).toArray();
        int failures = 0;

        // (1) same-path determinism: repeated batched prefills within the FP-parallel floor
        float[] ref = null;
        double repeatMax = 0;
        for (int r = 0; r < 4; r++) {
            Qwen35.State s = model.newState(4096, 512);
            model.ingest(s, Batch.prefill(ids));
            float[] snap = snapshot(model.logits(s), vocab);
            if (ref == null) {
                ref = snap;
                continue;
            }
            for (int i = 0; i < vocab; i++) repeatMax = Math.max(repeatMax, Math.abs((double) snap[i] - ref[i]));
        }
        boolean deterministic = repeatMax <= 1e-2;
        System.out.printf("%s repeated batched prefills agree (max |d| = %.3g)%n", deterministic ? "ok:  " : "FAIL:", repeatMax);
        if (!deterministic) failures++;

        // (2) batched vs step prefill: same next token; kernel reassociation reported
        Qwen35.State stepped = model.newState(4096, 512);
        for (int id : ids) model.ingest(stepped, Batch.step(id));
        float[] step = snapshot(model.logits(stepped), vocab);
        int argBatched = argmax(ref), argStepped = argmax(step);
        double crossMax = 0;
        for (int i = 0; i < vocab; i++) crossMax = Math.max(crossMax, Math.abs((double) ref[i] - step[i]));
        boolean sameNext = argBatched == argStepped;
        System.out.printf("%s batched and step prefill agree on the next token (%d vs %d; cross-path max |d| = %.3g)%n",
                sameNext ? "ok:  " : "FAIL:", argBatched, argStepped, crossMax);
        if (!sameNext) failures++;

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("Qwen35PrefillCheck: all checks passed");
    }

    static float[] snapshot(FloatTensor logits, int n) {
        float[] out = new float[n];
        for (int i = 0; i < n; i++) out[i] = logits.getFloat(i);
        return out;
    }

    static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }
}
