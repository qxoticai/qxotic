package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Views;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A prefill that reads only its last row may skip the work no other row needs: its logits, and the
 * caches and histories later steps read, must still match a pass that keeps every row.
 */
public final class LastRowPrefill {

    private static final int STEPS = 8;

    /** A prompt any length can be cut from: counting, whose continuation no tie can flip. */
    public static final String COUNTING =
            String.join(", ", IntStream.rangeClosed(1, 400).mapToObj(Integer::toString).toList());

    private LastRowPrefill() {}

    /**
     * {@code ids} whole, then as two batches: the second starts mid-context and reads what a
     * skipping first batch wrote. The bound is {@code tolerance} or, if larger, how far splitting
     * the prompt moves the full pass (its second batch reads the first's keys from the F16 cache,
     * which jam-native's int8 activations turn into ~0.1-0.3): a tail runs as a chunk, so it may
     * drift as much as a split does.
     */
    public static <S extends ContextState> void assertMatchesFullPass(
            LanguageModel<?, ?, S> model, int[] ids, double tolerance) {
        int n = ids.length, half = n / 2;
        run(model, tokens(ids, half).apply(Batch.Outputs.ALL), n); // warm, as below
        double split =
                maxDrift(
                        run(model, tokens(ids, n).apply(Batch.Outputs.ALL), n),
                        run(model, tokens(ids, half).apply(Batch.Outputs.ALL), n));
        for (int at : new int[] {n, half})
            assertMatchesFullPass(model, tokens(ids, at), n, Math.max(tolerance, split));
    }

    /**
     * The logits after {@code prompt} and after each of {@value #STEPS} forced decode steps, its
     * batches built with {@link Batch.Outputs#LAST} against {@link Batch.Outputs#ALL}: no logit may
     * move more than {@code tolerance}.
     */
    public static <S extends ContextState> void assertMatchesFullPass(
            LanguageModel<?, ?, S> model,
            Function<Batch.Outputs, List<Batch>> prompt,
            int rows,
            double tolerance) {
        // a cold JIT tier alone moves logits by up to 0.3: compare warm passes
        run(model, prompt.apply(Batch.Outputs.ALL), rows);
        run(model, prompt.apply(Batch.Outputs.LAST), rows);
        List<float[]> full = run(model, prompt.apply(Batch.Outputs.ALL), rows);
        double drift = maxDrift(full, run(model, prompt.apply(Batch.Outputs.LAST), rows));
        assertTrue(drift <= tolerance, rows + " rows: max |logit drift| " + drift);
    }

    /** The largest logit difference at any step. */
    private static double maxDrift(List<float[]> a, List<float[]> b) {
        double drift = 0;
        for (int step = 0; step <= STEPS; step++)
            for (int i = 0; i < a.get(step).length; i++)
                drift = Math.max(drift, Math.abs(a.get(step)[i] - b.get(step)[i]));
        return drift;
    }

    private static Function<Batch.Outputs, List<Batch>> tokens(int[] ids, int at) {
        return outputs ->
                Stream.of(Arrays.copyOfRange(ids, 0, at), Arrays.copyOfRange(ids, at, ids.length))
                        .filter(part -> part.length > 0)
                        .map(part -> new Batch(new Batch.Input.Tokens(part), outputs))
                        .toList();
    }

    private static <S extends ContextState> List<float[]> run(
            LanguageModel<?, ?, S> model, List<Batch> prompt, int rows) {
        int vocab = model.configuration().vocabularySize();
        try (S state = model.newState(rows + STEPS, rows)) {
            for (Batch batch : prompt) model.ingest(state, batch);
            List<float[]> logits = new ArrayList<>();
            for (int step = 0; ; step++) {
                var row = Views.castToSegmentBacked(model.logits(state), "logits");
                float[] copy = new float[vocab];
                for (int i = 0; i < vocab; i++) copy[i] = Views.getFloat(row, i, "logits");
                logits.add(copy);
                if (step == STEPS) return logits;
                model.ingest(state, Batch.step(100 + step)); // any fixed tokens will do
            }
        }
    }
}
