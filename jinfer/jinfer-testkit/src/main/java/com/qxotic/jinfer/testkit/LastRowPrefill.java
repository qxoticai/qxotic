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
 * caches later steps read, must still match a pass that keeps every row.
 */
public final class LastRowPrefill {

    private static final int STEPS = 8;

    /** A prompt any length can be cut from: counting, whose continuation no tie can flip. */
    public static final String COUNTING =
            String.join(", ", IntStream.rangeClosed(1, 400).mapToObj(Integer::toString).toList());

    private LastRowPrefill() {}

    /**
     * {@code ids} whole, then as two batches: the second starts mid-context and reads what a
     * skipping first batch wrote.
     */
    public static <S extends ContextState> void assertMatchesFullPass(
            LanguageModel<?, ?, S> model, int[] ids, double tolerance) {
        for (int at : new int[] {ids.length, ids.length / 2})
            assertMatchesFullPass(model, tokens(ids, at), ids.length, tolerance);
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
        List<float[]> last = run(model, prompt.apply(Batch.Outputs.LAST), rows);
        double drift = 0;
        for (int step = 0; step <= STEPS; step++)
            for (int i = 0; i < full.get(step).length; i++)
                drift = Math.max(drift, Math.abs(full.get(step)[i] - last.get(step)[i]));
        assertTrue(drift <= tolerance, rows + " rows: max |logit drift| " + drift);
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
