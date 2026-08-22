package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.Shape;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * {@link Batch#prepare} behavior tests PLUS the guard-rail for its copy-saving rewrite: a verbatim
 * copy of the OLD concat-then-slice algorithm ({@code referencePrepare}) as oracle, and a
 * randomized sweep asserting the improved implementation emits the identical chunk sequence for
 * every input shape. If this ever goes red, the optimization is wrong by definition.
 */
class BatchTest {

    @Test
    void mergesAdjacentTokenBatches() {
        List<Batch> out =
                Batch.prepare(List.of(Batch.prefill(ids(100)), Batch.prefill(ids(200))), 512);
        assertEquals(1, out.size());
        assertEquals(300, out.get(0).count());
    }

    @Test
    void splitsOversizedAtCapacity() {
        List<Batch> out = Batch.prepare(List.of(Batch.prefill(ids(1000))), 512);
        assertEquals(2, out.size());
        assertEquals(512, out.get(0).count());
        assertEquals(488, out.get(1).count());
    }

    @Test
    void passesNonTokenBatchesThroughUnsplit() {
        Batch seqs = Batch.pack(new int[][] {ids(40), ids(60)});
        Batch score = Batch.score(ids(700)); // Tokens but Outputs.ALL: never fused
        List<Batch> out = Batch.prepare(List.of(Batch.prefill(ids(10)), seqs, score), 512);
        assertEquals(3, out.size());
        assertSame(seqs, out.get(1));
        assertSame(score, out.get(2));
    }

    @Test
    void dropsEmptyTokenBatches() {
        assertEquals(List.of(), Batch.prepare(List.of(Batch.prefill(new int[0])), 512));
    }

    @Test
    void legalSingleBatchPassesThroughUncopied() {
        // documented by test: prepare ALIASES a legal single batch's array
        int[] prompt = ids(512);
        List<Batch> out = Batch.prepare(List.of(Batch.prefill(prompt)), 512);
        assertEquals(1, out.size());
        assertSame(prompt, ((Batch.Input.Tokens) out.get(0).input()).ids());
    }

    @Test
    void rejectsNonPositiveCapacity() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Batch.prepare(List.of(Batch.prefill(ids(1))), 0));
    }

    @Test
    void rejectsMissingStructureAtConstruction() {
        assertThrows(NullPointerException.class, () -> new Batch(null, Batch.Outputs.LAST));
        assertThrows(
                NullPointerException.class,
                () -> new Batch(new Batch.Input.Tokens(new int[] {1}), null));
        assertThrows(NullPointerException.class, () -> new Batch.Input.Tokens(null));
        assertThrows(
                NullPointerException.class,
                () -> new Batch.Input.Sequences(new Batch.Input.Tokens(new int[] {1}), null));
    }

    @Test
    void embeddingBlocksAreAtomic() {
        try (Arena arena = Arena.ofConfined()) {
            var rows = Views.allocateF32(new PanamaMemoryArena(arena), 24).view(Shape.flat(6, 4));
            Batch bidirectional = Batch.embeddings(rows, 6);
            assertSame(bidirectional, Batch.prepare(List.of(bidirectional), 6).get(0));
            assertThrows(
                    IllegalArgumentException.class, () -> Batch.prepare(List.of(bidirectional), 5));

            Batch causal = Batch.embeddings(rows, 6, false);
            assertSame(causal, Batch.prepare(List.of(causal), 5).get(0));
            assertEquals(6, causal.count());
        }
    }

    @Test
    void propertyMatchesReferenceAlgorithm() {
        Random rng = new Random(42);
        int[] capacities = {1, 2, 3, 7, 64, 512, 1000};
        for (int iter = 0; iter < 500; iter++) {
            int cap = capacities[rng.nextInt(capacities.length)];
            List<Batch> batches = new ArrayList<>();
            for (int b = 0, nb = rng.nextInt(6); b < nb; b++) {
                switch (rng.nextInt(4)) {
                    case 0 -> batches.add(Batch.prefill(ids(rng.nextInt(1500))));
                    case 1 -> batches.add(Batch.prefill(ids(rng.nextInt(3)))); // empty/tiny runs
                    case 2 -> batches.add(Batch.score(ids(rng.nextInt(700))));
                    default ->
                            batches.add(
                                    Batch.pack(
                                            new int[][] {
                                                ids(1 + rng.nextInt(40)), ids(1 + rng.nextInt(40))
                                            }));
                }
            }
            assertSameBatches(
                    referencePrepare(batches, cap),
                    Batch.prepare(batches, cap),
                    "cap=" + cap + " batches=" + batches);
        }
    }

    // ------------------------------------------------------------------
    // The oracle: the OLD algorithm, verbatim (concat the run, then slice), over Batch.
    // ------------------------------------------------------------------

    private static List<Batch> referencePrepare(List<Batch> batches, int batchCapacity) {
        var out = new ArrayList<Batch>(batches.size());
        var run = new ArrayList<int[]>();
        for (Batch b : batches) {
            if (b.input() instanceof Batch.Input.Tokens t && b.outputs() == Batch.Outputs.LAST) {
                run.add(t.ids());
                continue;
            }
            referenceFlush(run, batchCapacity, out);
            out.add(b);
        }
        referenceFlush(run, batchCapacity, out);
        return out;
    }

    private static void referenceFlush(List<int[]> run, int batchCapacity, List<Batch> out) {
        if (run.isEmpty()) return;
        int total = 0;
        for (int[] part : run) total += part.length;
        int[] ids = new int[total];
        int off = 0;
        for (int[] part : run) {
            System.arraycopy(part, 0, ids, off, part.length);
            off += part.length;
        }
        run.clear();
        for (int from = 0; from < ids.length; from += batchCapacity) {
            out.add(
                    Batch.prefill(
                            Arrays.copyOfRange(
                                    ids, from, Math.min(from + batchCapacity, ids.length))));
        }
    }

    private static void assertSameBatches(List<Batch> expected, List<Batch> actual, String what) {
        assertEquals(expected.size(), actual.size(), what + ": chunk count");
        for (int i = 0; i < expected.size(); i++) {
            Batch e = expected.get(i), a = actual.get(i);
            assertEquals(e.outputs(), a.outputs(), what + ": outputs at " + i);
            assertEquals(e.input().getClass(), a.input().getClass(), what + ": input kind at " + i);
            switch (e.input()) {
                case Batch.Input.Tokens t ->
                        assertArrayEquals(
                                t.ids(),
                                ((Batch.Input.Tokens) a.input()).ids(),
                                what + ": ids at " + i);
                case Batch.Input.Sequences s -> {
                    assertArrayEquals(
                            s.tokens().ids(),
                            ((Batch.Input.Sequences) a.input()).tokens().ids(),
                            what + ": packed ids at " + i);
                    assertArrayEquals(
                            s.seqLen(),
                            ((Batch.Input.Sequences) a.input()).seqLen(),
                            what + ": seqLen at " + i);
                }
                case Batch.Input.Embeddings ignored ->
                        throw new AssertionError("reference inputs never contain embeddings");
            }
        }
    }

    private static int[] ids(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = i * 17 + 1;
        return out;
    }
}
