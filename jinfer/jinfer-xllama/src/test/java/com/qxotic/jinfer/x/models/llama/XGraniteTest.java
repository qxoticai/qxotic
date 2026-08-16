package com.qxotic.jinfer.x.models.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * THE cycle-3 gate for the Granite port: the x Granite vs the old FloatTensor Granite on the REAL
 * granite-3.3-2b-instruct Q4_K_M checkpoint (Granite's four scalars - embedding / residual / logit
 * / attention scale - all live), ONE shared tokenizer instance fed to both loaders, greedy decoding
 * — strict token-id parity at every step plus a logits max-abs-diff check (both trees route
 * gemm/gemv to the same dot-based Java floor: the JAM backends are surefire-excluded). Two legs: a
 * short prompt (single-chunk prefill) and a long prompt at a forced small batchCapacity (chunked
 * prefill: the cross-chunk KV carry AND the lazy last-layer tail finishing over a chunk boundary).
 * Skipped when the checkpoint is not in the HF cache.
 */
class XGraniteTest {

    private static final Path HF_CACHE =
            Path.of(System.getProperty("user.home"), ".cache/huggingface/hub");
    private static final float LOGIT_TOLERANCE = 1e-2f;

    private static Path graniteModel;

    @BeforeAll
    static void findModels() throws IOException {
        graniteModel =
                findGguf("models--ibm-granite--granite-4.1-3b-GGUF", "granite-4.1-3b-Q8_0.gguf");
    }

    private static Path findGguf(String repoName, String fileName) throws IOException {
        Path repo = HF_CACHE.resolve(repoName).resolve("snapshots");
        if (!Files.isDirectory(repo)) return null;
        try (Stream<Path> snaps = Files.list(repo)) {
            return snaps.flatMap(
                            s -> {
                                try {
                                    return Files.list(s);
                                } catch (IOException e) {
                                    return Stream.empty();
                                }
                            })
                    .filter(p -> p.getFileName().toString().equals(fileName))
                    .findFirst()
                    .orElse(null);
        }
    }

    @Test
    void greedyParity() throws Exception {
        assumeTrue(graniteModel != null, "granite-4.1-3b-Q8_0.gguf not in the HF cache");
        // batchCapacity >= prompt: one prefill chunk
        assertGreedyParity(graniteModel, "The capital of France is", 64, 0);
    }

    @Test
    void greedyParityChunked() throws Exception {
        assumeTrue(graniteModel != null, "granite-4.1-3b-Q8_0.gguf not in the HF cache");
        // a ~48-token prompt at batchCapacity 16: the prefill spans three chunks, so decode runs
        // on a KV carry across chunk boundaries (and the lazy tail finishes over one)
        assertGreedyParity(
                graniteModel,
                "The history of the Eiffel Tower begins in 1884, when two engineers at Gustave"
                        + " Eiffel's company sketched a 300-metre iron lattice tower for the 1889"
                        + " Exposition Universelle in Paris. The tower was",
                48,
                16);
    }

    private static void assertGreedyParity(
            Path model, String promptText, int nTokens, int forcedBatchCapacity) throws Exception {
        try (FileChannel channel = FileChannel.open(model)) {
            GGUF gguf = ModelLoader.readGguf(channel, "granite");
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);

            List<Integer> prompt = new ArrayList<>();
            prompt.add(gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", 1));
            for (int id : tokenizer.encodeToArray(promptText)) prompt.add(id);
            int[] ids = prompt.stream().mapToInt(Integer::intValue).toArray();

            int[] oldIds;
            float[] oldFirst, oldLast;
            var om =
                    com.qxotic.jinfer.models.llama.Granite.loadModel(
                            channel, gguf, Arena.ofAuto(), tokenizer);
            var oc = om.config();
            int vocab = oc.vocabularySize();
            int bc = forcedBatchCapacity > 0 ? forcedBatchCapacity : Math.max(16, ids.length);
            try (var os =
                    om.newState(Math.min(oc.contextLength(), ids.length + nTokens + 16), bc)) {
                for (com.qxotic.jinfer.Batch chunk :
                        com.qxotic.jinfer.Batch.prepare(
                                java.util.List.of(com.qxotic.jinfer.Batch.prefill(ids)), bc)) {
                    om.ingest(os, chunk);
                }
                oldIds = new int[nTokens];
                FloatTensor logits = om.logits(os);
                oldFirst = snapshotOld(logits, vocab);
                int tok = logits.argmax();
                for (int n = 0; n < nTokens; n++) {
                    oldIds[n] = tok;
                    om.ingest(os, com.qxotic.jinfer.Batch.step(tok));
                    logits = om.logits(os);
                    tok = logits.argmax();
                }
                oldLast = snapshotOld(logits, vocab);
            }

            int[] xIds;
            float[] xFirst, xLast;
            var xm = Granite.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            var xc = xm.configuration();
            try (var xs =
                    xm.newState(Math.min(xc.contextLength(), ids.length + nTokens + 16), bc)) {
                for (Batch chunk : Batch.prepare(List.of(Batch.prefill(ids)), bc)) {
                    xm.ingest(xs, chunk);
                }
                xIds = new int[nTokens];
                MemoryView<MemorySegment> logits =
                        Views.castToSegmentBacked(xm.logits(xs), "logits");
                xFirst = snapshotX(logits, vocab);
                int tok = Ops.argmax(logits, 0, vocab);
                for (int n = 0; n < nTokens; n++) {
                    xIds[n] = tok;
                    xm.ingest(xs, Batch.step(tok));
                    logits = Views.castToSegmentBacked(xm.logits(xs), "logits");
                    tok = Ops.argmax(logits, 0, vocab);
                }
                xLast = snapshotX(logits, vocab);
            }

            float firstDiff = maxAbsDiff(oldFirst, xFirst);
            float lastDiff = maxAbsDiff(oldLast, xLast);
            System.err.printf(
                    "logits max-abs-diff: prefill=%.6g final=%.6g%n", firstDiff, lastDiff);
            assertTrue(firstDiff < LOGIT_TOLERANCE, "prefill logits diverged: " + firstDiff);
            assertTrue(lastDiff < LOGIT_TOLERANCE, "final logits diverged: " + lastDiff);
            assertArrayEquals(oldIds, xIds, "greedy token-id divergence");
        }
    }

    private static float[] snapshotOld(FloatTensor logits, int vocab) {
        float[] out = new float[vocab];
        for (int i = 0; i < vocab; i++) out[i] = logits.getFloat(i);
        return out;
    }

    private static float[] snapshotX(MemoryView<MemorySegment> logits, int vocab) {
        float[] out = new float[vocab];
        for (int i = 0; i < vocab; i++) out[i] = Views.getFloat(logits, i, "logits");
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float max = 0;
        for (int i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
        return max;
    }
}
