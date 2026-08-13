package com.qxotic.jinfer.x.models.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.testkit.ModelFixture;
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
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * THE cycle-3 gate for the Llama port: the x Llama vs the old FloatTensor Llama on REAL
 * checkpoints, ONE shared tokenizer instance fed to both loaders, greedy decoding — strict token-id
 * parity at every step plus a logits max-abs-diff check (both trees route gemm/gemv to the same
 * dot-based Java floor: the JAM backends are surefire-excluded). Legs, one per metadata variant of
 * the graph:
 *
 * <ul>
 *   <li>Llama-3.2-1B-Instruct Q4_K_M — plain Llama 3 (llama3 rope_freqs), the Q4_K/Q6_K recipe the
 *       k-quant dtypes and MatMul arms landed for; twice: single-chunk prefill and a long prompt at
 *       batchCapacity 16 (chunked prefill: cross-chunk KV carry + the lazy last-layer tail
 *       finishing over a chunk boundary).
 *   <li>SmolLM3-3B Q4_K_M — NoPE: RoPE skipped on every 4th layer (noRopeLayerStep).
 *   <li>Ministral-3-3B-Instruct-2512 Q4_K_M — mistral3: YaRN rope scaling + attention temperature
 *       tuning (attnTempScale=0.1, live from position 0).
 *   <li>MiniCPM5-1B Q8_0 — arch "llama" (the MiniCPM scale paths carry no metadata in this
 *       checkpoint; the live-scales gate is XGraniteTest, granite supplies all four).
 * </ul>
 *
 * Skipped per leg when the checkpoint is not in the HF cache.
 */
class XLlamaTest {

    private static final Path HF_CACHE =
            Path.of(System.getProperty("user.home"), ".cache/huggingface/hub");
    private static final float LOGIT_TOLERANCE = 1e-2f;

    private static Path llamaModel, smollm3Model, mistralModel, minicpmModel;

    @BeforeAll
    static void findModels() throws IOException {
        llamaModel =
                findGguf(
                        "models--unsloth--Llama-3.2-1B-Instruct-GGUF",
                        "Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        smollm3Model = findGguf("models--unsloth--SmolLM3-3B-GGUF", "SmolLM3-3B-Q4_K_M.gguf");
        mistralModel =
                findGguf(
                        "models--mistralai--Ministral-3-3B-Instruct-2512-GGUF",
                        "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf");
        minicpmModel = findGguf("models--openbmb--MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q8_0.gguf");
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
        assumeTrue(llamaModel != null, "Llama-3.2-1B-Instruct-Q4_K_M.gguf not in the HF cache");
        // batchCapacity >= prompt: one prefill chunk
        assertGreedyParity(llamaModel, "The capital of France is", 64, 0);
    }

    @Test
    void greedyParityChunked() throws Exception {
        assumeTrue(llamaModel != null, "Llama-3.2-1B-Instruct-Q4_K_M.gguf not in the HF cache");
        // a ~48-token prompt at batchCapacity 16: the prefill spans three chunks, so decode runs
        // on a KV carry across chunk boundaries (and the lazy tail finishes over one)
        assertGreedyParity(
                llamaModel,
                "The history of the Eiffel Tower begins in 1884, when two engineers at Gustave"
                        + " Eiffel's company sketched a 300-metre iron lattice tower for the 1889"
                        + " Exposition Universelle in Paris. The tower was",
                48,
                16);
    }

    /**
     * The prompt-cache law end to end on a real checkpoint: blocks are captured at ingestion
     * boundaries (the only moments a save is legal), replayed into a fresh state one token short,
     * and the final token is re-ingested - the restored state must then be byte-identical to the
     * live one and continue greedily in lockstep.
     */
    @Test
    void exactStateRestore() throws Exception {
        try (FileChannel channel = FileChannel.open(ModelFixture.LLAMA32_1B_Q8.require())) {
            GGUF gguf = ModelLoader.readGguf(channel, "llama");
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            List<Integer> prompt = new ArrayList<>();
            prompt.add(gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", 1));
            for (int id :
                    tokenizer.encodeToArray(
                            "Cache this exact Llama state across more than one physical history"
                                    + " chunk.")) prompt.add(id);
            int[] ids = prompt.stream().mapToInt(Integer::intValue).toArray();
            int[] prefix = Arrays.copyOf(ids, ids.length - 1);
            int lastToken = ids[ids.length - 1];

            Llama model = Llama.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            var codec = new LlamaStateCodec(model.config());
            int capacity = ids.length + 2;

            List<int[]> spans = new ArrayList<>();
            List<MemorySegment> blocks = new ArrayList<>();
            MemorySegment wholeSpan;
            float[] liveLogits;
            try (Arena arena = Arena.ofConfined();
                    var live = model.newState(capacity, 8)) {
                int from = 0;
                for (Batch chunk : Batch.prepare(List.of(Batch.prefill(prefix)), 8)) {
                    model.ingest(live, chunk);
                    int to = live.position();
                    MemorySegment block = arena.allocate(codec.checkpointBytes(to - from), 64);
                    codec.saveCheckpoint(live, from, to, block);
                    spans.add(new int[] {from, to});
                    blocks.add(block);
                    from = to;
                }
                wholeSpan = arena.allocate(codec.checkpointBytes(prefix.length), 64);
                codec.saveCheckpoint(live, 0, prefix.length, wholeSpan);

                model.ingest(live, Batch.step(lastToken));
                liveLogits =
                        Views.toFloatArray(
                                Views.castToSegmentBacked(model.logits(live), "live logits"),
                                "live logits");

                try (var restored = model.newState(capacity, 8)) {
                    for (int i = 0; i < spans.size(); i++) {
                        codec.restoreCheckpoint(
                                restored, spans.get(i)[0], spans.get(i)[1], blocks.get(i));
                    }
                    restored.resumeAt(prefix.length);

                    MemorySegment resaved =
                            arena.allocate(codec.checkpointBytes(prefix.length), 64);
                    codec.saveCheckpoint(restored, 0, prefix.length, resaved);
                    assertEquals(-1, wholeSpan.mismatch(resaved), "re-saved history bytes");

                    model.ingest(restored, Batch.step(lastToken));
                    assertEquals(ids.length, restored.position());
                    assertEquals(1, restored.outputCount());
                    float[] restoredLogits =
                            Views.toFloatArray(
                                    Views.castToSegmentBacked(
                                            model.logits(restored), "restored logits"),
                                    "restored logits");
                    assertArrayEquals(
                            liveLogits, restoredLogits, 1e-4f, "restored endpoint logits");

                    int token =
                            Ops.argmax(
                                    Views.castToSegmentBacked(model.logits(live), "live logits"),
                                    0,
                                    model.config().vocabularySize());
                    model.ingest(live, Batch.step(token));
                    model.ingest(restored, Batch.step(token));
                    int liveToken =
                            Ops.argmax(
                                    Views.castToSegmentBacked(
                                            model.logits(live), "live continuation"),
                                    0,
                                    model.config().vocabularySize());
                    int restoredToken =
                            Ops.argmax(
                                    Views.castToSegmentBacked(
                                            model.logits(restored), "restored continuation"),
                                    0,
                                    model.config().vocabularySize());
                    assertEquals(liveToken, restoredToken, "continued greedy token");
                }
            }
        }
    }

    @Test
    void greedyParitySmolLm3() throws Exception {
        assumeTrue(smollm3Model != null, "SmolLM3-3B-Q4_K_M.gguf not in the HF cache");
        assertGreedyParity(smollm3Model, "The capital of France is", 48, 0);
    }

    @Test
    void greedyParityMistral() throws Exception {
        assumeTrue(mistralModel != null, "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf not cached");
        assertGreedyParity(mistralModel, "The capital of France is", 48, 0);
    }

    @Test
    void greedyParityMiniCpm() throws Exception {
        assumeTrue(minicpmModel != null, "MiniCPM5-1B-Q8_0.gguf not in the HF cache");
        assertGreedyParity(minicpmModel, "The capital of France is", 48, 0);
    }

    private static void assertGreedyParity(
            Path model, String promptText, int nTokens, int forcedBatchCapacity) throws Exception {
        try (FileChannel channel = FileChannel.open(model)) {
            GGUF gguf = ModelLoader.readGguf(channel, "llama");
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);

            List<Integer> prompt = new ArrayList<>();
            prompt.add(gguf.getValueOrDefault(int.class, "tokenizer.ggml.bos_token_id", 1));
            for (int id : tokenizer.encodeToArray(promptText)) prompt.add(id);
            int[] ids = prompt.stream().mapToInt(Integer::intValue).toArray();

            int[] oldIds;
            float[] oldFirst, oldLast;
            var om =
                    com.qxotic.jinfer.models.llama.Llama.loadModel(
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
            var xm = Llama.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            var xc = xm.config();
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
