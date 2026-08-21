package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * Gemma 4 MTP (multi-token prediction / self-speculative draft head), arch {@code
 * gemma4-assistant}. A 4-layer draft transformer that, at a decoded position, predicts the NEXT
 * token from the backbone's final hidden and the just-sampled token, then chains its own hidden to
 * draft further.
 *
 * <p>Structure (from llama.cpp {@code src/models/gemma4-assistant.cpp}, verified against the
 * sidecar GGUF): {@code xh = concat(backbone.tokEmbd[next]*sqrt(1536), backboneHidden[1536])} →
 * {@code pre_projection[3072,256]} → 4 layers → {@code output_norm} → tied {@code
 * token_embd[256,vocab]} for draft logits; {@code post_projection[256,1536]} produces the next
 * backbone-dim hidden to chain depth&gt;1.
 *
 * <p>The draft layers project Q ONLY (no K/V weights, {@code shared_kv_layers=4}) and attend
 * against the BACKBONE's KV cache: SWA draft layers (0-2) read backbone layer {@code
 * ownKvLayers-2}, the full draft layer (3) reads {@code ownKvLayers-1} - exactly {@link
 * Gemma4.Configuration#kvSourceLayer}'s mapping, and llama.cpp's assistant KV-share map. This
 * shared-KV Q-only attention is the crux the draft forward ({@link Gemma4MtpDecoder}) implements
 * and the token-identity gate verifies.
 *
 * <p>This class is the load + capability surface: a shape-verified sidecar reader. The draft
 * forward and the speculative loop are {@link Gemma4MtpDecoder} and {@link Gemma4Speculative},
 * driven directly (a core MTP seam waits for a second family to shape it).
 */
public final class Gemma4Mtp {

    /** Draft-transformer geometry, read from the {@code gemma4-assistant.*} sidecar metadata. */
    public record Configuration(
            int embeddingLength, // draft dim (256)
            int backboneDim, // embedding_length_out (1536) - the backbone hidden the draft consumes
            int numberOfLayers, // 4
            int feedForwardLength, // 2048
            int numberOfHeads, // 4
            int numberOfKvHeads, // 1 (unused for projection - drafts share backbone KV)
            int headSizeFull, // 512
            int headSizeSWA, // 256
            int slidingWindow, // 512
            float rmsNormEps,
            float ropeThetaFull, // 1e6
            float ropeThetaSWA, // 1e4
            boolean[] isSWA, // [true,true,true,false]
            int vocabularySize) {

        int headSize(int layer) {
            return isSWA[layer] ? headSizeSWA : headSizeFull;
        }

        int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }
    }

    /**
     * Draft weights. {@code tokenEmbeddings} is tied: the draft input embedding AND the LM head.
     */
    public static final class Weights {
        final MemoryView<MemorySegment> tokenEmbeddings; // token_embd.weight [256, vocab], tied
        final MemoryView<MemorySegment> preProjection; // nextn.pre_projection [3072, 256]
        final MemoryView<MemorySegment> postProjection; // nextn.post_projection [256, 1536]
        final MemoryView<MemorySegment> outputNorm; // output_norm.weight [256]
        final MemoryView<MemorySegment>[] attnNorm, attnQNorm, attnPostNorm;
        final MemoryView<MemorySegment>[] ffnNorm, ffnPostNorm;
        final MemoryView<MemorySegment>[] wq, wo, ffnGate, ffnUp, ffnDown;
        final float[] layerOutputScales;
        final float[] ropeFreqFactors; // rope_freqs.weight (full-layer rope factors)

        Weights(
                MemoryView<MemorySegment> tokenEmbeddings,
                MemoryView<MemorySegment> preProjection,
                MemoryView<MemorySegment> postProjection,
                MemoryView<MemorySegment> outputNorm,
                MemoryView<MemorySegment>[] attnNorm,
                MemoryView<MemorySegment>[] attnQNorm,
                MemoryView<MemorySegment>[] attnPostNorm,
                MemoryView<MemorySegment>[] ffnNorm,
                MemoryView<MemorySegment>[] ffnPostNorm,
                MemoryView<MemorySegment>[] wq,
                MemoryView<MemorySegment>[] wo,
                MemoryView<MemorySegment>[] ffnGate,
                MemoryView<MemorySegment>[] ffnUp,
                MemoryView<MemorySegment>[] ffnDown,
                float[] layerOutputScales,
                float[] ropeFreqFactors) {
            this.tokenEmbeddings = tokenEmbeddings;
            this.preProjection = preProjection;
            this.postProjection = postProjection;
            this.outputNorm = outputNorm;
            this.attnNorm = attnNorm;
            this.attnQNorm = attnQNorm;
            this.attnPostNorm = attnPostNorm;
            this.ffnNorm = ffnNorm;
            this.ffnPostNorm = ffnPostNorm;
            this.wq = wq;
            this.wo = wo;
            this.ffnGate = ffnGate;
            this.ffnUp = ffnUp;
            this.ffnDown = ffnDown;
            this.layerOutputScales = layerOutputScales;
            this.ropeFreqFactors = ropeFreqFactors;
        }
    }

    private final Configuration configuration;
    private final Weights weights;

    private Gemma4Mtp(Configuration configuration, Weights weights) {
        this.configuration = configuration;
        this.weights = weights;
    }

    public Configuration configuration() {
        return configuration;
    }

    public Weights weights() {
        return weights;
    }

    /**
     * Loads and shape-verifies the MTP sidecar. {@code backboneVocab} is the backbone's vocab,
     * which the tied draft head must match (the draft predicts backbone tokens).
     */
    public static Gemma4Mtp loadSidecar(Path sidecar, int backboneVocab, Arena arena)
            throws IOException {
        try (FileChannel fc = FileChannel.open(sidecar, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fc, sidecar.toString());
            String arch = gguf.getValueOrDefault(String.class, "general.architecture", "");
            if (!arch.equals("gemma4-assistant")) {
                throw new IllegalArgumentException(
                        "not a gemma4-assistant MTP sidecar: arch=" + arch);
            }
            Configuration configuration = readConfiguration(gguf, backboneVocab);
            Map<String, MemoryView<MemorySegment>> tensors =
                    ModelLoader.loadTensors(fc, gguf, arena);
            Weights weights = loadWeights(tensors, configuration);
            return new Gemma4Mtp(configuration, weights);
        }
    }

    private static Configuration readConfiguration(GGUF gguf, int backboneVocab) {
        String p = "gemma4-assistant.";
        int layers = gguf.getValue(int.class, p + "block_count");
        int headFull = gguf.getValue(int.class, p + "attention.key_length");
        int headSWA = gguf.getValue(int.class, p + "attention.key_length_swa");
        // isSWA per layer: full-attention layers carry the larger head (key_length); SWA carry
        // key_length_swa. The GGUF flags the pattern via the per-layer attn_q width, matched here
        // to the Gemma convention [SWA, SWA, SWA, FULL] (blk0-2 head=key_length_swa, blk3
        // head=key_length).
        boolean[] isSWA = new boolean[layers];
        for (int i = 0; i < layers; i++) isSWA[i] = i < layers - 1;
        return new Configuration(
                gguf.getValue(int.class, p + "embedding_length"),
                gguf.getValue(int.class, p + "embedding_length_out"),
                layers,
                gguf.getValue(int.class, p + "feed_forward_length"),
                gguf.getValue(int.class, p + "attention.head_count"),
                headCountKv(gguf, p + "attention.head_count_kv", layers),
                headFull,
                headSWA,
                gguf.getValue(int.class, p + "attention.sliding_window"),
                gguf.getValueOrDefault(float.class, p + "attention.layer_norm_rms_epsilon", 1e-6f),
                gguf.getValueOrDefault(float.class, p + "rope.freq_base", 1000000f),
                gguf.getValueOrDefault(float.class, p + "rope.freq_base_swa", 10000f),
                isSWA,
                backboneVocab);
    }

    /**
     * The KV-head count is descriptive only (drafts project Q and share the backbone's KV), and
     * sidecars disagree on its shape: E2B stores a scalar {@code 1}, the 26B stores per-layer
     * {@code [8, 8, 8, 2]}. Read either; a per-layer array is validated against the layer count and
     * summarized by its first entry.
     */
    private static int headCountKv(GGUF gguf, String key, int layers) {
        Object value = gguf.getValue(Object.class, key);
        if (value instanceof int[] perLayer) {
            if (perLayer.length != layers) {
                throw new IllegalArgumentException(
                        key
                                + ": per-layer array of "
                                + perLayer.length
                                + " does not match block_count "
                                + layers);
            }
            return perLayer[0];
        }
        return (Integer) value;
    }

    private static Weights loadWeights(Map<String, MemoryView<MemorySegment>> t, Configuration c) {
        int n = c.numberOfLayers();
        int dim = c.embeddingLength();

        MemoryView<MemorySegment> tokenEmbeddings =
                req(t, "token_embd.weight", c.vocabularySize() * (long) dim);
        MemoryView<MemorySegment> preProjection =
                req(t, "nextn.pre_projection.weight", 2L * c.backboneDim() * dim);
        MemoryView<MemorySegment> postProjection =
                req(t, "nextn.post_projection.weight", (long) dim * c.backboneDim());
        MemoryView<MemorySegment> outputNorm = req(t, "output_norm.weight", dim);

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] attnNorm = new MemoryView[n], attnQNorm = new MemoryView[n];
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] attnPostNorm = new MemoryView[n],
                ffnNorm = new MemoryView[n],
                ffnPostNorm = new MemoryView[n];
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] wq = new MemoryView[n], wo = new MemoryView[n];
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] ffnGate = new MemoryView[n],
                ffnUp = new MemoryView[n],
                ffnDown = new MemoryView[n];
        float[] outScales = new float[n];

        for (int i = 0; i < n; i++) {
            String b = "blk." + i + ".";
            int qDim = c.queryDim(i);
            int hs = c.headSize(i);
            attnNorm[i] = req(t, b + "attn_norm.weight", dim);
            wq[i] = req(t, b + "attn_q.weight", (long) dim * qDim);
            wo[i] = req(t, b + "attn_output.weight", (long) qDim * dim);
            attnQNorm[i] = req(t, b + "attn_q_norm.weight", hs);
            attnPostNorm[i] = req(t, b + "post_attention_norm.weight", dim);
            ffnNorm[i] = req(t, b + "ffn_norm.weight", dim);
            ffnGate[i] = req(t, b + "ffn_gate.weight", (long) dim * c.feedForwardLength());
            ffnUp[i] = req(t, b + "ffn_up.weight", (long) dim * c.feedForwardLength());
            ffnDown[i] = req(t, b + "ffn_down.weight", (long) c.feedForwardLength() * dim);
            ffnPostNorm[i] = req(t, b + "post_ffw_norm.weight", dim);
            outScales[i] =
                    Views.getFloat(
                            req(t, b + "layer_output_scale.weight", 1),
                            0,
                            b + "layer_output_scale.weight");
        }
        float[] ropeFreqFactors = ModelLoader.ropeFreqFactors(t).orElse(null);
        return new Weights(
                tokenEmbeddings,
                preProjection,
                postProjection,
                outputNorm,
                attnNorm,
                attnQNorm,
                attnPostNorm,
                ffnNorm,
                ffnPostNorm,
                wq,
                wo,
                ffnGate,
                ffnUp,
                ffnDown,
                outScales,
                ropeFreqFactors);
    }

    private static MemoryView<MemorySegment> req(
            Map<String, MemoryView<MemorySegment>> t, String name, long expectedElems) {
        MemoryView<MemorySegment> view = ModelLoader.require(t, name);
        long got = view.logicalSize();
        if (got != expectedElems) {
            throw new IllegalStateException(
                    name + " has " + got + " elements, expected " + expectedElems);
        }
        return view;
    }
}
