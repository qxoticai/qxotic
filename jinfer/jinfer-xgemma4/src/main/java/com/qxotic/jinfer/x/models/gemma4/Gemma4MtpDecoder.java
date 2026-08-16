package com.qxotic.jinfer.x.models.gemma4;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.kernels.Activations;
import com.qxotic.jinfer.x.kernels.Convert;
import com.qxotic.jinfer.x.kernels.FlashAttention;
import com.qxotic.jinfer.x.kernels.MatMul;
import com.qxotic.jinfer.x.kernels.Norms;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jinfer.x.kernels.RoPE;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * Gemma 4 MTP draft forward: one self-speculative draft step. Given the backbone's pre-final-norm
 * hidden for a position and the token sampled there, predicts the next-token distribution through
 * the 4-layer {@code gemma4-assistant} draft transformer.
 *
 * <p>The draft layers project Q ONLY and attend the BACKBONE's KV rings (shared-KV, {@code
 * shared_kv_layers=4}): the 3 SWA draft layers read backbone own-KV layer {@code ownKv-2}, the full
 * draft layer reads {@code ownKv-1} (llama.cpp {@code ctx_other} map). Geometry: draft 4 heads x
 * {256 SWA, 512 full}, backbone 1 KV head at the same head size -> GQA with {@code kvMul=4} over
 * the backbone's single-KV-head cache; {@link FlashAttention#flashDecode} with null batch buffers
 * reads exactly the cached window {@code [attStart, position]}.
 *
 * <p>Flow (from {@code gemma4-assistant.cpp}): {@code x = backbone.tokEmbd[token]*sqrt(1536)};
 * {@code cur = pre_proj @ concat(x, hidden)[3072] -> 256}; 4 layers (attn_norm, wq, q_norm, rope,
 * Q-only attn, wo, attn_post_norm, +res, ffn gelu-par, ffn_post_norm, +res, *out_scale); {@code
 * output_norm}; draft logits = tied {@code token_embd @ cur} (no softcap); {@code post_proj @ cur
 * -> 1536} chains depth&gt;1. The embedding gather is a span dequant-copy ({@link
 * Convert#copyToF32}) plus a vectorized scale - not the old per-element virtual reads.
 */
public final class Gemma4MtpDecoder {

    private final Gemma4Mtp.Configuration cfg;
    private final Gemma4Mtp.Weights w;
    private final Gemma4 backbone;
    private final int backboneOwnKv;

    // rope tables per attention regime (same theta/head-size convention as the backbone) - its OWN
    // schedules: the draft head carries its own rope_freqs, so it is not simply the backbone's.
    // They cost nothing to keep separate now that a schedule is not a table.
    private final RoPE.Schedule ropeSWA, ropeFull;
    // one row: the draft head decodes a single position at a time
    private final MemoryView<MemorySegment> cosSWA, sinSWA, cosFull, sinFull;

    // scratch (single-token draft; small)
    private final MemoryView<MemorySegment> xh, cur, xb, q, attn, hb, hb2, hNext, draftLogits;
    private final FlashAttention.DecodeScratch decodeScratch;

    public Gemma4MtpDecoder(
            Gemma4Mtp mtp, Gemma4 backbone, MemoryAllocator<MemorySegment> allocator) {
        this.decodeScratch = new FlashAttention.DecodeScratch(allocator);
        this.cfg = mtp.configuration();
        this.w = mtp.weights();
        this.backbone = backbone;
        this.backboneOwnKv = backbone.configuration().ownKvLayers();

        this.ropeSWA = RoPE.plain(cfg.headSizeSWA(), cfg.ropeThetaSWA());
        this.ropeFull =
                w.ropeFreqFactors != null
                        ? RoPE.withFreqFactors(
                                cfg.headSizeFull(), cfg.ropeThetaFull(), w.ropeFreqFactors)
                        : RoPE.plain(cfg.headSizeFull(), cfg.ropeThetaFull());
        this.cosSWA = Views.allocateF32(allocator, cfg.headSizeSWA() / 2);
        this.sinSWA = Views.allocateF32(allocator, cfg.headSizeSWA() / 2);
        this.cosFull = Views.allocateF32(allocator, cfg.headSizeFull() / 2);
        this.sinFull = Views.allocateF32(allocator, cfg.headSizeFull() / 2);

        int dim = cfg.embeddingLength();
        int maxQ = cfg.numberOfHeads() * cfg.headSizeFull();
        this.xh = Views.allocateF32(allocator, 2L * cfg.backboneDim());
        this.cur = Views.allocateF32(allocator, dim);
        this.xb = Views.allocateF32(allocator, dim);
        this.q = Views.allocateF32(allocator, maxQ);
        this.attn = Views.allocateF32(allocator, maxQ);
        this.hb = Views.allocateF32(allocator, cfg.feedForwardLength());
        this.hb2 = Views.allocateF32(allocator, cfg.feedForwardLength());
        this.hNext = Views.allocateF32(allocator, cfg.backboneDim());
        this.draftLogits = Views.allocateF32(allocator, cfg.vocabularySize());
    }

    /**
     * Draft the next-token logits given {@code hidden} (the backbone pre-final-norm hidden of the
     * position where {@code token} was produced) at attention {@code position}. Returns the draft
     * logits (reused buffer); {@link #chainedHidden()} holds the backbone-dim hidden for
     * depth&gt;1.
     */
    public MemoryView<MemorySegment> draft(
            Gemma4.State backboneState,
            MemoryView<MemorySegment> hidden,
            long hiddenOffset,
            int token,
            int position) {
        int bd = cfg.backboneDim(), dim = cfg.embeddingLength();
        float eps = cfg.rmsNormEps();
        // one row for this position, both schedules; every layer reads it
        RoPE.fill(cosSWA, sinSWA, position, 1, cfg.headSizeSWA() / 2, ropeSWA);
        RoPE.fill(cosFull, sinFull, position, 1, cfg.headSizeFull() / 2, ropeFull);

        // xh = concat( backbone.tokEmbd[token] * sqrt(bd) , hidden )
        float scale = (float) Math.sqrt(bd);
        Convert.copyToF32(backbone.weights().tokenEmbeddings(), (long) token * bd, xh, 0, bd);
        Ops.mapInPlace(xh, 0, bd, v -> v * scale);
        Convert.copyF32(hidden, hiddenOffset, xh, bd, bd);

        // cur = pre_projection @ xh   [3072] -> [256]
        MatMul.gemv(w.preProjection, xh, cur, dim, 2 * bd);

        for (int l = 0; l < cfg.numberOfLayers(); l++) {
            boolean swa = cfg.isSWA()[l];
            int headSize = cfg.headSize(l), halfHead = headSize / 2, qDim = cfg.queryDim(l);
            MemoryView<MemorySegment> cos = swa ? cosSWA : cosFull, sin = swa ? sinSWA : sinFull;

            // attn: norm -> Q -> per-head q_norm + rope -> Q-only attention on backbone rings ->
            // wo -> post_norm -> +res
            Norms.rmsnorm(xb, 0, cur, 0, w.attnNorm[l], dim, eps);
            MatMul.gemv(w.wq[l], xb, q, qDim, dim);
            for (int h = 0; h < cfg.numberOfHeads(); h++) {
                Norms.rmsnorm(q, h * headSize, q, h * headSize, w.attnQNorm[l], headSize, eps);
            }
            for (int h = 0; h < cfg.numberOfHeads(); h++) {
                RoPE.applyNeox(q, (long) h * headSize, 0, cos, sin, halfHead);
            }

            int kvSrc = swa ? backboneOwnKv - 2 : backboneOwnKv - 1;
            int bkvDim = backbone.configuration().kvDim(kvSrc);
            int kvMul =
                    cfg.numberOfHeads()
                            / backbone.configuration().numberOfKeyValueHeadsPerLayer()[kvSrc];
            int window = backbone.configuration().slidingWindow();
            int attStart = swa ? Math.max(0, position - window + 1) : 0;
            // the draft has no own KV: null batch buffers -> attend only the backbone cache
            // [attStart, position]
            FlashAttention.flashDecode(
                    q,
                    attn,
                    backboneState.keyCache[kvSrc],
                    backboneState.valueCache[kvSrc],
                    null,
                    null,
                    cfg.numberOfHeads(),
                    position,
                    attStart,
                    headSize,
                    bkvDim,
                    kvMul,
                    1.0f,
                    swa ? window - 1 : 0,
                    null,
                    decodeScratch);
            MatMul.gemv(w.wo[l], attn, xb, dim, qDim);
            Norms.rmsnorm(xb, 0, xb, 0, w.attnPostNorm[l], dim, eps);
            Ops.addInPlace(cur, 0, xb, 0, dim);

            // ffn: norm -> gelu-par gate*up -> down -> post_norm -> +res, then * layer_output_scale
            Norms.rmsnorm(xb, 0, cur, 0, w.ffnNorm[l], dim, eps);
            MatMul.gemv(w.ffnGate[l], xb, hb, cfg.feedForwardLength(), dim);
            MatMul.gemv(w.ffnUp[l], xb, hb2, cfg.feedForwardLength(), dim);
            Activations.geluMultiply(hb, 0, hb2, 0, cfg.feedForwardLength());
            MatMul.gemv(w.ffnDown[l], hb, xb, dim, cfg.feedForwardLength());
            Norms.rmsnorm(xb, 0, xb, 0, w.ffnPostNorm[l], dim, eps);
            Ops.addInPlace(cur, 0, xb, 0, dim);
            final float outScale = w.layerOutputScales[l];
            Ops.mapInPlace(cur, 0, dim, v -> v * outScale);
        }

        Norms.rmsnorm(cur, 0, cur, 0, w.outputNorm, dim, eps);
        MatMul.gemv(w.tokenEmbeddings, cur, draftLogits, cfg.vocabularySize(), dim);
        MatMul.gemv(w.postProjection, cur, hNext, bd, dim);
        return draftLogits;
    }

    /** The backbone-dim hidden produced by the last {@link #draft} call, to chain depth&gt;1. */
    public MemoryView<MemorySegment> chainedHidden() {
        return hNext;
    }
}
