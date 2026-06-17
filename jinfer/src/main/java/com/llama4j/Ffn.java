// Feed-forward (MLP) blocks shared across architectures. The dense FFN is the atom — a single
// gate/up/activation/down MLP — used both as a dense layer and as a model's always-on shared expert.
// MoE routing and routed-expert dispatch stay per-architecture (they diverge by router and grouping);
// this file holds only the dense compute they have in common. gpt-oss's biased, clamped-SwiGLU
// experts stay custom.
package com.llama4j;

final class Ffn {
    private Ffn() {}

    /** Activation/gate form of a dense FFN. The GLU forms gate with a second projection; RELU_SQR is
     *  ungated (squared-ReLU on the up projection only, no gate). gpt-oss's clamped SwiGLU stays custom. */
    enum Act { SILU_GLU, RELU_SQR }

    /**
     * Dense FFN over {@code rows} rows: {@code out = down( act(gate·in) ⊙ up·in )} (gated), or
     * {@code out = down( act(up·in) )} when {@code gate == null} (ungated, RELU_SQR). {@code in}/{@code out}
     * are row-major at stride {@code dim}; {@code hb}/{@code hb2} are hidden scratch at stride
     * {@code hiddenDim}. {@code gemm} routes {@code rows == 1} to gemv, so this serves both decode and
     * prefill. Used by dense layers and shared/always-on experts; MoE routing stays per-architecture.
     */
    static void dense(FloatTensor gate, FloatTensor up, FloatTensor down,
                      FloatTensor in, FloatTensor hb, FloatTensor hb2, FloatTensor out,
                      int rows, int dim, int hiddenDim, Act act) {
        if (gate != null) gate.gemm(in, dim, hb, hiddenDim, rows, hiddenDim, dim);
        up.gemm(in, dim, hb2, hiddenDim, rows, hiddenDim, dim);
        FloatTensor h = switch (act) {
            case SILU_GLU -> { hb.siluMultiplyInPlace(0, hb2, 0, rows * hiddenDim); yield hb; }
            case RELU_SQR -> { hb2.reluSqrInPlace(0, rows * hiddenDim); yield hb2; }
        };
        down.gemm(h, hiddenDim, out, dim, rows, dim, hiddenDim);
    }
}
