// Scalar activation functions shared across architectures (Mamba SSM gates, MoE routers, ...).
// Tensor-wide vectorized variants live on FloatTensor (e.g. siluMultiplyInPlace); these are the
// element-wise scalar forms used in the recurrences and routers that aren't whole-tensor ops.
package com.llama4j;

final class Activations {
    private Activations() {}

    static float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    static float silu(float x) {
        return x * sigmoid(x);
    }

    static float softplus(float x) {
        if (x > 20f) return x;
        if (x < -20f) return (float) Math.exp(x);
        return (float) Math.log1p(Math.exp(x));
    }
}
