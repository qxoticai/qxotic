package com.qxotic.jinfer.libjinfer;

/**
 * Sample-rate conversion for PCM handed to jinfer_transcribe at a rate the model does not take.
 * Windowed sinc (Lanczos, a = 3), the filter widened by the decimation ratio so a downsample is
 * band-limited. ponytail: direct evaluation, O(n * a * ratio); a polyphase resampler in
 * jinfer-codecs would serve both this face and the decoders when speed matters.
 */
final class Resample {
    private static final int A = 3;

    static float[] to(float[] in, int fromRate, int toRate) {
        if (fromRate == toRate || in.length == 0) return in;
        double step = (double) fromRate / toRate; // input samples per output sample
        double width = Math.max(1, step); // widen the kernel when decimating
        int n = (int) Math.floor(in.length / step);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            double center = i * step;
            int lo = (int) Math.ceil(center - A * width), hi = (int) Math.floor(center + A * width);
            double sum = 0, weight = 0;
            for (int j = Math.max(0, lo); j <= Math.min(in.length - 1, hi); j++) {
                double x = (j - center) / width;
                double w = x == 0 ? 1 : sinc(x) * sinc(x / A);
                sum += w * in[j];
                weight += w;
            }
            out[i] = weight == 0 ? 0 : (float) (sum / weight);
        }
        return out;
    }

    private static double sinc(double x) {
        double px = Math.PI * x;
        return Math.sin(px) / px;
    }

    private Resample() {}
}
