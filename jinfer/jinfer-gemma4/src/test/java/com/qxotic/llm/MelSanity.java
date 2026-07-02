package com.qxotic.llm;
public class MelSanity {
    public static void main(String[] a) {
        int sr = 16000, n = sr; float[] pcm = new float[n];
        for (int i = 0; i < n; i++) pcm[i] = (float) (0.5 * Math.sin(2 * Math.PI * 440 * i / sr));
        AudioPreprocess ap = new AudioPreprocess(128);
        int[] nf = new int[1];
        float[] mel = ap.logMel(pcm, nf);
        double mn = 1e9, mx = -1e9, sum = 0; boolean bad = false;
        for (float v : mel) { if (Float.isNaN(v) || Float.isInfinite(v)) bad = true; mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; }
        // find peak mel bin of frame 50
        int nFrames = nf[0]; int peak = 0; float pv = -1e9f;
        for (int j = 0; j < 128; j++) { float v = mel[j * nFrames + Math.min(50, nFrames - 1)]; if (v > pv) { pv = v; peak = j; } }
        System.out.printf("nMel=128 nFrames=%d  min=%.3f max=%.3f mean=%.3f  finite=%b  peakMelBin(440Hz)=%d%n",
                nFrames, mn, mx, sum / mel.length, !bad, peak);
    }
}
