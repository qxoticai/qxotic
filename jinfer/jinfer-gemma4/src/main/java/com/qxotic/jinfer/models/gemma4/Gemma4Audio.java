// Gemma 4 "unified" audio embedder (projector_type = gemma4ua, e.g. gemma-4-12b).
//
// Reference: llama.cpp tools/mtmd/models/gemma4ua.cpp (clip_graph_gemma4ua::build) and
// tools/mtmd/mtmd-audio.cpp (mtmd_audio_preprocessor_gemma4ua::preprocess).
//
// This is the most minimal encoder of the family: it is "encoder-free" - NO FFT, NO mel filterbank,
// NO conformer, NO attention. The raw 16 kHz waveform is chunked into non-overlapping 640-sample
// frames
// (640 samples @ 16 kHz = 40 ms per token, hop = 640, last frame zero-padded), and each frame
// becomes one
// token via a single RMSNorm + linear projection:
//   n_tokens = ceil(n_samples / 640)
//   per frame [640]:  rms_norm(no weight, eps ~1e-6)  ->  mm.a.input_projection (640 -> 3840)
// clip.audio metadata: num_mel_bins is repurposed as the 640-sample FRAME SIZE (not a mel-bin
// count),
// projection_dim = embedding_length of the text model (3840). Attention over the audio tokens is
// CAUSAL
// (mtmd_decode_use_non_causal is false for GEMMA4UA) - unlike the bidirectional image path. Markers
// are
// <|audio> / <audio|>. Backs Gemma4's MultiModal Embedder<Media.Audio> when the mmproj is gemma4ua.
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Embedder;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.GGMLTensorEntry;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.ModelLoader;
import com.qxotic.jinfer.Norms;
import com.qxotic.jinfer.Parallel;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.function.Consumer;

public final class Gemma4Audio implements Embedder<Media.Audio> {

    static final int SAMPLE_RATE = 16000; // gemma4ua fixes the input rate at 16 kHz

    final int frameSize; // raw samples per token (640; clip.audio.num_mel_bins repurposed)
    final int modelDim; // projection_dim = text embedding_length (3840)
    final float eps; // pre-projection RMSNorm eps (~1e-6)
    final FloatTensor mmProj; // [modelDim, frameSize] = mm.a.input_projection (640 -> 3840)

    Gemma4Audio(int frameSize, int modelDim, float eps, FloatTensor mmProj) {
        this.frameSize = frameSize;
        this.modelDim = modelDim;
        this.eps = eps;
        this.mmProj = mmProj;
    }

    @Override
    public void embed(Media.Audio audio, int maxChunkSize, Consumer<FloatTensor> sink) {
        sink.accept(encode(audio));
    }

    /** Encode one audio clip -> projected rows (nTokens x modelDim). */
    public FloatTensor encode(Media.Audio audio) {
        float[] pcm = toMono16k(audio);
        int n = pcm.length;
        int nTok = Math.max(1, (n + frameSize - 1) / frameSize); // ceil, non-overlapping frames

        // 1. frame the raw waveform: [nTok, frameSize], last frame zero-padded (no
        // windowing/normalization).
        FloatTensor frames = FloatTensor.allocateF32(nTok * frameSize);
        Parallel.forRows(
                nTok,
                t -> {
                    long row = (long) t * frameSize;
                    int base = t * frameSize;
                    for (int f = 0; f < frameSize; f++) {
                        int src = base + f;
                        frames.setFloat(row + f, src < n ? pcm[src] : 0f);
                    }
                });

        // 2. per-frame RMSNorm (no weight; embedding_pre_projection_norm), then the linear
        // projection.
        Parallel.forRows(
                nTok,
                t ->
                        Norms.rmsnormNoWeight(
                                frames,
                                (long) t * frameSize,
                                frames,
                                (long) t * frameSize,
                                frameSize,
                                eps));
        FloatTensor rows = FloatTensor.allocateF32(nTok * modelDim);
        mmProj.gemm(frames, frameSize, rows, modelDim, nTok, modelDim, frameSize);
        return rows;
    }

    /** Number of audio tokens a clip of {@code nSamples} 16 kHz samples produces. */
    public int tokenCount(int nSamples) {
        return Math.max(1, (nSamples + frameSize - 1) / frameSize);
    }

    /**
     * Downmix to mono and resample to 16 kHz (linear). llama.cpp resamples with a higher-order
     * kernel, so for exact parity supply already-16 kHz-mono audio; this keeps arbitrary inputs
     * usable.
     */
    private static float[] toMono16k(Media.Audio audio) {
        int ch = Math.max(1, audio.channels());
        float[] in = audio.pcm();
        int frames = in.length / ch;
        float[] mono = new float[frames];
        if (ch == 1) {
            mono = in;
        } else {
            for (int i = 0; i < frames; i++) {
                float s = 0f;
                for (int c = 0; c < ch; c++) s += in[i * ch + c];
                mono[i] = s / ch;
            }
        }
        if (audio.sampleRate() == SAMPLE_RATE) return mono;
        double ratio = (double) SAMPLE_RATE / audio.sampleRate();
        int outLen = (int) Math.round(mono.length * ratio);
        float[] out = new float[Math.max(1, outLen)];
        for (int i = 0; i < out.length; i++) {
            double srcPos = i / ratio;
            int j = (int) srcPos;
            double frac = srcPos - j;
            float a = mono[Math.min(j, mono.length - 1)];
            float b = mono[Math.min(j + 1, mono.length - 1)];
            out[i] = (float) (a + (b - a) * frac);
        }
        return out;
    }

    // === loader ===

    public static Gemma4Audio loadModel(Path mmprojPath) throws IOException {
        try (FileChannel fc = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            var gguf = ModelLoader.readGguf(fc, mmprojPath.toString());
            Map<String, GGMLTensorEntry> t = ModelLoader.loadTensors(fc, gguf);
            int modelDim = gguf.getValueOrDefault(int.class, "clip.audio.projection_dim", 3840);
            float eps =
                    gguf.getValueOrDefault(
                            float.class, "clip.audio.attention.layer_norm_epsilon", 1e-6f);
            FloatTensor mmProj = ModelLoader.loadQuantized(t.get("mm.a.input_projection.weight"));
            // The raw-waveform FRAME SIZE (640) is the projection's input dim.
            // clip.audio.num_mel_bins reads
            // 128 (a stale/repurposed value); llama.cpp hardcodes 640 for gemma4ua, so derive it
            // from the weight.
            int frameSize = (int) (mmProj.size() / modelDim);
            return new Gemma4Audio(frameSize, modelDim, eps, mmProj);
        }
    }
}
