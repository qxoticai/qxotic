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
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.Norms;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.kernels.*;
import java.io.IOException;
import java.lang.foreign.Arena;
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
        // NOTHING escapes: rows live in the per-encode scratch; the sink sees an ephemeral view
        // (the Embedder contract) and must copy what it keeps.
        try (Arena scratch = Arena.ofShared()) {
            sink.accept(encode(audio, scratch));
        }
    }

    /** The plan's row count for {@code audio}: mono-16k length over non-overlapping frames. */
    @Override
    public int positions(Media.Audio audio) {
        int monoLen = audio.pcm().length / Math.max(1, audio.channels());
        int n =
                audio.sampleRate() == SAMPLE_RATE
                        ? monoLen
                        : Math.max(
                                1,
                                (int)
                                        Math.round(
                                                monoLen
                                                        * ((double) SAMPLE_RATE
                                                                / audio.sampleRate())));
        return Math.max(1, (n + frameSize - 1) / frameSize);
    }

    /**
     * Encode one audio clip -> projected rows (nTokens x modelDim), returned as a caller-owned
     * GC-managed copy (the {@link #embed} seam skips the copy: its rows die with the per-encode
     * scratch). Contract: at most one encode at a time per pipeline (the state's serial-pipeline
     * law covers its media encodes); the tower is stateless, so many pipelines may share it.
     */
    public FloatTensor encode(Media.Audio audio) {
        try (Arena scratch = Arena.ofShared()) {
            FloatTensor rows = encode(audio, scratch);
            FloatTensor out = FloatTensor.allocateF32(Arena.ofAuto(), (int) rows.size());
            rows.copyTo(0, out, 0, (int) rows.size());
            return out;
        }
    }

    private FloatTensor encode(Media.Audio audio, Arena scratch) {
        float[] pcm = toMono16k(audio);
        int n = pcm.length;
        int nTok = Math.max(1, (n + frameSize - 1) / frameSize); // ceil, non-overlapping frames

        // 1. frame the raw waveform: [nTok, frameSize], last frame zero-padded (no
        // windowing/normalization).
        FloatTensor frames = FloatTensor.allocateF32(scratch, nTok * frameSize);
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
        FloatTensor rows = FloatTensor.allocateF32(scratch, nTok * modelDim);
        mmProj.gemm(frames, frameSize, rows, modelDim, nTok, modelDim, frameSize);
        return rows;
    }

    private static float[] toMono16k(Media.Audio audio) {
        return AudioPreprocess.toMono16k(audio);
    }

    // === loader ===

    public static Gemma4Audio loadModel(Path mmprojPath, Arena arena) throws IOException {
        try (FileChannel fc = FileChannel.open(mmprojPath, StandardOpenOption.READ)) {
            var gguf = ModelLoader.readGguf(fc, mmprojPath.toString());
            return loadModel(mmprojPath, gguf, ModelLoader.loadTensors(fc, gguf, arena), arena);
        }
    }

    /**
     * As {@link #loadModel(Path, Arena)} over an mmproj ALREADY parsed and mapped: one header read
     * and one mapping serve both towers of a sidecar carrying vision AND audio. {@code mmprojPath}
     * is a label for messages here - nothing is read from it.
     */
    public static Gemma4Audio loadModel(
            Path mmprojPath,
            com.qxotic.format.gguf.GGUF gguf,
            Map<String, GGMLTensorEntry> t,
            Arena arena)
            throws IOException {
        {
            int modelDim = gguf.getValueOrDefault(int.class, "clip.audio.projection_dim", 3840);
            float eps =
                    gguf.getValueOrDefault(
                            float.class, "clip.audio.attention.layer_norm_epsilon", 1e-6f);
            GGMLTensorEntry mmProjEntry = t.get("mm.a.input_projection.weight");
            if (mmProjEntry == null) {
                throw new IllegalArgumentException(
                        "'"
                                + mmprojPath.getFileName()
                                + "' declares a gemma4ua audio projector but carries no"
                                + " mm.a.input_projection.weight - the sidecar is malformed");
            }
            FloatTensor mmProj = ModelLoader.loadQuantized(mmProjEntry);
            // The raw-waveform FRAME SIZE (640) is the projection's input dim.
            // clip.audio.num_mel_bins reads
            // 128 (a stale/repurposed value); llama.cpp hardcodes 640 for gemma4ua, so derive it
            // from the weight.
            if (mmProj.size() % modelDim != 0) {
                throw new IllegalArgumentException(
                        "'"
                                + mmprojPath.getFileName()
                                + "': mm.a.input_projection has "
                                + mmProj.size()
                                + " elements, not a multiple of projection_dim "
                                + modelDim
                                + " - the frame size would truncate; the sidecar is malformed");
            }
            int frameSize = (int) (mmProj.size() / modelDim);
            return new Gemma4Audio(frameSize, modelDim, eps, mmProj);
        }
    }
}
