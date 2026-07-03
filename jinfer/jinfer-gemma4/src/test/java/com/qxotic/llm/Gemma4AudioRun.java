// Gemma 4 audio (gemma4ua): encoder stats, and end-to-end audio -> description.
//   encode only:  java ... Gemma4AudioRun encode <mmproj.gguf> <audio.wav>
//   end-to-end:   java ... Gemma4AudioRun e2e <text.gguf> <mmproj.gguf> <audio.wav> ["prompt"]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Gemma4AudioRun {
    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("e2e")) {
            e2e(args[1], args[2], args[3], args.length > 4 ? args[4] : "Describe this audio.");
            return;
        }
        if (args.length > 0 && args[0].equals("aingest")) {
            aingest(args[1], args[2], args[3], args.length > 4 ? Integer.parseInt(args[4]) : 10);
            return;
        }
        // encode: <mmproj> <audio.*>
        Gemma4Audio enc = Gemma4Audio.loadModel(Path.of(args[1]));
        Media.Audio audio = com.qxotic.jinfer.media.AudioCodec.load(Path.of(args[2]));   // ffmpeg -> 16k mono, any format
        long t0 = System.nanoTime();
        FloatTensor rows = enc.encode(audio);
        double ms = (System.nanoTime() - t0) / 1e6;
        int dim = enc.modelDim, n = (int) (rows.size() / dim);
        double sum = 0, min = Double.MAX_VALUE, max = -Double.MAX_VALUE, abs = 0;
        for (int i = 0; i < n * dim; i++) { float v = rows.getFloat(i); sum += v; abs += Math.abs(v); if (v < min) min = v; if (v > max) max = v; }
        double mean = sum / (n * dim), var = 0;
        for (int i = 0; i < n * dim; i++) { double d = rows.getFloat(i) - mean; var += d * d; }
        System.out.printf("tokens=%d dim=%d  encode=%.1fms  mean=%.6f meanAbs=%.6f std=%.6f min=%.4f max=%.4f%n",
                n, dim, ms, mean, abs / (n * dim), Math.sqrt(var / (n * dim)), min, max);
        System.out.printf("row0[0..7]= %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f%n",
                rows.getFloat(0), rows.getFloat(1), rows.getFloat(2), rows.getFloat(3), rows.getFloat(4), rows.getFloat(5), rows.getFloat(6), rows.getFloat(7));
    }

    /** Times ONLY the model ingest of the audio embeddings (encode + resample excluded): warmup, then
     *  timed reps of {@code ingest(Batch.embeddings)} with the state reset each rep. This is the audio
     *  prefill throughput. Usage: aingest &lt;text.gguf&gt; &lt;mmproj.gguf&gt; &lt;audio.wav&gt; [reps] */
    static void aingest(String textGguf, String mmproj, String audioPath, int reps) throws Exception {
        Gemma4 model = Gemma4.loadModel(Path.of(textGguf), Path.of(mmproj), 4096);
        @SuppressWarnings("unchecked")
        var embedder = (com.qxotic.jinfer.Embedder<Media.Audio>) model.embedder(Media.Audio.class).orElseThrow();
        Media.Audio audio = loadWav(audioPath);
        FloatTensor rows = ((Gemma4Audio) embedder).encode(audio);   // encode + resample NOT timed
        int dim = model.config().embeddingLength(), n = (int) (rows.size() / dim);
        int cap = n + 8;   // fresh small-context state per rep (reset() is unsupported); alloc is outside the timed region
        for (int w = 0; w < 3; w++) model.ingest(model.newState(cap, cap), Batch.embeddings(rows, n, false));   // warmup
        double best = Double.MAX_VALUE, sum = 0;
        for (int r = 0; r < reps; r++) {
            Gemma4.State s = model.newState(cap, cap);   // allocation NOT timed
            long t0 = System.nanoTime();
            model.ingest(s, Batch.embeddings(rows, n, false));
            double ms = (System.nanoTime() - t0) / 1e6;
            best = Math.min(best, ms); sum += ms;
        }
        System.out.printf("audio ingest: %d tokens (dim=%d)  best=%.1f ms (%.1f tok/s)  mean=%.1f ms (%.1f tok/s)%n",
                n, dim, best, n / (best / 1000.0), sum / reps, n / ((sum / reps) / 1000.0));
    }

    /** audio -> Gemma4Audio -> rows -> ingest between <|audio>/<audio|> (CAUSAL) -> generate. */
    static void e2e(String textGguf, String mmproj, String audioPath, String prompt) throws Exception {
        Gemma4 model = Gemma4.loadModel(Path.of(textGguf), Path.of(mmproj), 4096);
        var tk = model.tokenizer();
        var sp = tk.getSpecialTokens();
        int bos = sp.getOrDefault("<bos>", 2), sot = sp.getOrDefault("<|turn>", -1), eot = sp.getOrDefault("<turn|>", -1);
        int soa = sp.getOrDefault("<|audio>", -1), eoa = sp.getOrDefault("<audio|>", -1);   // gemma4 audio markers
        System.err.printf("modalities=%s  special: bos=%d sot=%d eot=%d soa=%d eoa=%d%n", model.modalities(), bos, sot, eot, soa, eoa);

        @SuppressWarnings("unchecked")
        var embedder = (com.qxotic.jinfer.Embedder<Media.Audio>) model.embedder(Media.Audio.class).orElseThrow();
        Media.Audio audio = com.qxotic.jinfer.media.AudioCodec.load(Path.of(audioPath));   // ffmpeg decodes + resamples to 16k mono
        FloatTensor rows = (embedder instanceof Gemma4Audio ga) ? ga.encode(audio) : null;
        int dim = model.config().embeddingLength(), n = (int) (rows.size() / dim);
        System.err.printf("audio %.2fs @%dHz -> %d audio tokens (dim=%d)%n", audio.pcm().length / (float) audio.sampleRate(), audio.sampleRate(), n, dim);

        List<Integer> pre = new ArrayList<>();
        pre.add(bos); if (sot >= 0) pre.add(sot); pre.addAll(tk.encode("user\n")); if (soa >= 0) pre.add(soa);
        List<Integer> post = new ArrayList<>();
        if (eoa >= 0) post.add(eoa); post.addAll(tk.encode("\n" + prompt)); if (eot >= 0) post.add(eot);
        if (sot >= 0) post.add(sot); post.addAll(tk.encode("model\n"));

        int cap = model.config().contextLength();
        Gemma4.State s = model.newState(cap, Math.max(n, Math.max(pre.size(), post.size())) + 4);
        model.ingest(s, Batch.prefill(arr(pre)));
        model.ingest(s, Batch.embeddings(rows, n, false));   // audio is CAUSAL (gemma4ua)
        model.ingest(s, Batch.prefill(arr(post)));

        Set<Integer> stops = model.stopTokens();
        int vocab = model.config().vocabularySize();
        StringBuilder out = new StringBuilder();
        int tok = LLM.argmax(model.logits(s), vocab);
        for (int i = 0; i < 220 && !stops.contains(tok); i++) {
            out.append(tk.decode(tok));
            model.ingest(s, Batch.step(tok));
            tok = LLM.argmax(model.logits(s), vocab);
        }
        System.out.println("=== Gemma4 (new API) audio description ===");
        System.out.println(out);
    }

    static int[] arr(List<Integer> l) { int[] a = new int[l.size()]; for (int i = 0; i < a.length; i++) a[i] = l.get(i); return a; }

    /** Minimal 16-bit PCM WAV reader -> Media.Audio (float [-1,1]). */
    static Media.Audio loadWav(String path) throws Exception {
        try (RandomAccessFile f = new RandomAccessFile(new File(path), "r")) {
            byte[] hdr = new byte[12];
            f.readFully(hdr);
            int sampleRate = 16000, channels = 1, bits = 16;
            long dataPos = -1, dataLen = 0;
            while (f.getFilePointer() + 8 <= f.length()) {
                byte[] cid = new byte[4]; f.readFully(cid);
                int csz = Integer.reverseBytes(f.readInt());
                String id = new String(cid);
                if (id.equals("fmt ")) {
                    byte[] fmt = new byte[csz]; f.readFully(fmt);
                    channels = (fmt[2] & 0xff) | ((fmt[3] & 0xff) << 8);
                    sampleRate = (fmt[4] & 0xff) | ((fmt[5] & 0xff) << 8) | ((fmt[6] & 0xff) << 16) | ((fmt[7] & 0xff) << 24);
                    bits = (fmt[14] & 0xff) | ((fmt[15] & 0xff) << 8);
                } else if (id.equals("data")) {
                    dataPos = f.getFilePointer(); dataLen = csz; f.seek(f.getFilePointer() + csz);
                } else {
                    f.seek(f.getFilePointer() + csz + (csz & 1));
                }
            }
            if (dataPos < 0 || bits != 16) throw new IllegalArgumentException("expected 16-bit PCM WAV with a data chunk");
            f.seek(dataPos);
            byte[] raw = new byte[(int) dataLen]; f.readFully(raw);
            int nSamp = raw.length / 2;
            float[] pcm = new float[nSamp];
            for (int i = 0; i < nSamp; i++) {
                short v = (short) ((raw[2 * i] & 0xff) | (raw[2 * i + 1] << 8));
                pcm[i] = v / 32768f;
            }
            return new Media.Audio(pcm, sampleRate, channels);
        }
    }
}
