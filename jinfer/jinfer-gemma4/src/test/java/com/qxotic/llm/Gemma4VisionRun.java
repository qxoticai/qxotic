// Gemma 4 vision: encoder smoke/parity, and end-to-end image -> description.
//   encode only:  java ... Gemma4VisionRun <mmproj.gguf> [image.png]
//   end-to-end:   java ... Gemma4VisionRun e2e <text.gguf> <mmproj.gguf> <image.png> ["prompt"]
package com.qxotic.llm;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.media.FfmpegImageDecoder;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.jinfer.media.ImageIoDecoder;
import com.qxotic.jinfer.Media;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class Gemma4VisionRun {
    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("e2e")) {
            e2e(args[1], args[2], args[3], args.length > 4 ? args[4] : "Describe this image.");
            return;
        }
        if (args.length > 0 && args[0].equals("bench")) { bench(args); return; }
        if (args.length > 0 && args[0].equals("parity")) { parity(args[1]); return; }
        if (args.length > 0 && args[0].equals("uenc")) { uenc(args[1], args.length > 2 ? loadImage(args[2]) : null); return; }
        Gemma4Vision enc = Gemma4Vision.loadModel(Path.of(args[0]));
        Media.Image img = args.length > 1 ? loadImage(args[1]) : synthetic(enc.imageSize);
        System.err.printf("image %dx%d c%d ; vision: dim=%d heads=%d layers=%d ffn=%d merge=%d -> modelDim=%d%n",
                img.width(), img.height(), img.channels(), enc.visionDim, enc.nHead, enc.nLayer, enc.ffnDim, enc.merge, enc.modelDim);
        long t0 = System.nanoTime();
        FloatTensor rows = enc.encode(img);
        double ms = (System.nanoTime() - t0) / 1e6;
        int dim = enc.modelDim, n = (int) (rows.size() / dim);
        double sum = 0, min = Double.MAX_VALUE, max = -Double.MAX_VALUE, abs = 0;
        for (int i = 0; i < n * dim; i++) { float v = rows.getFloat(i); sum += v; abs += Math.abs(v); if (v < min) min = v; if (v > max) max = v; }
        System.out.printf("tokens=%d dim=%d  encode=%.1fms  mean=%.6f meanAbs=%.6f min=%.4f max=%.4f%n",
                n, dim, ms, sum / (n * dim), abs / (n * dim), min, max);
        System.out.printf("row0[0..7]= %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f%n",
                rows.getFloat(0), rows.getFloat(1), rows.getFloat(2), rows.getFloat(3), rows.getFloat(4), rows.getFloat(5), rows.getFloat(6), rows.getFloat(7));
    }

    /** Vision-encoder throughput (pp): warmup + timed reps over one image. Usage: bench <mmproj> [image] [reps] */
    static void bench(String[] args) throws Exception {
        Gemma4Vision enc = Gemma4Vision.loadModel(Path.of(args[1]));
        Media.Image img = args.length > 2 ? loadImage(args[2]) : synthetic(enc.imageSize);
        int reps = args.length > 3 ? Integer.parseInt(args[3]) : 12, warmup = 3;
        FloatTensor r0 = enc.encode(img);
        int dim = enc.modelDim, nTok = (int) (r0.size() / dim), nPatch = nTok * enc.merge * enc.merge;
        for (int w = 0; w < warmup; w++) enc.encode(img);
        double[] t = new double[reps];
        for (int i = 0; i < reps; i++) { long t0 = System.nanoTime(); enc.encode(img); t[i] = (System.nanoTime() - t0) / 1e6; }
        double mean = 0; for (double v : t) mean += v; mean /= reps;
        double sd = 0; for (double v : t) sd += (v - mean) * (v - mean); sd = Math.sqrt(sd / Math.max(1, reps - 1));
        System.out.printf("vision-encoder bench: %dx%d img -> %d patches -> %d tokens (dim %d), %d reps (warm)%n",
                img.width(), img.height(), nPatch, nTok, dim, reps);
        System.out.printf("  %.1f ± %.1f ms/image  |  %.1f img/s  |  %.0f patches/s  |  %.0f out-tok/s%n",
                mean, sd, 1000.0 / mean, nPatch * 1000.0 / mean, nTok * 1000.0 / mean);
    }

    /** image -> Gemma4Vision -> embedding rows -> ingest between <start_of_image>/<end_of_image> -> generate. */
    static void e2e(String textGguf, String mmproj, String imagePath, String prompt) throws Exception {
        Gemma4 model = Gemma4.loadModel(Path.of(textGguf), Path.of(mmproj), 4096);
        var tk = model.tokenizer();
        var sp = tk.getSpecialTokens();
        int bos = sp.getOrDefault("<bos>", 2), sot = sp.getOrDefault("<|turn>", -1), eot = sp.getOrDefault("<turn|>", -1);   // gemma4 turn markers (NOT <start_of_turn>)
        int soi = sp.getOrDefault("<|image>", -1), eoi = sp.getOrDefault("<image|>", -1);   // gemma4 open/close image wrappers
        System.err.printf("modalities=%s  special: bos=%d sot=%d eot=%d soi=%d eoi=%d%n", model.modalities(), bos, sot, eot, soi, eoi);

        @SuppressWarnings("unchecked")
        var embedder = (com.qxotic.jinfer.Embedder<Media.Image>) model.embedder(Media.Image.class).orElseThrow();
        Media.Image img = loadImage(imagePath);
        FloatTensor imgRows = embedder instanceof Gemma4Vision gv ? gv.encode(img)
                : embedder instanceof Gemma4VisionUnified gu ? gu.encode(img) : null;
        int dim = model.config().embeddingLength(), nImg = (int) (imgRows.size() / dim);
        System.err.printf("image %dx%d -> %d vision tokens (dim=%d)%n", img.width(), img.height(), nImg, dim);

        List<Integer> pre = new ArrayList<>();
        pre.add(bos); if (sot >= 0) pre.add(sot); pre.addAll(tk.encode("user\n")); if (soi >= 0) pre.add(soi);
        List<Integer> post = new ArrayList<>();
        if (eoi >= 0) post.add(eoi); post.addAll(tk.encode("\n" + prompt)); if (eot >= 0) post.add(eot);
        if (sot >= 0) post.add(sot); post.addAll(tk.encode("model\n"));

        int cap = model.config().contextLength();
        Gemma4.State s = model.newState(cap, Math.max(nImg, Math.max(pre.size(), post.size())) + 4);
        model.ingest(s, Batch.prefill(arr(pre)));
        model.ingest(s, Batch.embeddings(imgRows, nImg));
        model.ingest(s, Batch.prefill(arr(post)));

        Set<Integer> stops = model.stopTokens();
        int vocab = model.config().vocabularySize();
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s).argmax();
        for (int i = 0; i < 220 && !stops.contains(tok); i++) {
            out.append(tk.decode(tok));
            model.ingest(s, Batch.step(tok));
            tok = model.logits(s).argmax();
        }
        System.out.println("=== Gemma4 (new API) image description ===");
        System.out.println(out);
    }

    /** Unified (gemma4uv, 12b) encoder: raw embedding stats for a numerical diff vs llama.cpp. */
    static void uenc(String mmproj, Media.Image img) throws Exception {
        Gemma4VisionUnified enc = Gemma4VisionUnified.loadModel(Path.of(mmproj));
        if (img == null) img = synthetic(768);
        long t0 = System.nanoTime();
        FloatTensor rows = enc.encode(img);
        double ms = (System.nanoTime() - t0) / 1e6;
        int dim = enc.modelDim, n = (int) (rows.size() / dim);
        double sum = 0, min = Double.MAX_VALUE, max = -Double.MAX_VALUE, abs = 0;
        for (int i = 0; i < n * dim; i++) { float v = rows.getFloat(i); sum += v; abs += Math.abs(v); if (v < min) min = v; if (v > max) max = v; }
        System.out.printf("gemma4uv tokens=%d dim=%d  encode=%.1fms  mean=%.6f meanAbs=%.6f min=%.4f max=%.4f%n",
                n, dim, ms, sum / (n * dim), abs / (n * dim), min, max);
        System.out.printf("row0[0..7]= %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f%n",
                rows.getFloat(0), rows.getFloat(1), rows.getFloat(2), rows.getFloat(3), rows.getFloat(4), rows.getFloat(5), rows.getFloat(6), rows.getFloat(7));
    }

    static int[] arr(List<Integer> l) { int[] a = new int[l.size()]; for (int i = 0; i < a.length; i++) a[i] = l.get(i); return a; }

    /** Decode via the configured backend (ImageCodec: ffmpeg or imageio). RGB, values in [0,1], HWC. */
    static Media.Image loadImage(String path) throws Exception {
        return ImageCodec.load(Path.of(path));
    }

    /** Decode the same image with both backends (ffmpeg + javax.imageio) and report the max abs pixel
     *  difference - proves the two decoders produce identical Media.Image values. */
    static void parity(String path) throws Exception {
        System.out.println("ImageCodec selected backend: " + ImageCodec.decoder().name());
        Media.Image a = new FfmpegImageDecoder().load(Path.of(path));   // ffmpeg
        Media.Image b = new ImageIoDecoder().load(Path.of(path));       // javax.imageio
        System.out.printf("ffmpeg: %dx%d c%d (%d values) | imageio: %dx%d c%d (%d values)%n",
                a.width(), a.height(), a.channels(), a.values().length,
                b.width(), b.height(), b.channels(), b.values().length);
        if (a.width() != b.width() || a.height() != b.height() || a.channels() != b.channels()) {
            System.out.println("DIMENSION MISMATCH"); return;
        }
        float maxAbs = 0; long ndiff = 0;
        for (int i = 0; i < a.values().length; i++) {
            float d = Math.abs(a.values()[i] - b.values()[i]);
            if (d > 0) ndiff++;
            if (d > maxAbs) maxAbs = d;
        }
        System.out.printf("max abs pixel diff = %.6f  (differing values: %d / %d)%n", maxAbs, ndiff, a.values().length);
    }

    static Media.Image synthetic(int S) {
        float[] v = new float[S * S * 3];
        for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) { int idx = (y * S + x) * 3; v[idx] = (float) x / S; v[idx + 1] = (float) y / S; v[idx + 2] = 0.5f; }
        return new Media.Image(v, S, S, 3);
    }
}
