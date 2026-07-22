// Multimodal prompt-cache validation on Gemma 4: image (bidirectional) and audio (causal)
// embeddings flow through CachedSession/PromptCache like tokens - content-hash fingerprints,
// one block per media group, byte-identical restore, and a resume that skips BOTH the media
// encode and the prefill (the double win).
//   E2B (image):        java ... Gemma4MultimodalCacheRun
//   12B (image+audio):  java ... Gemma4MultimodalCacheRun 12b
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class Gemma4MultimodalCacheRun {

    static final Path E2B =
            Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
    static final Path E2B_MMPROJ =
            Path.of(
                    "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
    static final Path B12 =
            Path.of(
                    "/home/mukel/Desktop/playground/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf");
    static final Path B12_MMPROJ =
            Path.of(
                    "/home/mukel/Desktop/playground/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf");

    static int failures;
    static Gemma4 model;
    static TurnTemplate template;
    static StateCodec<Gemma4.State> codec;
    static Set<Integer> stops;
    static long budget;
    static byte[] seed;

    @Test
    @Tag("integration")
    void run() throws Exception {
        Assumptions.assumeTrue(
                java.nio.file.Files.exists(
                        java.nio.file.Path.of(
                                "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf")),
                "model not found:"
                    + " /home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) throws Exception {
        boolean big = args.length > 0 && args[0].equals("12b");
        Path text = big ? B12 : E2B, mmproj = big ? B12_MMPROJ : E2B_MMPROJ;
        budget = big ? 14L << 30 : 4L << 30; // 12B SWA checkpoints are ~hundreds of MB per block
        model = Gemma4.loadModel(text, mmproj, 4096);
        template = model.turnTemplate().orElseThrow();
        codec = model.stateCodec().orElseThrow();
        stops = model.stopTokens();
        seed = PromptCache.modelSeed(text);
        System.out.println(
                "model="
                        + text.getFileName()
                        + " modalities="
                        + model.modalities()
                        + " blockBytes(1)="
                        + (codec.blockBytes(1) >> 20)
                        + "MB");

        battery(
                "image",
                new Part.Blob(solidImage(1f, 0f, 0f)),
                new Part.Blob(solidImage(0f, 0f, 1f)),
                "\nWhat is the dominant color in this image? Answer with one word.",
                "red",
                big ? null : "blue");
        if (big) {
            battery(
                    "audio",
                    new Part.Blob(sine(440)),
                    new Part.Blob(sine(880)),
                    "\nDescribe this audio clip in one short sentence.",
                    null,
                    null);
        }

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            throw new AssertionError("failure(s) - see output above");
        }
        System.out.println("Gemma4MultimodalCacheRun: all checks passed");
    }

    /**
     * The full battery for one modality: cache, cold resume byte-identity, divergence at the media
     * block, and the resume-vs-full TTFT benchmark (encode + prefill both skipped).
     */
    static void battery(
            String name,
            Part.Blob media,
            Part.Blob otherMedia,
            String question,
            String expectWord,
            String expectOther) {
        System.out.println("=== " + name + " ===");
        Message turn = new Message(Role.USER, List.of(media, new Part.Text(question)));

        // -- session A: encode (timed) + ingest (timed) + cached greedy reply --
        PromptCache<Gemma4.State> cache =
                new PromptCache<>(codec, CacheStore.inMemory(), budget, seed);
        CachedSession<Gemma4.State> a =
                CachedSession.resume(model, cache, model.newState(4096, 512), new long[0]);
        long t0 = System.nanoTime();
        List<Batch> batches =
                concat(
                        template.conversationStart(),
                        template.encodeTurn(turn)); // media encode happens here
        double encodeMs = (System.nanoTime() - t0) / 1e6;
        long t1 = System.nanoTime();
        a.ingest(batches);
        double ingestMs = (System.nanoTime() - t1) / 1e6;
        long[] turnFp = a.fingerprints(); // the dual view up to the turn end
        double fullTtft = encodeMs + ingestMs + firstTokenMs(a);
        String reply = decode(a, 16);
        System.out.printf("reply: %s%n", reply.strip());
        int mediaRows = mediaRows(batches);
        check(mediaRows > 0, name + ": media block present (" + mediaRows + " rows)");
        if (expectWord != null) {
            check(
                    reply.toLowerCase().contains(expectWord),
                    name + ": coherent (mentions '" + expectWord + "')");
        }

        // -- cold resume: everything restored, byte-identical through the codec --
        long[] all = a.fingerprints();
        CachedSession<Gemma4.State> b =
                CachedSession.resume(model, cache, model.newState(4096, 512), all);
        check(
                b.position() == all.length,
                name
                        + ": cold resume restores all "
                        + all.length
                        + " positions (got "
                        + b.position()
                        + ")");
        check(
                statesEqual(a.state(), b.state(), all.length),
                name + ": restored state byte-identical via codec");

        // -- divergence inside the media block: only the pre-media text block is reused --
        int[] bounds = blockBounds(batches); // [preMediaEnd, mediaEnd]
        long[] mutated = all.clone();
        mutated[bounds[0] + mediaRows / 2] ^=
                0x5DEECE66DL; // flip a fingerprint inside the media block
        CachedSession<Gemma4.State> d =
                CachedSession.resume(model, cache, model.newState(4096, 512), mutated);
        check(
                d.position() == bounds[0],
                name
                        + ": divergent media resumes only the text prefix ("
                        + d.position()
                        + "/"
                        + all.length
                        + ", media block excluded)");

        // -- re-encode determinism (informational): same media, fresh encode -> same fingerprints?
        // --
        List<Batch> again = concat(template.conversationStart(), template.encodeTurn(turn));
        PromptCache<Gemma4.State> scratch =
                new PromptCache<>(codec, CacheStore.inMemory(), budget, seed);
        CachedSession<Gemma4.State> e =
                CachedSession.resume(model, scratch, model.newState(4096, 512), new long[0]);
        e.ingest(again);
        boolean reEncodeStable = java.util.Arrays.equals(turnFp, e.fingerprints());
        System.out.println(
                "      (re-encode fingerprint-stable: "
                        + reEncodeStable
                        + " - a stateless echo re-encode "
                        + (reEncodeStable
                                ? "hits"
                                : "misses; servers retain the fingerprint stream")
                        + ")");

        // -- the DOUBLE WIN: same media, different question resumes at/after the media block
        // end - both the media encode and the media prefill are skipped --
        long[] sameMediaFp = java.util.Arrays.copyOf(all, bounds[1]);
        CachedSession<Gemma4.State> dw =
                CachedSession.resume(model, cache, model.newState(4096, 512), sameMediaFp);
        check(
                dw.position() >= bounds[1],
                name
                        + ": same media, different question resumes at/after media end ("
                        + dw.position()
                        + " >= "
                        + bounds[1]
                        + ")");

        // -- a genuinely different media on the SAME cache: shares the text prefix, answers
        // differently --
        Message otherTurn = new Message(Role.USER, List.of(otherMedia, new Part.Text(question)));
        CachedSession<Gemma4.State> c =
                CachedSession.resume(model, cache, model.newState(4096, 512), new long[0]);
        c.ingest(concat(template.conversationStart(), template.encodeTurn(otherTurn)));
        c.ingest(template.generationPrompt(true));
        String otherReply = decode(c, 16);
        System.out.printf("other-media reply: %s%n", otherReply.strip());
        if (expectOther != null) {
            check(
                    otherReply.toLowerCase().contains(expectOther),
                    name + ": different media, coherent different answer ('" + expectOther + "')");
        }

        // -- benchmark: resume TTFT (no encode, no prefill) vs the full path --
        double best = Double.MAX_VALUE;
        for (int r = 0; r < 3; r++) {
            long t2 = System.nanoTime();
            CachedSession<Gemma4.State> w =
                    CachedSession.resume(model, cache, model.newState(4096, 512), turnFp);
            if (w.position() != turnFp.length)
                throw new IllegalStateException("warm resume missed");
            w.ingest(template.generationPrompt(true));
            model.logits(w.state()).argmax();
            best = Math.min(best, (System.nanoTime() - t2) / 1e6);
        }
        System.out.printf(
                "TTFT: full %.0fms (encode %.0fms + prefill %.0fms + first token)  vs  resume"
                        + " %.0fms  (%.0fx)%n",
                fullTtft, encodeMs, ingestMs, best, fullTtft / best);
        check(best < fullTtft, name + ": resume TTFT beats the full path");

        // -- FROZEN artifact with MEDIA: freeze the whole tree (both conversations, both
        // images), reopen from disk, and serve the catalog - image KV blocks travel as opaque
        // frozen blobs, content-hash fingerprints and all --
        try {
            java.nio.file.Path artifact = java.nio.file.Files.createTempFile("mm-frozen", ".jkv");
            artifact.toFile().deleteOnExit();
            cache.freeze(artifact);
            com.qxotic.jinfer.cache.FrozenBlocks fz =
                    com.qxotic.jinfer.cache.FrozenBlocks.open(artifact, seed);
            System.out.println("      frozen catalog: " + fz);

            // 1. the first image's turn serves fully from disk, and greedy decode from the
            // restored KV matches the live session's reply (the encode was skipped entirely)
            CachedSession<Gemma4.State> fa =
                    fz.serve(model, codec, seed, model.newState(4096, 512), turnFp);
            check(
                    fa.position() == turnFp.length,
                    name + ": frozen serve restores the media turn (" + fa.position() + ")");
            fa.ingest(template.generationPrompt(true));
            check(
                    decode(fa, 16).equals(reply),
                    name + ": frozen greedy reply identical to the live session");

            // 2. the SECOND image's conversation serves from the same artifact (catalog)
            CachedSession<Gemma4.State> fc =
                    fz.serve(model, codec, seed, model.newState(4096, 512), c.fingerprints());
            check(
                    fc.position() >= bounds[1],
                    name + ": second image serves from the same catalog (" + fc.position() + ")");

            // 3. double win from disk: same media, different question resumes past media end
            CachedSession<Gemma4.State> fd =
                    fz.serve(
                            model,
                            codec,
                            seed,
                            model.newState(4096, 512),
                            java.util.Arrays.copyOf(all, bounds[1]));
            check(
                    fd.position() >= bounds[1],
                    name
                            + ": frozen double win (media encode + prefill skipped, "
                            + fd.position()
                            + " >= "
                            + bounds[1]
                            + ")");

            // 4. divergent media against the artifact resumes only the text prefix
            CachedSession<Gemma4.State> fm =
                    fz.serve(model, codec, seed, model.newState(4096, 512), mutated);
            check(
                    fm.position() == bounds[0],
                    name
                            + ": frozen divergent media resumes the text prefix ("
                            + fm.position()
                            + "/"
                            + all.length
                            + ")");
        } catch (java.io.IOException ioe) {
            throw new RuntimeException(ioe);
        }
    }

    /** genPrompt ingest + first argmax on the session. */
    static double firstTokenMs(CachedSession<Gemma4.State> s) {
        long t = System.nanoTime();
        s.ingest(template.generationPrompt(true));
        model.logits(s.state()).argmax();
        return (System.nanoTime() - t) / 1e6;
    }

    static String decode(CachedSession<Gemma4.State> s, int maxTokens) {
        StringBuilder out = new StringBuilder();
        int tok = model.logits(s.state()).argmax();
        for (int n = 0; n < maxTokens && !stops.contains(tok); n++) {
            out.append(model.tokenizer().decode(new int[] {tok}));
            s.step(tok);
            tok = model.logits(s.state()).argmax();
        }
        s.ingest(template.closeTurn());
        return out.toString();
    }

    /**
     * Position bounds from the prepared batches: [end of the pre-media token block, end of media].
     */
    static int[] blockBounds(List<Batch> batches) {
        int pos = 0, preEnd = -1, mediaEnd = -1;
        for (Batch b : Batch.prepare(batches, 512)) {
            if (b.input() instanceof Batch.Input.Embeddings e) {
                preEnd = pos;
                mediaEnd = pos + e.count();
            }
            pos += b.count();
        }
        return new int[] {preEnd, mediaEnd};
    }

    static int mediaRows(List<Batch> batches) {
        int rows = 0;
        for (Batch b : batches) {
            if (b.input() instanceof Batch.Input.Embeddings e) rows += e.count();
        }
        return rows;
    }

    static boolean statesEqual(Gemma4.State x, Gemma4.State y, int positions) {
        long bytes = codec.blockBytes(positions);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sx = arena.allocate(bytes, 64);
            MemorySegment sy = arena.allocate(bytes, 64);
            codec.save(x, 0, positions, sx);
            codec.save(y, 0, positions, sy);
            return MemorySegment.mismatch(sx, 0, bytes, sy, 0, bytes) == -1;
        }
    }

    /** Solid-color RGB test image (deterministic content, trivially describable). */
    static Media.Image solidImage(float r, float g, float b) {
        int h = 256, w = 256;
        float[] v = new float[h * w * 3];
        for (int i = 0; i < h * w; i++) {
            v[i * 3] = r;
            v[i * 3 + 1] = g;
            v[i * 3 + 2] = b;
        }
        return new Media.Image(v, h, w, 3);
    }

    /** 2s mono 16kHz sine (deterministic PCM). */
    static Media.Audio sine(double hz) {
        int rate = 16000, n = rate * 2;
        float[] pcm = new float[n];
        for (int i = 0; i < n; i++) pcm[i] = (float) (0.4 * Math.sin(2 * Math.PI * hz * i / rate));
        return new Media.Audio(pcm, rate, 1);
    }

    @SafeVarargs
    static List<Batch> concat(List<Batch>... groups) {
        List<Batch> out = new ArrayList<>();
        for (List<Batch> g : groups) out.addAll(g);
        return out;
    }

    static void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }
}
