// Stage 1 gate for the Gemma 4 MTP sidecar loader: load the real gemma4-assistant GGUF and assert
// the
// grounded geometry + that all 49 tensors are present at the expected shapes (req() throws
// otherwise).
//   java ... com.qxotic.llm.Gemma4MtpLoadTest [mtp-sidecar.gguf]
package com.qxotic.llm;

import java.nio.file.Files;
import java.nio.file.Path;

public final class Gemma4MtpLoadTest {

    static int failures;

    static void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }

    public static void main(String[] args) throws Exception {
        Path sidecar =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/unsloth/mtp-gemma-4-E2B-it.gguf");
        if (!Files.exists(sidecar)) {
            System.out.println("Gemma4MtpLoadTest: sidecar not found (" + sidecar + "), skipping");
            return;
        }

        // backbone vocab = 262144 (Gemma 4 E2B); the tied draft head must match it.
        Gemma4Mtp mtp = Gemma4Mtp.loadSidecar(sidecar, 262144);
        Gemma4Mtp.Config c = mtp.config();

        check(c.embeddingLength() == 256, "draft dim 256");
        check(c.backboneDim() == 1536, "backbone hidden dim 1536");
        check(c.numberOfLayers() == 4, "4 draft layers");
        check(c.feedForwardLength() == 2048, "ffn 2048");
        check(c.numberOfHeads() == 4, "4 heads");
        check(c.numberOfKvHeads() == 1, "1 kv head");
        check(c.headSizeFull() == 512 && c.headSizeSWA() == 256, "head sizes 512/256");
        check(c.slidingWindow() == 512, "window 512");
        check(c.ropeThetaFull() == 1_000_000f && c.ropeThetaSWA() == 10_000f, "rope 1e6/1e4");
        check(
                java.util.Arrays.equals(c.isSWA(), new boolean[] {true, true, true, false}),
                "isSWA [T,T,T,F]");
        check(c.vocabularySize() == 262144, "tied head vocab 262144");
        check(c.queryDim(0) == 1024 && c.queryDim(3) == 2048, "queryDim 1024 (swa) / 2048 (full)");

        // All 49 tensors loaded at grounded shapes (req() already threw on any mismatch);
        // spot-check sizes.
        Gemma4Mtp.Weights w = mtp.weights();
        check(w.tokenEmbeddings.size() == 262144L * 256, "tied token_embd [256,262144]");
        check(w.preProjection.size() == 2L * 1536 * 256, "pre_projection [3072,256]");
        check(w.postProjection.size() == 256L * 1536, "post_projection [256,1536]");
        check(
                w.wq[0].size() == 256L * 1024 && w.wq[3].size() == 256L * 2048,
                "wq per-layer widths");
        check(
                w.attnQNorm[0].size() == 256 && w.attnQNorm[3].size() == 512,
                "q_norm per-layer head size");
        check(w.layerOutputScales.length == 4, "4 layer output scales");
        check(
                w.ropeFreqFactors != null && w.ropeFreqFactors.length == 256,
                "rope_freqs 256 factors (full layer)");

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println(
                "Gemma4MtpLoadTest: all checks passed (sidecar loads, geometry + 49 tensors"
                        + " verified)");
    }
}
