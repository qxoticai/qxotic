// Stage-1 gate for the Gemma 4 MTP sidecar loader: load the real gemma4-assistant GGUF
// and assert the grounded geometry + that all 49 tensors are present at the expected shapes
// (req() throws otherwise).
package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class Gemma4MtpLoadTest {

    private static final String SIDECAR_REF =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf";

    @Test
    @Tag("integration")
    void sidecarLoadsAtGroundedGeometry() throws Exception {
        Path sidecar = TestModels.require(SIDECAR_REF);

        // backbone vocab = 262144 (Gemma 4 E2B); the tied draft head must match it.
        Gemma4Mtp mtp = Gemma4Mtp.loadSidecar(sidecar, 262144, Arena.ofAuto());
        Gemma4Mtp.Configuration c = mtp.configuration();

        assertEquals(256, c.embeddingLength(), "draft dim");
        assertEquals(1536, c.backboneDim(), "backbone hidden dim");
        assertEquals(4, c.numberOfLayers(), "draft layers");
        assertEquals(2048, c.feedForwardLength(), "ffn");
        assertEquals(4, c.numberOfHeads(), "heads");
        assertEquals(1, c.numberOfKvHeads(), "kv head");
        assertEquals(512, c.headSizeFull(), "head size full");
        assertEquals(256, c.headSizeSWA(), "head size swa");
        assertEquals(512, c.slidingWindow(), "window");
        assertEquals(1_000_000f, c.ropeThetaFull(), "rope full");
        assertEquals(10_000f, c.ropeThetaSWA(), "rope swa");
        assertArrayEquals(new boolean[] {true, true, true, false}, c.isSWA(), "isSWA");
        assertEquals(262144, c.vocabularySize(), "tied head vocab");
        assertEquals(1024, c.queryDim(0), "queryDim swa");
        assertEquals(2048, c.queryDim(3), "queryDim full");

        // All 49 tensors loaded at grounded shapes (req() already threw on any mismatch);
        // spot-check sizes.
        Gemma4Mtp.Weights w = mtp.weights();
        assertEquals(
                262144L * 256, w.tokenEmbeddings.logicalSize(), "tied token_embd [256,262144]");
        assertEquals(2L * 1536 * 256, w.preProjection.logicalSize(), "pre_projection [3072,256]");
        assertEquals(256L * 1536, w.postProjection.logicalSize(), "post_projection [256,1536]");
        assertEquals(256L * 1024, w.wq[0].logicalSize(), "wq swa width");
        assertEquals(256L * 2048, w.wq[3].logicalSize(), "wq full width");
        assertEquals(256, w.attnQNorm[0].logicalSize(), "q_norm swa head size");
        assertEquals(512, w.attnQNorm[3].logicalSize(), "q_norm full head size");
        assertEquals(4, w.layerOutputScales.length, "layer output scales");
        assertNotNull(w.ropeFreqFactors, "rope_freqs present");
        assertEquals(256, w.ropeFreqFactors.length, "rope_freqs factors (full layer)");
    }

    @Test
    @Tag("integration")
    void sidecarsCommute() throws Exception {
        Path text = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0");
        Path sidecar = TestModels.require(SIDECAR_REF);
        Path mmproj = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        try (Arena arena = Arena.ofConfined()) {
            Gemma4 mtpFirst = Gemma4.loadWithMtp(text, sidecar, arena).withMedia(mmproj, arena);
            Gemma4 mediaFirst =
                    Gemma4.loadModel(text, arena)
                            .withMedia(mmproj, arena)
                            .attachMtp(sidecar, arena);
            for (Gemma4 model : new Gemma4[] {mtpFirst, mediaFirst}) {
                assertTrue(model.speculationReady(), "the draft head survives either order");
                assertTrue(
                        model.projector(com.qxotic.jinfer.media.Media.Image.class).isPresent(),
                        "the vision projector survives either order");
            }
        }
    }
}
