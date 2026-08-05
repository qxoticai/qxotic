package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Media;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Manual bisect probe: prints stage anchors to compare against the MTMD_DEBUG_GRAPH trace. */
class ConformerBisect {

    @Test
    void printAnchors() throws Exception {
        Path mmproj =
                Path.of(
                        "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        Path wav = Path.of("../../test-fixtures/audio/sine440-3s.wav").toAbsolutePath().normalize();
        Assumptions.assumeTrue(Files.exists(mmproj) && Files.exists(wav));
        Gemma4Conformer tower = Gemma4Conformer.loadModel(mmproj, Arena.ofAuto());
        int t2 = 150, f2 = 64, t4 = 75, f4 = 32;
        tower.tap =
                new Gemma4Conformer.Tap() {
                    @Override
                    public void stageArr(String name, float[] d) {
                        if (name.equals("sub0")) {
                            // trace node_12 (pre-relu, ours post-relu): (t=0,f=0) channels 0..2
                            System.out.printf(
                                    "sub0 t0f0 c0..2: %.4f %.4f %.4f   t0f1 c0: %.4f%n",
                                    d[(0 * t2 + 0) * f2 + 0],
                                    d[(1 * t2 + 0) * f2 + 0],
                                    d[(2 * t2 + 0) * f2 + 0],
                                    d[(0 * t2 + 0) * f2 + 1]);
                        } else {
                            System.out.printf(
                                    "sub1 t0f0 c0..2: %.4f %.4f %.4f%n",
                                    d[(0 * t4 + 0) * f4 + 0],
                                    d[(1 * t4 + 0) * f4 + 0],
                                    d[(2 * t4 + 0) * f4 + 0]);
                        }
                    }

                    @Override
                    public void stage(String name, com.qxotic.jinfer.FloatTensor d) {
                        System.out.printf(
                                "%-8s tok0 d0..3: %9.4f %9.4f %9.4f %9.4f   tok1 d0: %9.4f%n",
                                name,
                                d.getFloat(0),
                                d.getFloat(1),
                                d.getFloat(2),
                                d.getFloat(3),
                                d.getFloat(1024));
                    }
                };
        float[] pcm = ConformerEmbedParityTest.readWav16kMono(wav);
        tower.encode(new Media.Audio(pcm, 16000, 1));
    }
}
