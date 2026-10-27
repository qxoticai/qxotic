package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * LFM3-Omni's speech head against llama.cpp ({@code lfm3-oracle speech}): the TTS prompt, the
 * backbone's hidden state, every depthformer pass, the feedback row, the codec and ISTFT on
 * llama.cpp's own codes, and the greedy loop - codebooks revealed in index order, argmax codes, the
 * one deterministic schedule both sides share. The engine's spoken replies are {@link
 * Lfm3SpokenChatTest}'s.
 */
@Tag("integration")
final class Lfm3SpeechParityTest {

    static final String MODELS = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/";
    static final String SENTENCE = "Hello world, one two three.";
    private static final Lfm3Speech.Sampling GREEDY = new Lfm3Speech.Sampling(0f, 0, 1f, 0);
    private static final int[] INDEX_ORDER = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    /**
     * All F32 (the text model widened from BF16): only the order of float sums is left, and the
     * greedy loop speaks every frame llama.cpp does. With the Q8_0 backbone, its activation
     * requantization moves the hidden state at 2e-4, enough to flip an argmax over 2017 codes
     * within a few frames. The F16 mmproj adds ggml's F16 rounding of every codec activation, to
     * which the phase output is sensitive.
     */
    @ParameterizedTest
    @CsvSource({
        "speech-F32-text, LFM3-Omni-2B-F32.gguf, F32, 0.999999, 1e-4, 60, 288",
        "speech-F32, LFM3-Omni-2B-Q8_0.gguf, F32, 0.9995, 1e-4, 60, 24",
        "speech-F16, LFM3-Omni-2B-Q8_0.gguf, F16, 0.9995, 1e-3, 30, 12"
    })
    void theSpeechHeadMatchesLlamaCpp(
            String dump,
            String text,
            String mmproj,
            double hiddenCos,
            double logitBound,
            double snrBound,
            int codesAgreeing)
            throws Exception {
        Path dir = Oracle.dir(dump);
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model =
                    Lfm3.loadModel(
                            TestModels.require(MODELS + text),
                            TestModels.require(MODELS + "mmproj-LFM3-Omni-2B-" + mmproj + ".gguf"),
                            arena);
            Lfm3Speech speech = model.speech();
            int dim = model.configuration().embeddingLength(), cbs = Lfm3Speech.CODEBOOKS;
            int[] prompt = ttsPrompt(model, SENTENCE);
            assertArrayEquals(Oracle.ints(dir.resolve("tokens.i32")), prompt, "the tts prompt");
            float[] h = Oracle.floats(dir.resolve("h.f32"));
            int[] expectedCodes = Oracle.ints(dir.resolve("codes.i32"));

            try (Lfm3.State state = model.newState(256, 64);
                    Lfm3Speech.Stream stream = speech.newStream(GREEDY)) {
                model.ingest(state, Batch.prefill(prompt));
                model.hiddenState(state, stream.hidden);
                Oracle.Score hidden = Oracle.score(h, Oracle.rows(stream.hidden, 1, dim), dim);
                System.out.printf("%s hidden state %s%n", dump, hidden);
                assertTrue(hidden.minCos() > hiddenCos, "hidden " + hidden);

                // the depthformer on llama.cpp's exact hidden state, 4 more codes known each pass
                Views.copyFromArray(stream.hidden, 0, h, 0, dim, "h");
                int[] known = new int[cbs];
                Arrays.fill(known, -1);
                for (int pass = 0; pass < 3; pass++) {
                    float[] logits =
                            Oracle.rows(stream.codeLogits(stream.hidden, known), cbs, speech.vocab);
                    Oracle.Score s =
                            Oracle.score(
                                    Oracle.floats(dir.resolve("logits" + pass + ".f32")),
                                    logits,
                                    speech.vocab);
                    System.out.printf("  depthformer pass %d %s%n", pass, s);
                    assertTrue(s.maxRel() < logitBound, "pass " + pass + " " + s);
                    for (int cb = pass * 4; cb < pass * 4 + 4; cb++) known[cb] = expectedCodes[cb];
                }
                stream.feedback(known);
                Oracle.Score feedback =
                        Oracle.score(
                                Oracle.floats(dir.resolve("fb.f32")),
                                Oracle.rows(stream.feedback, 1, dim),
                                dim);
                assertTrue(feedback.maxRel() < 1e-5, "feedback " + feedback);
            }

            // the codec and ISTFT on llama.cpp's codes, frame by frame, then the tail
            float[] expectedPcm = Oracle.floats(dir.resolve("pcm.f32"));
            float[] pcm = new float[expectedPcm.length];
            int at = 0;
            try (Lfm3Speech.Stream stream = speech.newStream(GREEDY)) {
                int[] codes = new int[cbs];
                for (int f = 0; f < expectedCodes.length / cbs; f++) {
                    System.arraycopy(expectedCodes, f * cbs, codes, 0, cbs);
                    at = append(pcm, at, stream.decode(codes));
                }
                at = append(pcm, at, stream.flush());
            }
            assertEquals(expectedPcm.length, at, "samples");
            double snr = snr(expectedPcm, pcm);
            System.out.printf(
                    "  pcm %d samples %s, SNR %.1f dB%n",
                    at, Oracle.score(expectedPcm, pcm, expectedPcm.length), snr);
            assertTrue(snr > snrBound, "SNR " + snr);

            // the loop: our backbone and head, greedy, the same codes as long as no tie flips
            int[] spoken = new int[expectedCodes.length];
            try (Lfm3.State state = model.newState(256, 64);
                    Lfm3Speech.Stream stream = speech.newStream(GREEDY)) {
                model.ingest(state, Batch.prefill(prompt));
                int[] codes = new int[cbs];
                for (int f = 0; f < expectedCodes.length / cbs; f++) {
                    model.hiddenState(state, stream.hidden);
                    if (!stream.frame(stream.hidden, INDEX_ORDER, GREEDY, new Random(0), codes))
                        break;
                    System.arraycopy(codes, 0, spoken, f * cbs, cbs);
                    model.ingest(state, Batch.embeddings(stream.feedback, 1, false));
                }
            }
            int same = 0;
            while (same < expectedCodes.length && spoken[same] == expectedCodes[same]) same++;
            System.out.printf(
                    "  greedy loop: first %d of %d codes identical%n", same, expectedCodes.length);
            assertTrue(same >= codesAgreeing, "only " + same + " codes agree");
        }
    }

    /**
     * The prompt the engine sends for text to speech: the template's reading conversation, then the
     * opener - which must be LiquidAI's {@code tts} prompt, token for token.
     */
    static int[] ttsPrompt(Lfm3 model, String text) {
        Lfm3ChatTemplate template = new Lfm3ChatTemplate(model);
        Conversation tts =
                new Conversation(
                        List.of(
                                Message.system(
                                        template.speechInstruction("en", null).orElseThrow()),
                                Message.user(text)),
                        List.of(),
                        false);
        List<Integer> ids = new ArrayList<>();
        template.encode(
                tts,
                4096,
                batch -> {
                    for (int id : ((Batch.Input.Tokens) batch.input()).ids()) ids.add(id);
                });
        ids.add(model.opener(false));
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int append(float[] into, int at, float[] chunk) {
        System.arraycopy(chunk, 0, into, at, chunk.length);
        return at + chunk.length;
    }

    private static double snr(float[] reference, float[] actual) {
        double signal = 0, noise = 0;
        for (int i = 0; i < reference.length; i++) {
            signal += (double) reference[i] * reference[i];
            noise += (double) (reference[i] - actual[i]) * (reference[i] - actual[i]);
        }
        return 10 * Math.log10(signal / Math.max(noise, 1e-30));
    }
}
