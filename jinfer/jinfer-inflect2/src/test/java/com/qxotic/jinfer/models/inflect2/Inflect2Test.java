package com.qxotic.jinfer.models.inflect2;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.models.inflect2.frontend.TextNormalizer;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The frontend checks run everywhere; the model checks resolve their GGUF through {@link
 * TestModels} and SKIP when it is absent - {@code scripts/download-models.sh --only Inflect}
 * fetches it.
 */
class Inflect2Test {

    private static final String REF = "hf.co/remixerdec/Inflect-Nano-v2-GGUF:Q8_0";

    // ── frontend (no weights) ─────────────────────────────────────────────

    @Test
    void symbolTableMatchesTheModelVocabulary() {
        assertEquals(178, Symbols.count(), "the embedding table has 178 rows");
        assertEquals(0, Symbols.idOf('#'), "an unknown codepoint reads as the blank");
        assertTrue(Symbols.idOf(' ') > 0, "the word separator is a symbol of its own");
        assertTrue(Symbols.idOf('a') > 0, "letters are in the table");
    }

    @Test
    void tokensAreBlankInterspersed() {
        int[] tokens = Symbols.toTokens("ab");
        // blank, a, blank, b, blank
        assertEquals(5, tokens.length);
        assertEquals(0, tokens[0]);
        assertEquals(Symbols.idOf('a'), tokens[1]);
        assertEquals(0, tokens[2]);
        assertEquals(Symbols.idOf('b'), tokens[3]);
        assertEquals(0, tokens[4]);
    }

    @Test
    void normalizerSpeaksNumbersAndAbbreviations() {
        assertAll(
                () -> assertEquals("three", TextNormalizer.normalize("3")),
                () -> assertEquals("twenty one", TextNormalizer.normalize("21")),
                () -> assertEquals("one thousand nine", TextNormalizer.normalize("1009")),
                () -> assertEquals("third", TextNormalizer.normalize("3rd")),
                () ->
                        assertEquals(
                                "five dollars and fifty cents", TextNormalizer.normalize("$5.50")),
                () -> assertEquals("doctor Smith", TextNormalizer.normalize("Dr. Smith")),
                () -> assertEquals("ay bee see", TextNormalizer.normalize("ABC")),
                () -> assertEquals("pie torch", TextNormalizer.normalize("PyTorch")));
    }

    @Test
    void normalizerSpellsNumbersTooBigForAnInt() {
        // these used to throw NumberFormatException and fail the whole utterance
        assertEquals(
                "two zero one two three four five six seven eight nine",
                TextNormalizer.normalize("20123456789"));
        assertTrue(
                TextNormalizer.normalize("Call 20123456789 now.").startsWith("Call two zero one"));
        assertTrue(
                TextNormalizer.normalize("$99999999999.99")
                        .endsWith("dollars and ninety nine cents"));
    }

    @Test
    void normalizerSaysAcronymsAndDottedAcronymsDifferently() {
        // a dotted acronym keeps bare letters (the phonemizer names a lone letter itself)
        assertEquals("U S A", TextNormalizer.normalize("U.S.A."));
        assertEquals("ay bee see", TextNormalizer.normalize("ABC"));
    }

    @Test
    void normalizerAppliesUserOverridesFirst() {
        assertEquals("pie torch two", TextNormalizer.normalize("PyTorch 2", Map.of()).trim());
        assertEquals("torchy", TextNormalizer.normalize("PyTorch", Map.of("PyTorch", "torchy")));
    }

    @Test
    void textSplitsIntoSentences() {
        assertEquals(
                List.of("Hello world.", "How are you?"),
                InflectTTS.split("Hello world.  How are you?"));
        // The model renders a chunk's last phoneme only when a mark closes it: unterminated,
        // the waveform stops at full level mid-word ("one two three four five" loses "five",
        // "Hello, GraalVM" loses the "em"). Text that ends without punctuation is ordinary,
        // so the frontend supplies the stop rather than clipping the word.
        assertEquals(List.of("No punctuation here."), InflectTTS.split("No punctuation here"));
        assertEquals(List.of("Already asked?"), InflectTTS.split("Already asked?"));
        assertEquals(List.of("\"Quoted.\""), InflectTTS.split("\"Quoted.\""));
    }

    @Test
    void everyChunkOfALongSentenceCanBeFinished() {
        // the same clipping hits each interior join, where it sounds like a swallowed word
        String sentence = "word ".repeat(120).trim();
        List<String> chunks = InflectTTS.split(sentence);
        assertTrue(chunks.size() > 1, "a 600-char sentence must be split");
        for (int i = 0; i < chunks.size(); i++) {
            String chunk = chunks.get(i);
            assertTrue(chunk.length() <= 280, "chunk over the model limit: " + chunk.length());
            // a full stop only at the end; an interior cut is mid-sentence and rests on a comma
            assertEquals(i == chunks.size() - 1 ? '.' : ',', chunk.charAt(chunk.length() - 1));
        }
    }

    @Test
    void anAbbreviationDoesNotEndASentence() {
        // the utterance is normalized before it is chunked: "Dr." is "doctor" by the time the
        // sentence split runs, so the title and the name stay in one breath
        assertEquals(
                List.of("doctor Smith is here.", "Fine."),
                InflectTTS.chunks("Dr. Smith is here. Fine.", Map.of()));
    }

    @Test
    void overlongSentenceBreaksAtAClause() {
        String sentence = "word ".repeat(40) + ", and " + "more ".repeat(40) + ".";
        List<String> chunks = InflectTTS.split(sentence);
        assertTrue(chunks.size() > 1, "a 400-char sentence must be split");
        for (String chunk : chunks)
            assertTrue(chunk.length() <= 280, "chunk over the model limit: " + chunk.length());
    }

    // ── model ─────────────────────────────────────────────────────────────

    /** "həloʊ wɜːld", blank-interspersed. */
    private static final int[] HELLO = Symbols.toTokens("həloʊ wɜːld");

    private static Inflect2 model() throws IOException {
        return Inflect2.load(TestModels.require(REF), Arena.ofAuto());
    }

    /** One state per call: these tests are about the model, not about state reuse. */
    private static Media.Audio synthesize(
            Inflect2 model, int[] tokens, float lengthScale, float variation, long seed) {
        try (Inflect2.State state = model.newState()) {
            return model.synthesize(state, tokens, lengthScale, variation, seed);
        }
    }

    @Test
    void synthesizesPlausibleAudio() throws IOException {
        Inflect2 model = model();
        var audio = synthesize(model, HELLO, 1f, 0f, 1234);
        assertEquals(model.sampleRate(), audio.sampleRate());
        assertEquals(1, audio.channels());
        float[] pcm = audio.pcm();
        assertTrue(pcm.length > model.sampleRate() / 10, "at least 100ms of audio: " + pcm.length);
        float peak = 0;
        for (float sample : pcm) {
            assertTrue(Float.isFinite(sample), "non-finite sample");
            peak = Math.max(peak, Math.abs(sample));
        }
        assertTrue(peak > 0.01f && peak <= 1f, "peak out of range: " + peak);
    }

    @Test
    void lengthScaleStretchesTheWaveform() throws IOException {
        Inflect2 model = model();
        int fast = synthesize(model, HELLO, 1f, 0f, 1234).pcm().length;
        int slow = synthesize(model, HELLO, 1.5f, 0f, 1234).pcm().length;
        assertTrue(slow > fast, "lengthScale 1.5 must produce more samples than 1.0");
    }

    /**
     * Same tokens and seed twice. A tolerance, not equality, because the FIRST call in a JVM runs
     * partly interpreted and the Vector API accumulates differently there than once compiled -
     * measured convergence is at pass 2, and the divergence is ~1 LSB. Warm both passes and this
     * would be bit-exact.
     */
    @Test
    void repeatedSynthesisAgrees() throws IOException {
        Inflect2 model = model();
        float[] first = synthesize(model, HELLO, 1f, 0f, 1234).pcm();
        float[] second = synthesize(model, HELLO, 1f, 0f, 1234).pcm();
        assertEquals(first.length, second.length);
        double worst = 0;
        for (int i = 0; i < first.length; i++)
            worst = Math.max(worst, Math.abs(first[i] - second[i]));
        assertTrue(worst < 1e-3, "same seed diverged by " + worst);
    }

    @Test
    void seedChangesTheNoiseRealization() throws IOException {
        Inflect2 model = model();
        float[] one = synthesize(model, HELLO, 1f, 0.667f, 1).pcm();
        float[] two = synthesize(model, HELLO, 1f, 0.667f, 2).pcm();
        boolean differs = false;
        for (int i = 0; i < Math.min(one.length, two.length) && !differs; i++)
            differs = Math.abs(one[i] - two[i]) > 1e-3;
        assertTrue(differs, "a different seed must give different audio");
    }

    @Test
    void rejectsUnusableArguments() throws IOException {
        Inflect2 model = model();
        assertAll(
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> synthesize(model, new int[0], 1f, 0f, 1),
                                "empty tokens"),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> synthesize(model, new int[] {999}, 1f, 0f, 1),
                                "token outside the vocabulary"),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> synthesize(model, HELLO, 0f, 0f, 1),
                                "zero lengthScale"),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> synthesize(model, HELLO, Float.NaN, 0f, 1),
                                "NaN lengthScale"),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> synthesize(model, HELLO, 1f, -1f, 1),
                                "negative variation"),
                () ->
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> synthesize(model, HELLO, 1f, 1.01f, 1),
                                "variation above one"));
    }

    @Test
    @Tag("integration")
    void aRunOfIdentifiersIsSpokenNotRefused() throws IOException {
        // 277 raw characters, 1174 once every digit is a word: chunked on the raw text the
        // whole run was one chunk and the model refused it over the frame ceiling, with no
        // audio at all for the sentences before it
        InflectTTS tts = InflectTTS.load(TestModels.require(REF), Arena.ofAuto());
        try (Inflect2.State state = tts.newState()) {
            String text = "Your reference number is " + "4829173650 ".repeat(23);
            Media.Audio audio = tts.speak(state, text.strip(), SpeechOptions.NONE);
            assertTrue(audio.pcm().length > 24000 * 30, "half a minute of digits at least");
        }
    }
}
