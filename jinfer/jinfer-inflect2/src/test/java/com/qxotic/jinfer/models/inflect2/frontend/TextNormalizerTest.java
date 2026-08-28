package com.qxotic.jinfer.models.inflect2.frontend;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class TextNormalizerTest {

    @ParameterizedTest(name = "{0} -> {1}")
    @CsvSource(
            delimiter = '|',
            value = {
                "1st | first",
                "22nd | twenty second",
                "103rd | one hundred third",
                "$1,000,000 | one million dollars",
                "$2,500,000 | two million five hundred thousand dollars",
            })
    void quantitiesSpeakTheirScale(String text, String spoken) {
        assertEquals(spoken, TextNormalizer.normalize(text));
    }

    @Test
    void millionsAreAScaleNotDigits() {
        // a bare 7+ digit run is an identifier (read out), but a known quantity of that size
        // used to fall off the end of numberWords and be read out too
        assertEquals("one million", TextNormalizer.spokenNumber("1000000"));
        assertEquals(
                "nine hundred ninety nine million nine hundred ninety nine thousand nine hundred"
                        + " ninety nine",
                TextNormalizer.spokenNumber("999999999"));
        assertEquals("one zero zero zero zero zero zero", TextNormalizer.normalize("1000000"));
    }

    @Test
    void anAbsurdOrdinalIsReadOutNotACrash() {
        // 10+ digits overflowed Integer.parseInt and threw out of speak()
        assertEquals(
                "one two three four five six seven eight nine zero one",
                TextNormalizer.normalize("12345678901st"));
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @CsvSource(
            delimiter = '|',
            value = {
                // \b needs a non-letter on both sides, so an acronym welded to a word was
                // invisible to the acronym rule and the compound reached the phonemizer whole:
                // "GraalVM" came out as "graal" with the VM swallowed into a schwa (espeak), or
                // into "lvm", a vowelless run that is silence with extra steps (letter-to-sound)
                "Hello, GraalVM | Hello, Graal vee em",
                "macOS | mac oh ess",
                "iOS | i oh ess",
                "OpenAI | Open ay eye",
                "HTTPServer | aitch tee tee pee Server",
            })
    void anAcronymWeldedToAWordIsStillAnAcronym(String text, String spoken) {
        assertEquals(spoken, TextNormalizer.normalize(text));
    }

    @ParameterizedTest
    @CsvSource({"VMware", "McDonald", "iPhone", "JavaScript", "Qwen"})
    void aCapitalInsideAWordIsNotAnAcronym(String text) {
        // The split is deliberately narrow: a run of two or more capitals, and for the leading
        // case only with two capitals already behind it, so "VMware" does not become "V Mware".
        // These are words a reader says whole, and the phonemizer already says them whole.
        assertEquals(text, TextNormalizer.normalize(text));
    }

    @Test
    void groupedNumbersAreQuantities() {
        // the comma grouping is the writer saying "quantity": never digit by digit, and a
        // grouped decimal keeps its integer part whole
        assertEquals(
                "About thirty thousand people came.",
                TextNormalizer.normalize("About 30,000 people came."));
        assertEquals(
                "It costs one thousand two hundred thirty four point five six dollars.",
                TextNormalizer.normalize("It costs 1,234.56 dollars."));
    }
}
