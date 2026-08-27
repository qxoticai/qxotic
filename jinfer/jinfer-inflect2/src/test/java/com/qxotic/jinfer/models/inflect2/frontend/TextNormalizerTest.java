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
}
