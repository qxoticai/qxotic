package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class TokenizerRegexPatternTest {

    private static final String R50K_PATTERN = TiktokenFixtures.encoding("r50k_base").pattern();
    private static final String CL100K_PATTERN = TiktokenFixtures.encoding("cl100k_base").pattern();
    private static final String O200K_PATTERN = TiktokenFixtures.encoding("o200k_base").pattern();

    @Test
    void r50kPatternCompiles() {
        assertDoesNotThrow(() -> Pattern.compile(R50K_PATTERN));
    }

    @Test
    void cl100kPatternCompiles() {
        assertDoesNotThrow(() -> Pattern.compile(CL100K_PATTERN));
    }

    @Test
    void o200kPatternCompiles() {
        assertDoesNotThrow(() -> Pattern.compile(O200K_PATTERN));
    }

    @Test
    void r50kPatternMatchesContractions() {
        Pattern pattern = Pattern.compile(R50K_PATTERN);
        String[] contractions = {"don't", "won't", "can't", "it's", "I'm", "you'll", "I'd"};
        for (String contraction : contractions) {
            assertTrue(pattern.matcher(contraction).find(), contraction);
        }
    }

    @Test
    void r50kPatternMatchesWordsAndNumbers() {
        Pattern pattern = Pattern.compile(R50K_PATTERN);
        String[] samples = {"hello", "world", "1", "12", "1234"};
        for (String sample : samples) {
            assertTrue(pattern.matcher(sample).find(), sample);
        }
    }

    @Test
    void cl100kPatternMatchesContractionsAndNumberChunks() {
        Pattern pattern = Pattern.compile(CL100K_PATTERN);
        String[] samples = {"don't", "you're", "1", "12", "123"};
        for (String sample : samples) {
            assertTrue(pattern.matcher(sample).find(), sample);
        }
    }

    @Test
    void o200kPatternMatchesCaseVariants() {
        Pattern pattern = Pattern.compile(O200K_PATTERN);
        String[] samples = {"Hello", "HELLO", "hello"};
        for (String sample : samples) {
            assertTrue(pattern.matcher(sample).find(), sample);
        }
    }
}
