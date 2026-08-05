package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import org.junit.jupiter.api.Test;

/** The SPI is consulted: a service on the test classpath must receive the builder. */
final class TokenizerContributionTest {

    static volatile boolean contributed;

    public static final class TestContribution implements TokenizerContribution {
        @Override
        public void contribute(GGUFTokenizerLoader.Builder builder) {
            contributed = true;
        }
    }

    @Test
    void contributionsAreConsulted() {
        contributed = false;
        Tokenizers.builder();
        assertTrue(contributed, "TokenizerContribution service was not consulted");
    }
}
