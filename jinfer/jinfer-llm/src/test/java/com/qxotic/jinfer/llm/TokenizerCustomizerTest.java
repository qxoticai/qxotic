package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import org.junit.jupiter.api.Test;

/** The SPI is consulted: a service on the test classpath must receive the builder. */
final class TokenizerCustomizerTest {

    static volatile boolean customized;

    public static final class TestCustomizer implements TokenizerCustomizer {
        @Override
        public void customize(GGUFTokenizerLoader.Builder builder) {
            customized = true;
        }
    }

    @Test
    void customizersAreConsulted() {
        customized = false;
        Tokenizers.builder();
        assertTrue(customized, "TokenizerCustomizer service was not consulted");
    }
}
