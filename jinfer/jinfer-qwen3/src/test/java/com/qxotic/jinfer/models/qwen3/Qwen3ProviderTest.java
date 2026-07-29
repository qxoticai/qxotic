package com.qxotic.jinfer.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Architecture dispatch for "qwen3": this port serves the retrieval family only, so a generative
 * load must be refused with the entry point that actually works - the mistake is easy to make,
 * since the generative Qwen 3.5 models sit one character away in jinfer-qwen35.
 *
 * <p>Header-only: the throw happens before any weight is mapped.
 */
@Tag("integration")
class Qwen3ProviderTest {

    @Test
    void generativeLoadIsRefusedWithTheRightAdvice() {
        UnsupportedOperationException e =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                Models.load(
                                        ModelFixture.QWEN3_RERANKER_06B_Q8.require(),
                                        512,
                                        Arena.ofAuto()));
        assertTrue(e.getMessage().contains("loadEmbedder"), e.getMessage());
        assertTrue(e.getMessage().contains("loadReranker"), e.getMessage());
    }
}
