package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/** Builder validation, no model needed. */
class JinferDocumentPostProcessorTest {

    @Test
    void aNegativeTopKIsRefusedRatherThanReadAsKeepAll() {
        // 0 is the documented "keep all" sentinel; a negative is a caller bug, not a synonym
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferDocumentPostProcessor.builder().topK(-1));
        assertDoesNotThrow(() -> JinferDocumentPostProcessor.builder().topK(0));
    }
}
