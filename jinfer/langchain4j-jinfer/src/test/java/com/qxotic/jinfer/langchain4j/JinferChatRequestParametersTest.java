package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import org.junit.jupiter.api.Test;

/**
 * The merge matrix for {@link JinferChatRequestParameters} - LOAD-BEARING, not style: {@code
 * ChatModel.chat} merges {@code defaults.overrideWith(request)} before {@code doChat}, so any
 * combination that loses the jinfer fields silently disables the feature. Every quadrant of
 * (default|jinfer) x (default|jinfer) is pinned, plus field-level override precedence.
 */
class JinferChatRequestParametersTest {

    private static final String YES_NO = "root ::= \"yes\" | \"no\"";

    @Test
    void builderRoundTrip() {
        JinferChatRequestParameters p =
                JinferChatRequestParameters.builder()
                        .grammar(YES_NO)
                        .seed(42L)
                        .temperature(0.7)
                        .maxOutputTokens(64)
                        .build();
        assertEquals(YES_NO, p.grammar());
        assertEquals(42L, p.seed());
        assertEquals(0.7, p.temperature());
        assertEquals(64, p.maxOutputTokens());
    }

    @Test
    void jinferDefaultsSurviveADefaultTypedRequest() {
        // the AiServices path: the model holds jinfer defaults, requests carry plain parameters
        JinferChatRequestParameters defaults =
                JinferChatRequestParameters.builder()
                        .grammar(YES_NO)
                        .seed(7L)
                        .temperature(0.0)
                        .build();
        ChatRequestParameters request =
                DefaultChatRequestParameters.builder().temperature(0.9).build();
        ChatRequestParameters merged = defaults.overrideWith(request);
        JinferChatRequestParameters j = (JinferChatRequestParameters) merged;
        assertEquals(YES_NO, j.grammar(), "jinfer fields must survive the merge");
        assertEquals(7L, j.seed());
        assertEquals(0.9, j.temperature(), "request-side standard fields must win");
    }

    @Test
    void jinferRequestFieldsSurviveJinferDefaults() {
        JinferChatRequestParameters defaults =
                JinferChatRequestParameters.builder().seed(1L).temperature(0.0).build();
        JinferChatRequestParameters request =
                JinferChatRequestParameters.builder().grammar(YES_NO).seed(2L).build();
        JinferChatRequestParameters merged =
                (JinferChatRequestParameters) defaults.overrideWith(request);
        assertEquals(2L, merged.seed(), "request seed wins");
        assertEquals(YES_NO, merged.grammar(), "request grammar arrives");
        assertEquals(0.0, merged.temperature(), "unset request fields inherit defaults");
    }

    @Test
    void nullRequestFieldsInheritJinferDefaults() {
        JinferChatRequestParameters defaults =
                JinferChatRequestParameters.builder().grammar(YES_NO).seed(9L).build();
        JinferChatRequestParameters request = JinferChatRequestParameters.builder().build();
        JinferChatRequestParameters merged =
                (JinferChatRequestParameters) defaults.overrideWith(request);
        assertEquals(YES_NO, merged.grammar());
        assertEquals(9L, merged.seed());
    }

    @Test
    void jinferBuilderCopiesStandardFieldsFromDefaultTyped() {
        ChatRequestParameters plain =
                DefaultChatRequestParameters.builder().temperature(0.3).maxOutputTokens(10).build();
        JinferChatRequestParameters j =
                JinferChatRequestParameters.builder().overrideWith(plain).build();
        assertEquals(0.3, j.temperature());
        assertEquals(10, j.maxOutputTokens());
        assertNull(j.grammar());
        assertNull(j.seed());
    }

    @Test
    void requestGrammarReplacesDefaultsGrammar() {
        JinferChatRequestParameters defaults =
                JinferChatRequestParameters.builder().grammar(YES_NO).build();
        JinferChatRequestParameters request =
                JinferChatRequestParameters.builder().grammar("root ::= \"ok\"").build();
        JinferChatRequestParameters merged =
                (JinferChatRequestParameters) defaults.overrideWith(request);
        assertEquals("root ::= \"ok\"", merged.grammar(), "request grammar must win");
    }

    @Test
    void equalsCoversTheJinferFields() {
        JinferChatRequestParameters a =
                JinferChatRequestParameters.builder().grammar(YES_NO).seed(1L).build();
        JinferChatRequestParameters same =
                JinferChatRequestParameters.builder().grammar(YES_NO).seed(1L).build();
        JinferChatRequestParameters otherSeed =
                JinferChatRequestParameters.builder().grammar(YES_NO).seed(2L).build();
        JinferChatRequestParameters otherGrammar =
                JinferChatRequestParameters.builder().grammar("root ::= \"ok\"").seed(1L).build();
        assertEquals(a, same);
        assertEquals(a.hashCode(), same.hashCode());
        assertNotEquals(a, otherSeed);
        assertNotEquals(a, otherGrammar);
    }
}
