package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import java.nio.ByteBuffer;
import org.junit.jupiter.api.Test;

/**
 * The estimator's composition laws over a deterministic fake tokenizer (1 token per character):
 * degenerate tool-call arguments (null, blank, "{}") normalize instead of crashing, duplicates
 * count each, the reasoning lane is free, whitespace text is counted verbatim, and media on a
 * text-only estimator refuses BEFORE any decode. The OpenAI TokenCountEstimator edge cases, moved
 * to jinfer's message-shape boundary; exact real-vocabulary counts are pinned by {@code
 * JinferEmbeddingModelIT.tokenCountsAreExactOnText} against billed usage.
 */
final class EstimatorsTest {

    /** One token per character: expectations are computable by hand, nothing hides. */
    private static final Tokenizer CHARS =
            new Tokenizer() {
                @Override
                public Vocabulary vocabulary() {
                    throw new UnsupportedOperationException();
                }

                @Override
                public void encodeInto(
                        CharSequence text, int start, int end, IntSequence.Builder out) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public int countTokens(CharSequence text, int start, int end) {
                    return end - start;
                }

                @Override
                public int decodeBytesInto(
                        IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
                    throw new UnsupportedOperationException();
                }
            };

    /** Text-only, like the chat models without an mmproj companion. */
    private static final Estimators ESTIMATOR = new Estimators(CHARS, null, null);

    private static ToolExecutionRequest call(String arguments) {
        return ToolExecutionRequest.builder()
                .id("1")
                .name("get_weather")
                .arguments(arguments)
                .build();
    }

    @Test
    void emptyTextIsZeroTokens() {
        assertEquals(0, ESTIMATOR.estimateTokenCountInText(""));
    }

    @Test
    void whitespaceTextCountsVerbatim() {
        // the OpenAI newline edge cases: no whitespace collapsing, no crash, no free tokens
        assertEquals(1, ESTIMATOR.estimateTokenCountInText("\n"));
        assertEquals(1, ESTIMATOR.estimateTokenCountInText(" "));
        assertEquals(4, ESTIMATOR.estimateTokenCountInText("\n \n\n"));
    }

    @Test
    void toolCallWithNullArgumentsCountsNamePlusEmptyObject() {
        int estimate = ESTIMATOR.estimateTokenCountInMessage(AiMessage.from(call(null)));
        // name (11 chars) + the normalized empty arguments object "{}" (2 chars)
        assertEquals("get_weather".length() + 2, estimate);
    }

    @Test
    void degenerateArgumentSpellingsNormalize() {
        // null, blank and "{}" are the same call downstream - so they must cost the same
        int nul = ESTIMATOR.estimateTokenCountInMessage(AiMessage.from(call(null)));
        int blank = ESTIMATOR.estimateTokenCountInMessage(AiMessage.from(call("  ")));
        int emptyObject = ESTIMATOR.estimateTokenCountInMessage(AiMessage.from(call("{}")));
        assertEquals(nul, blank);
        assertEquals(nul, emptyObject);
    }

    @Test
    void duplicateToolCallsEachCounted() {
        int one = ESTIMATOR.estimateTokenCountInMessage(AiMessage.from(call(null)));
        AiMessage two =
                AiMessage.builder()
                        .toolExecutionRequests(
                                java.util.List.of(
                                        call(null),
                                        ToolExecutionRequest.builder()
                                                .id("2")
                                                .name("get_weather")
                                                .arguments(null)
                                                .build()))
                        .build();
        assertEquals(2 * one, ESTIMATOR.estimateTokenCountInMessage(two));
    }

    @Test
    void reasoningLaneIsNotCounted() {
        // thinking is not re-prompted by default, so it must not be billed to the context
        AiMessage withThinking =
                AiMessage.builder().thinking("a long internal monologue").text("hi").build();
        assertEquals(2, ESTIMATOR.estimateTokenCountInMessage(withThinking));
    }

    @Test
    void mediaOnATextOnlyEstimatorRefusesBeforeDecoding() {
        UnsupportedFeatureException e =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                ESTIMATOR.estimateTokenCountInMessage(
                                        UserMessage.from(
                                                ImageContent.from("http://localhost/x.png"))));
        assertTrue(e.getMessage().contains("cannot ingest media"), e.getMessage());
    }
}
