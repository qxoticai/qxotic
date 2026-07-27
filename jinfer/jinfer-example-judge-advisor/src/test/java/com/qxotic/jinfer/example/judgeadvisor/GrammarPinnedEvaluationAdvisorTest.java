package com.qxotic.jinfer.example.judgeadvisor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

/** The advisor loop with stub models: no GGUF, no API key. */
class GrammarPinnedEvaluationAdvisorTest {

    /** A judge that returns canned verdict JSON, recording what it was asked. */
    private static final class StubJudge implements ChatModel {
        final Queue<String> verdicts = new ArrayDeque<>();
        final List<String> asked = new ArrayList<>();

        @Override
        public ChatResponse call(Prompt prompt) {
            asked.add(prompt.getContents());
            String next = verdicts.remove();
            // "LENGTH:<text>" simulates a budget-truncated verdict
            if (next.startsWith("LENGTH:")) {
                return new ChatResponse(
                        List.of(
                                new Generation(
                                        AssistantMessage.builder()
                                                .content(next.substring(7))
                                                .build(),
                                        org.springframework.ai.chat.metadata.ChatGenerationMetadata
                                                .builder()
                                                .finishReason("length")
                                                .build())));
            }
            return text(next);
        }

        @Override
        public ChatOptions getOptions() {
            return ChatOptions.builder().build();
        }
    }

    /** The generator side of the chain: canned answers, recording every request. */
    private static final class StubChain implements CallAdvisorChain {
        final Queue<ChatClientResponse> responses;
        final List<ChatClientRequest> seen = new ArrayList<>();

        StubChain(Queue<ChatClientResponse> responses) {
            this.responses = responses;
        }

        @Override
        public ChatClientResponse nextCall(ChatClientRequest request) {
            seen.add(request);
            return responses.remove();
        }

        @Override
        public List<CallAdvisor> getCallAdvisors() {
            return List.of();
        }

        @Override
        public CallAdvisorChain copy(CallAdvisor after) {
            return this;
        }
    }

    private static ChatResponse text(String content) {
        return new ChatResponse(
                List.of(new Generation(AssistantMessage.builder().content(content).build())));
    }

    private static ChatClientResponse response(String content) {
        return ChatClientResponse.builder().chatResponse(text(content)).build();
    }

    private static ChatClientResponse toolCallResponse() {
        AssistantMessage.ToolCall call =
                new AssistantMessage.ToolCall("c1", "function", "weather", "{}");
        return ChatClientResponse.builder()
                .chatResponse(
                        new ChatResponse(
                                List.of(
                                        new Generation(
                                                AssistantMessage.builder()
                                                        .content("")
                                                        .toolCalls(List.of(call))
                                                        .build()))))
                .build();
    }

    private static GrammarPinnedEvaluationAdvisor advisor(StubJudge judge, int maxAttempts) {
        return GrammarPinnedEvaluationAdvisor.builder()
                .judge(judge, ChatOptions.builder().build())
                .maxRepeatAttempts(maxAttempts)
                .successRating(3)
                .build();
    }

    private static ChatClientRequest request() {
        return ChatClientRequest.builder()
                .prompt(new Prompt("What is the weather in Paris?"))
                .build();
    }

    private static String verdict(int rating, String feedback) {
        return "{\"rating\":" + rating + ",\"evaluation\":\"e\",\"feedback\":\"" + feedback + "\"}";
    }

    @Test
    void passesOnFirstGoodVerdict() {
        StubJudge judge = new StubJudge();
        judge.verdicts.add(verdict(4, ""));
        StubChain chain = new StubChain(new ArrayDeque<>(List.of(response("sunny, 15C"))));
        ChatClientResponse out = advisor(judge, 3).adviseCall(request(), chain);
        assertEquals("sunny, 15C", out.chatResponse().getResult().getOutput().getText());
        assertEquals(1, chain.seen.size());
        assertEquals(1, judge.asked.size());
    }

    @Test
    void failedVerdictRetriesWithFeedbackInTheUserMessage() {
        StubJudge judge = new StubJudge();
        judge.verdicts.add(verdict(2, "-255C is physically impossible"));
        judge.verdicts.add(verdict(4, ""));
        StubChain chain =
                new StubChain(
                        new ArrayDeque<>(List.of(response("it is -255C"), response("it is 15C"))));
        ChatClientResponse out = advisor(judge, 3).adviseCall(request(), chain);
        assertEquals("it is 15C", out.chatResponse().getResult().getOutput().getText());
        assertEquals(2, chain.seen.size());
        String retryText = chain.seen.get(1).prompt().getUserMessage().getText();
        assertTrue(
                retryText.contains("-255C is physically impossible"),
                "feedback must ride into the retry: " + retryText);
    }

    @Test
    void maxAttemptsReturnsTheLastResponse() {
        StubJudge judge = new StubJudge();
        judge.verdicts.add(verdict(1, "bad"));
        judge.verdicts.add(verdict(2, "still bad"));
        StubChain chain =
                new StubChain(new ArrayDeque<>(List.of(response("one"), response("two"))));
        ChatClientResponse out = advisor(judge, 1).adviseCall(request(), chain);
        assertEquals("two", out.chatResponse().getResult().getOutput().getText());
        assertEquals(2, chain.seen.size()); // 1 attempt + 1 retry, no more
    }

    @Test
    void toolCallResponsesSkipEvaluation() {
        StubJudge judge = new StubJudge(); // no verdicts queued: any judge call would throw
        StubChain chain = new StubChain(new ArrayDeque<>(List.of(toolCallResponse())));
        ChatClientResponse out = advisor(judge, 3).adviseCall(request(), chain);
        assertTrue(out.chatResponse().hasToolCalls());
        assertEquals(0, judge.asked.size());
    }

    @Test
    void outOfRangeRatingNeverPasses() {
        StubJudge judge = new StubJudge();
        judge.verdicts.add(verdict(9, "confused judge")); // a 9 would pass a naive >= check
        judge.verdicts.add(verdict(3, "ok"));
        StubChain chain =
                new StubChain(new ArrayDeque<>(List.of(response("one"), response("two"))));
        ChatClientResponse out = advisor(judge, 3).adviseCall(request(), chain);
        assertEquals("two", out.chatResponse().getResult().getOutput().getText());
        assertEquals(2, chain.seen.size());
    }

    @Test
    void truncatedVerdictDegradesToFailedEvaluation() {
        StubJudge judge = new StubJudge();
        judge.verdicts.add("LENGTH:{\"rating\": 1, \"evaluation\": \"cut off mid-str"); // invalid
        judge.verdicts.add(verdict(4, ""));
        StubChain chain =
                new StubChain(new ArrayDeque<>(List.of(response("one"), response("two"))));
        // the budget-cut verdict must NOT crash the loop: it degrades to a failed evaluation
        ChatClientResponse out = advisor(judge, 3).adviseCall(request(), chain);
        assertEquals("two", out.chatResponse().getResult().getOutput().getText());
        assertEquals(2, chain.seen.size());
        String retryText = chain.seen.get(1).prompt().getUserMessage().getText();
        assertTrue(retryText.contains("Be more concise"), retryText);
    }
}
