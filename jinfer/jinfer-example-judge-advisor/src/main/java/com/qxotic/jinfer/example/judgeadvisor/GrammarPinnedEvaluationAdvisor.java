package com.qxotic.jinfer.example.judgeadvisor;

import com.qxotic.format.json.Json;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.CallAdvisor;
import org.springframework.ai.chat.client.advisor.api.CallAdvisorChain;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Self-refine loop (the spring-ai-examples evaluation-recursive-advisor pattern) with a LOCAL judge
 * whose verdict is pinned by grammar-constrained decoding: the schema ({@code rating} is {@code
 * enum [1,2,3,4]}) is compiled to a GBNF automaton that masks logits, so the verdict ALWAYS parses
 * and the rating is ALWAYS in range - the two failure modes that make the original demo fragile (a
 * parse error kills the loop; an out-of-range rating silently passes) are unrepresentable here. On
 * a failed verdict the feedback is appended to the user message and the chain re-runs, up to {@code
 * maxRepeatAttempts}.
 */
public final class GrammarPinnedEvaluationAdvisor implements CallAdvisor {

    /** The judge's verdict. */
    public record Verdict(int rating, String evaluation, String feedback) {}

    /** Hand-written (not record-generated) so {@code rating} is pinned to 1-4 in the grammar. */
    static final String VERDICT_SCHEMA =
            "{\"type\":\"object\",\"properties\":{"
                    + "\"rating\":{\"enum\":[1,2,3,4]},"
                    + "\"evaluation\":{\"type\":\"string\"},"
                    + "\"feedback\":{\"type\":\"string\"}},"
                    + "\"required\":[\"rating\",\"evaluation\",\"feedback\"]}";

    private final ChatModel judge;
    private final ChatOptions judgeOptions;
    private final int maxRepeatAttempts;
    private final int successRating;

    private GrammarPinnedEvaluationAdvisor(Builder b) {
        this.judge = b.judge;
        this.judgeOptions = b.judgeOptions;
        this.maxRepeatAttempts = b.maxRepeatAttempts;
        this.successRating = b.successRating;
    }

    @Override
    public ChatClientResponse adviseCall(ChatClientRequest request, CallAdvisorChain chain) {
        ChatClientRequest current = request;
        for (int attempt = 1; attempt <= maxRepeatAttempts + 1; attempt++) {
            ChatClientResponse response = chain.copy(this).nextCall(current);
            ChatResponse chatResponse = response.chatResponse();
            // tool-call turns are never judged - only the final text answer
            if (chatResponse == null || chatResponse.hasToolCalls()) {
                return response;
            }
            String answer = chatResponse.getResult().getOutput().getText();
            ChatResponse judgeResponse =
                    judge.call(
                            new Prompt(
                                    "Question: "
                                            + questionOf(current.prompt())
                                            + "\nAnswer: "
                                            + answer,
                                    judgeOptions));
            Verdict verdict = verdictOf(judgeResponse);
            // defensive only: with the grammar pin, out-of-range is unrepresentable
            boolean pass =
                    verdict.rating() >= 1
                            && verdict.rating() <= 4
                            && verdict.rating() >= successRating;
            System.out.printf(
                    ">>> judge: attempt %d %s%n",
                    attempt, transcript(verdict, judgeResponse, pass, answer));
            if (pass) {
                return response;
            }
            if (attempt > maxRepeatAttempts) {
                System.out.printf(
                        ">>> judge: max attempts (%d) reached, giving up%n", maxRepeatAttempts);
                return response;
            }
            current = addFeedback(current, verdict);
        }
        throw new IllegalStateException("unexpected loop exit");
    }

    private Verdict verdictOf(ChatResponse judgeResponse) {
        // a budget-cut verdict is invalid JSON even under the grammar (the pin guarantees no
        // invalid tokens, not completion within budget): degrade to a failed evaluation with
        // generic feedback instead of killing the loop
        if ("length".equals(judgeResponse.getResult().getMetadata().getFinishReason())) {
            return new Verdict(
                    1, "judge verdict truncated at the token budget", "Be more concise.");
        }
        String text = judgeResponse.getResult().getOutput().getText();
        Map<String, Object> json = Json.parseMap(text);
        return new Verdict(
                ((Number) json.get("rating")).intValue(),
                (String) json.get("evaluation"),
                (String) json.get("feedback"));
    }

    /** One transcript line per attempt: verdict, judge cost, and what was judged. */
    private static String transcript(
            Verdict verdict, ChatResponse judgeResponse, boolean pass, String answer) {
        var usage = judgeResponse.getMetadata().getUsage();
        StringBuilder line =
                new StringBuilder(pass ? "passed" : "failed")
                        .append(" (rating ")
                        .append(verdict.rating())
                        .append(", ")
                        .append(usage.getCompletionTokens())
                        .append(" tokens");
        if (usage.getCacheReadInputTokens() != null) {
            line.append(", cacheRead ").append(usage.getCacheReadInputTokens());
        }
        Object decode = judgeResponse.getMetadata().get("eval-duration");
        if (decode != null) {
            line.append(", decode ").append(decode);
        }
        line.append(")");
        if (!pass && !verdict.feedback().isEmpty()) {
            line.append(": ").append(verdict.feedback());
        }
        return line.append(" [answer: ").append(preview(answer)).append("]").toString();
    }

    private static String preview(String text) {
        String oneLine = text == null ? "" : text.replace('\n', ' ').strip();
        return oneLine.length() <= 80 ? oneLine : oneLine.substring(0, 80) + "...";
    }

    private static String questionOf(Prompt prompt) {
        List<String> parts = new ArrayList<>();
        for (Message m : prompt.getInstructions()) {
            if (m.getMessageType() == MessageType.SYSTEM
                    || m.getMessageType() == MessageType.USER) {
                parts.add(m.getMessageType() + ": " + m.getText());
            }
        }
        return String.join("\n", parts);
    }

    private static ChatClientRequest addFeedback(ChatClientRequest request, Verdict verdict) {
        Prompt augmented =
                request.prompt()
                        .augmentUserMessage(
                                u ->
                                        u.mutate()
                                                .text(
                                                        u.getText()
                                                                + "\n\nThe previous answer failed"
                                                                + " evaluation with feedback: "
                                                                + verdict.feedback()
                                                                + "\nPlease answer again,"
                                                                + " addressing the feedback.")
                                                .build());
        return request.mutate().prompt(augmented).build();
    }

    @Override
    public String getName() {
        return "GrammarPinnedEvaluationAdvisor";
    }

    @Override
    public int getOrder() {
        return 0;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private ChatModel judge;
        private ChatOptions judgeOptions;
        private int maxRepeatAttempts = 3;
        private int successRating = 3;

        private Builder() {}

        /**
         * The judge model plus its per-call options - for the pinned path these must carry the
         * verdict schema as {@code outputSchema} (jinfer enforces it token-level) and temperature
         * 0. Typically a cached-prompt view whose rubric was prefilled once.
         */
        public Builder judge(ChatModel judge, ChatOptions judgeOptions) {
            this.judge = judge;
            this.judgeOptions = judgeOptions;
            return this;
        }

        public Builder maxRepeatAttempts(int maxRepeatAttempts) {
            this.maxRepeatAttempts = maxRepeatAttempts;
            return this;
        }

        /** Verdicts with {@code rating >= successRating} pass (scale 1-4). */
        public Builder successRating(int successRating) {
            this.successRating = successRating;
            return this;
        }

        public GrammarPinnedEvaluationAdvisor build() {
            if (judge == null || judgeOptions == null) {
                throw new IllegalArgumentException("judge and judgeOptions are required");
            }
            return new GrammarPinnedEvaluationAdvisor(this);
        }
    }
}
