package com.qxotic.jinfer.example.judgeadvisor;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.JinferChatOptions;
import java.util.List;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

/**
 * Remote genius + local judge: Kimi (OpenAI-compatible) generates, an in-JVM GGUF judges. The
 * judge's rubric is prefilled ONCE via jinfer's cached prompts (every evaluation then pays only the
 * question+answer tokens), and its verdict is grammar-pinned: valid JSON, rating always 1-4. The
 * weather tool deliberately returns -255C (below absolute zero) on alternating calls, so the judge
 * must fail at least the first answer. Run: {@code KIMI_API_KEY=... mvn spring-boot:run}.
 */
@SpringBootApplication
public class JudgeAdvisorApplication {

    /** No format instructions, no schema boilerplate - the grammar enforces structure. */
    static final String RUBRIC =
            "You are a strict evaluation judge. Given a user question and an assistant's answer,"
                    + " rate how well the answer addresses the question, 1 to 4:\n"
                    + "1: terrible - irrelevant, very partial, or contains physically impossible or"
                    + " factually wrong claims.\n"
                    + "2: mostly not helpful - misses key aspects.\n"
                    + "3: mostly helpful - solid but improvable.\n"
                    + "4: excellent - relevant, direct, complete.\n"
                    + "An answer that honestly states its data limitations (e.g. no real-time"
                    + " source) is acceptable: rate it 3 or 4 when it is direct and clear.\n"
                    + "Return your verdict: rating (1-4), evaluation (your rationale, at most two"
                    + " sentences), feedback (specific guidance for the retry, at most two"
                    + " sentences; empty string when rating 4).";

    /** Deterministic judge, schema pinned by grammar; maxTokens needs headroom (see README). */
    private static final JinferChatOptions JUDGE_OPTIONS =
            JinferChatOptions.builder()
                    .temperature(0.0)
                    .maxTokens(384)
                    .outputSchema(GrammarPinnedEvaluationAdvisor.VERDICT_SCHEMA)
                    .build();

    public static void main(String[] args) {
        SpringApplication.run(JudgeAdvisorApplication.class, args);
    }

    @Bean
    CommandLineRunner demo(OpenAiChatModel generator, JinferChatModel judgeBase) {
        return args -> run(generator, judgeBase);
    }

    /** The full hybrid loop: remote generator + local grammar-pinned judge with cached rubric. */
    static String run(ChatModel generator, JinferChatModel judgeBase) {
        JinferChatModel judge =
                judgeBase.withCachedPrompt(List.of(new SystemMessage(RUBRIC)), List.of());
        ChatClient client =
                ChatClient.builder(generator)
                        .defaultTools(new WeatherTools())
                        .defaultAdvisors(
                                GrammarPinnedEvaluationAdvisor.builder()
                                        .judge(judge, JUDGE_OPTIONS)
                                        .maxRepeatAttempts(4)
                                        .successRating(3)
                                        .build())
                        .build();
        String answer = client.prompt("What is the weather in Paris?").call().content().strip();
        System.out.println(">>> FINAL: " + answer);
        return answer;
    }
}
