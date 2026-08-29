///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-llama
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Prefill a system prompt once and report how many tokens each request restores.
//   jbang CachedPrompt.java
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferTokenUsage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;

import java.util.List;

public class CachedPrompt {

    private static final String DEFAULT_MODEL =
            "unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0";

    private static final String SYSTEM = ("You are a terse assistant for a coffee company. "
            + "Answer in one short sentence. Never invent prices. ").repeat(40);

    private static final List<String> QUESTIONS =
            List.of("Do you sell decaf?", "What is a flat white?", "Is arabica bitter?");

    public static void main(String[] args) {
        String modelRef = args.length > 0 ? args[0] : DEFAULT_MODEL;

        try (var base = JinferChatModel.builder()
                .model(modelRef)
                .maxOutputTokens(48)
                .build()) {
            var cached = base.withCachedPrompt(List.of(SystemMessage.from(SYSTEM)), List.of());
            for (String question : QUESTIONS) {
                var response = cached.chat(UserMessage.from(question));
                var usage = (JinferTokenUsage) response.tokenUsage();
                System.out.printf(
                        "Question: %s%nAnswer:   %s%nCache:    %,d of %,d prompt tokens restored%n%n",
                        question,
                        response.aiMessage().text(),
                        usage.cachedInputTokens(),
                        usage.inputTokenCount());
            }
        }
    }
}
