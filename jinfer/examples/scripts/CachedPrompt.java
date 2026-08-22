///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --release 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES Models.java

// A long system prompt is prefilled once and its KV restored per request. Each response reports
// the restored token count, so the saving is visible without a misleading wall-clock benchmark.
//   jbang CachedPrompt.java
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferTokenUsage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import java.util.List;

public class CachedPrompt {

    // Pretend this is your policy document, few-shot block or tool catalogue.
    static final String SYSTEM = ("You are a terse assistant for a coffee company. "
            + "Answer in one short sentence. Never invent prices. ").repeat(40);

    static final List<String> QUESTIONS =
            List.of("Do you sell decaf?", "What is a flat white?", "Is arabica bitter?");

    public static void main(String[] args) {
        try (var base = JinferChatModel.builder()
                .model(Models.chat(args, 0))
                .maxOutputTokens(48)
                .build()) {
            var cached = base.withCachedPrompt(List.of(SystemMessage.from(SYSTEM)), List.of());
            for (String question : QUESTIONS) {
                var response = cached.chat(UserMessage.from(question));
                var usage = (JinferTokenUsage) response.tokenUsage();
                System.out.printf(
                        "%s%n  %s%n  restored %,d of %,d prompt tokens%n%n",
                        question,
                        response.aiMessage().text(),
                        usage.cachedInputTokens(),
                        usage.inputTokenCount());
            }
        }
    }
}
