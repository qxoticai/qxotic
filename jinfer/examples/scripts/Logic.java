///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-qwen35
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
//DEPS org.slf4j:slf4j-nop:2.0.18

// Solve truth-teller puzzles and score the constrained answers directly.
//   jbang Logic.java
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferChatRequestParameters;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;

import java.util.List;

public class Logic {

    private static final String DEFAULT_MODEL = "unsloth/Qwen3.5-4B-GGUF:Q8_0"; // reasons before it answers

    private static final String ANSWER_GRAMMAR = """
            root ::= word "," ws word "," ws word
            ws   ::= " "*
            word ::= "yes" | "no"
            """;

    private record Puzzle(String name, String prompt, String answer) {}

    private static final List<Puzzle> PUZZLES = List.of(
            new Puzzle(
                    "village",
                    "In a village, every inhabitant is either a knight (always tells the truth) or "
                            + "a knave (always lies). Mira is a knight. Anna says: 'Mira lies.' "
                            + "Ben says: 'Anna tells the truth.' Dan says: 'Ben lies.' "
                            + "Is Anna a knight? Is Ben a knight? Is Dan a knight? "
                            + "Think step by step, then answer with three words, yes or no.",
                    "no, no, yes"),
            new Puzzle(
                    "harbor",
                    "On an island, every inhabitant is either a knight (always tells the truth) or "
                            + "a knave (always lies). Zara is a knave. Leo says: 'Zara tells the "
                            + "truth.' Mia says: 'Leo lies.' Nico says: 'Mia tells the truth.' Oli "
                            + "says: 'Nico lies.' "
                            + "Is Leo a knight? Is Mia a knight? Is Oli a knight? "
                            + "Think step by step, then answer with three words, yes or no.",
                    "no, yes, no"));

    public static void main(String[] args) {
        String modelRef = args.length > 0 ? args[0] : DEFAULT_MODEL;

        int correct = 0;
        try (var model = JinferChatModel.builder()
                .model(modelRef)
                .maxOutputTokens(1024)
                .thinking(true)
                .reasoningBudget(768) // room to reason before the grammar takes the answer
                .temperature(0.0) // deterministic: the same puzzles score the same every run
                .seed(42L)
                .build()) {
            for (Puzzle puzzle : PUZZLES) {
                String reply = ask(model, puzzle);
                boolean right = reply.replace(" ", "").equals(puzzle.answer().replace(" ", ""));
                correct += right ? 1 : 0;
                System.out.printf("%s  %-8s  %-12s  expected %s%n",
                        right ? "PASS" : "FAIL", puzzle.name(), reply, puzzle.answer());
            }
        }
        System.out.printf("%n%d/%d puzzles solved%n", correct, PUZZLES.size());
    }

    private static String ask(JinferChatModel model, Puzzle puzzle) {
        var request = ChatRequest.builder()
                .messages(UserMessage.from(puzzle.prompt()))
                .parameters(JinferChatRequestParameters.builder()
                        .grammar(ANSWER_GRAMMAR)
                        .build())
                .build();
        return model.chat(request).aiMessage().text().trim();
    }
}
