///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-llama
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Solve truth-teller puzzles and score the constrained answers directly.
//   jbang Logic.java
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferChatRequestParameters;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;

import java.util.List;

public class Logic {

    private static final String DEFAULT_MODEL =
            "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0";

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
                .maxOutputTokens(128)
                .thinking(true)
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
