///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES Models.java

// A long system prompt is prefilled ONCE and its KV restored per request, instead of being
// re-ingested every time. Output is byte-identical either way - this is a cost optimisation, not a
// behaviour change.
//
// Both passes run warm and answer the same questions; the only difference is whether the shared
// prefix is sent with each request or restored from the block tree.
//   jbang CachedPrompt.java
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import java.util.ArrayList;
import java.util.List;

public class CachedPrompt {

    // Pretend this is your policy document, few-shot block or tool catalogue.
    static final String SYSTEM = ("You are a terse assistant for a coffee company. "
            + "Answer in one short sentence. Never invent prices. ").repeat(40);

    static final List<String> QUESTIONS =
            List.of("Do you sell decaf?", "What is a flat white?", "Is arabica bitter?");

    public static void main(String[] args) throws Exception {
        try (var base = JinferChatModel.builder().modelPath(Models.chat(args, 0)).build()) {
            var cached = base.withCachedPrompt(List.of(SystemMessage.from(SYSTEM)), List.of());

            for (int i = 0; i < 2; i++) { plain(base); cached.chat(QUESTIONS.get(0)); } // JIT warmup

            var answers = new ArrayList<String>();
            long uncached = time(() -> plain(base));
            long withCache = time(() -> QUESTIONS.forEach(q -> answers.add(cached.chat(q))));

            System.out.println("answer (identical either way): " + answers.get(0));
            System.out.printf("%n%d questions over a %d-char system prompt:%n", QUESTIONS.size(), SYSTEM.length());
            System.out.printf("  prefix sent every time : %5d ms%n", uncached);
            System.out.printf("  prefix restored (cached): %5d ms   -> %.1fx%n",
                    withCache, uncached / (double) Math.max(withCache, 1));
        }
    }

    /** The prefix travels with every request - what you do without caching. */
    private static void plain(JinferChatModel model) {
        for (String q : QUESTIONS)
            model.chat(List.<ChatMessage>of(SystemMessage.from(SYSTEM), UserMessage.from(q)));
    }

    private static long time(Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        return (System.nanoTime() - t0) / 1_000_000;
    }
}
