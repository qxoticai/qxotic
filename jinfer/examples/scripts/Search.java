///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --release 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-qwen3
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
//SOURCES Models.java

// Semantic search with no vector database: embed once, cosine-rank in memory.
//   jbang Search.java "how do I make coffee?"
import com.qxotic.jinfer.langchain4j.JinferEmbeddingModel;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.CosineSimilarity;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Search {

    static final List<String> DOCS = List.of(
            "Grind the beans, then pour water just off the boil over them.",
            "The cat sat on the mat and refused to move all afternoon.",
            "Espresso needs about nine bars of pressure and a fine grind.",
            "Compile Java with javac, then run the class with java.",
            "Green tea should steep below boiling or it turns bitter.");

    public static void main(String[] args) {
        var query = args.length > 0 ? args[0] : "how do I make coffee?";
        try (var embedder = JinferEmbeddingModel.builder().model(Models.embed(args, 1)).build()) {
            var docs = embedder.embedAll(DOCS.stream().map(TextSegment::from).toList()).content();
            var q = embedder.embed(query).content();

            record Hit(double score, String text) {}
            var hits = new ArrayList<Hit>();
            for (int i = 0; i < DOCS.size(); i++)
                hits.add(new Hit(CosineSimilarity.between(q, docs.get(i)), DOCS.get(i)));
            hits.sort(Comparator.comparingDouble(Hit::score).reversed());

            System.out.println("query: " + query + "\n");
            hits.forEach(h -> System.out.printf("  %.3f  %s%n", h.score(), h.text()));
        }
    }
}
