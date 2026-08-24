///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-qwen3
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Rank a small document set by embedding similarity.
//   jbang Search.java "how do I make coffee?"
import com.qxotic.jinfer.langchain4j.JinferEmbeddingModel;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Search {

    private static final String DEFAULT_MODEL =
            "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0";

    private static final List<String> DOCUMENTS = List.of(
            "Grind the beans, then pour water just off the boil over them.",
            "The cat sat on the mat and refused to move all afternoon.",
            "Espresso needs about nine bars of pressure and a fine grind.",
            "Compile Java with javac, then run the class with java.",
            "Green tea should steep below boiling or it turns bitter.");

    public static void main(String[] args) {
        String query = args.length > 0 ? args[0] : "how do I make coffee?";
        String modelRef = args.length > 1 ? args[1] : DEFAULT_MODEL;

        try (var embedder = JinferEmbeddingModel.builder().model(modelRef).build()) {
            var vectors = embedder
                    .embedAll(DOCUMENTS.stream().map(TextSegment::from).toList())
                    .content();
            var queryVector = embedder.embed(query).content();

            record Hit(double score, String text) {}

            var hits = new ArrayList<Hit>();
            for (int i = 0; i < DOCUMENTS.size(); i++) {
                hits.add(new Hit(
                        CosineSimilarity.between(queryVector, vectors.get(i)), DOCUMENTS.get(i)));
            }
            hits.sort(Comparator.comparingDouble(Hit::score).reversed());

            System.out.printf("Query: %s%n%n", query);
            hits.forEach(hit -> System.out.printf("%.3f  %s%n", hit.score(), hit.text()));
        }
    }
}
