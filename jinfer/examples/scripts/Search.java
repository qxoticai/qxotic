///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES Models.java

// Semantic search with no vector database: embed once, cosine-rank in memory.
//   jbang Search.java "how do I make coffee?"
import com.qxotic.jinfer.langchain4j.JinferEmbeddingModel;
import dev.langchain4j.data.segment.TextSegment;
import java.util.*;

public class Search {

    static final List<String> DOCS = List.of(
            "Grind the beans, then pour water just off the boil over them.",
            "The cat sat on the mat and refused to move all afternoon.",
            "Espresso needs about nine bars of pressure and a fine grind.",
            "Compile Java with javac, then run the class with java.",
            "Green tea should steep below boiling or it turns bitter.");

    public static void main(String[] args) {
        var query = args.length > 0 ? args[0] : "how do I make coffee?";
        try (var embed = JinferEmbeddingModel.builder().modelPath(Models.embed(args, 1)).build()) {
            var docs = embed.embedAll(DOCS.stream().map(TextSegment::from).toList()).content();
            var q = embed.embed(query).content().vector();

            record Hit(double score, String text) {}
            var hits = new ArrayList<Hit>();
            for (int i = 0; i < DOCS.size(); i++) hits.add(new Hit(cosine(q, docs.get(i).vector()), DOCS.get(i)));
            hits.sort(Comparator.comparingDouble(Hit::score).reversed());

            System.out.println("query: " + query + "\n");
            hits.forEach(h -> System.out.printf("  %.3f  %s%n", h.score(), h.text()));
        }
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
