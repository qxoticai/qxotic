///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-xlangchain4j:0.1.0
//SOURCES Models.java

// Reranking with a cross-encoder: it reads the query and the document TOGETHER, which is why it
// beats embedding similarity on the hard cases. The classic second stage of a RAG pipeline.
//   jbang Rerank.java "what causes coffee bitterness?"
import com.qxotic.jinfer.langchain4j.JinferScoringModel;
import dev.langchain4j.data.segment.TextSegment;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Rerank {

    static final List<String> DOCS = List.of(
            "Over-extraction pulls harsh compounds from the grounds and tastes bitter.",
            "Coffee is grown across the equatorial belt, mostly at altitude.",
            "Water above 96C scalds the grounds and adds an acrid edge.",
            "The cat sat on the mat and refused to move all afternoon.");

    public static void main(String[] args) {
        var query = args.length > 0 ? args[0] : "what causes coffee bitterness?";
        try (var scorer = JinferScoringModel.builder().model(Models.rerank(args, 1)).build()) {
            var scores = scorer.scoreAll(DOCS.stream().map(TextSegment::from).toList(), query).content();

            record Hit(double score, String text) {}
            var hits = new ArrayList<Hit>();
            for (int i = 0; i < DOCS.size(); i++) hits.add(new Hit(scores.get(i), DOCS.get(i)));
            hits.sort(Comparator.comparingDouble(Hit::score).reversed());

            System.out.println("query: " + query + "\n");
            hits.forEach(h -> System.out.printf("  %.4f  %s%n", h.score(), h.text()));
        }
    }
}
