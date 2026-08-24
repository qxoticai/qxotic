///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-qwen3
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Rerank documents by scoring each query and document pair.
//   jbang Rerank.java "what causes coffee bitterness?"
import com.qxotic.jinfer.langchain4j.JinferScoringModel;
import dev.langchain4j.data.segment.TextSegment;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Rerank {

    private static final String DEFAULT_MODEL =
            "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0";

    private static final List<String> DOCUMENTS = List.of(
            "Over-extraction pulls harsh compounds from the grounds and tastes bitter.",
            "Coffee is grown across the equatorial belt, mostly at altitude.",
            "Water above 96C scalds the grounds and adds an acrid edge.",
            "The cat sat on the mat and refused to move all afternoon.");

    public static void main(String[] args) {
        String query = args.length > 0 ? args[0] : "what causes coffee bitterness?";
        String modelRef = args.length > 1 ? args[1] : DEFAULT_MODEL;

        try (var scorer = JinferScoringModel.builder().model(modelRef).build()) {
            var scores = scorer
                    .scoreAll(DOCUMENTS.stream().map(TextSegment::from).toList(), query)
                    .content();

            record Hit(double score, String text) {}

            var hits = new ArrayList<Hit>();
            for (int i = 0; i < DOCUMENTS.size(); i++) {
                hits.add(new Hit(scores.get(i), DOCUMENTS.get(i)));
            }
            hits.sort(Comparator.comparingDouble(Hit::score).reversed());

            System.out.printf("Query: %s%n%n", query);
            hits.forEach(hit -> System.out.printf("%.4f  %s%n", hit.score(), hit.text()));
        }
    }
}
