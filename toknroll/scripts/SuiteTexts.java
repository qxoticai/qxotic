import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The text lists of toknroll's test suite, as a JSON array on stdout, for scripts/hf_edge_parity.py:
 * the static List<String> or String[] fields named below, read by reflection from the compiled
 * classes, in this order, so the parity goldens follow the suite without a copy of its texts. A text
 * holding a lone surrogate is left out: not Unicode text, so no reference tokenizer takes it.
 *
 * <p>Run by the script, which puts every toknroll module's classes and test-classes on the class
 * path (mvn test-compile first):
 *
 * <pre>java -cp CLASSPATH scripts/SuiteTexts.java</pre>
 */
public final class SuiteTexts {
    private static final String[] SOURCES = {
        "com.qxotic.toknroll.SplitterUnicodeWhitespaceRegressionTest#UNICODE_WHITESPACE_SAMPLES",
        "com.qxotic.toknroll.TokenizerUnicodeCornerCaseTest#ROUND_TRIP_TEXTS",
        "com.qxotic.toknroll.testkit.TestCorpora#TOKENIZER_UNIVERSAL_SAMPLES",
        "com.qxotic.toknroll.testkit.TestCorpora#SPLITTER_CONTRACT_SAMPLES",
        "com.qxotic.toknroll.testkit.TestCorpora#SPLITTER_PARITY_SAMPLES",
        "com.qxotic.toknroll.testkit.TestCorpora#REGEX_SPLITTER_CORNER_SAMPLES",
        "com.qxotic.toknroll.testkit.TestCorpora#MODEL_TOKENIZER_AGREEMENT_TEXTS",
        "com.qxotic.toknroll.benchmarks.MeasureTokensPerCharLarge#TEST_TEXTS",
        "com.qxotic.toknroll.testkit.SplitterParityHarness#REPRESENTATIVE_SAMPLES",
        "com.qxotic.toknroll.impl.TiktokenSplitterTest#CASES",
        "com.qxotic.toknroll.gguf.GGUFFamilyRecreationTest#SMOKE_TEXTS",
        "com.qxotic.toknroll.gguf.GGUFTokenizerLoaderParityTest#SAMPLE_TEXTS",
        "com.qxotic.toknroll.gguf.GGUFTokenizerParityTest#INVARIANT_SMOKE_TEXTS",
        "com.qxotic.toknroll.hf.HuggingFaceTokenizerParityTest#INVARIANT_SMOKE_TEXTS",
    };

    public static void main(String[] args) throws ReflectiveOperationException {
        List<String> texts = new ArrayList<>();
        for (String source : SOURCES) {
            String[] parts = source.split("#");
            Field field = Class.forName(parts[0]).getDeclaredField(parts[1]);
            field.setAccessible(true);
            Object value = field.get(null);
            List<?> list = value instanceof String[] ? Arrays.asList((String[]) value) : (List<?>) value;
            for (Object text : list) texts.add((String) text);
        }
        StringBuilder json = new StringBuilder("[");
        for (String text : texts) {
            if (!new String(text.getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8).equals(text)) continue;
            if (json.length() > 1) json.append(",\n");
            json.append('"');
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                if (c == '"' || c == '\\') json.append('\\').append(c);
                else if (c < 0x20) json.append(String.format("\\u%04x", (int) c));
                else json.append(c);
            }
            json.append('"');
        }
        System.out.println(json.append("]"));
    }
}
