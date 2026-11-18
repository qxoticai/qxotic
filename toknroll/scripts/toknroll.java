///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//DEPS com.qxotic:toknroll-gguf:0.4.0
//DEPS com.qxotic:toknroll-hf:0.4.0
//DEPS com.qxotic:jinfer-hub:0.4.0
//DEPS info.picocli:picocli:4.7.7
//DEPS info.picocli:picocli-codegen:4.7.7
//JAVAC_OPTIONS -proc:full

import com.qxotic.jinfer.hub.ModelRef;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

@Command(
        name = "toknroll",
        version = "toknroll 0.4.0",
        description = "Tokenize text or decode token IDs with Hugging Face / ModelScope / GGUF tokenizers")
class toknroll implements Callable<Integer> {

    private static GGUFTokenizerLoader ggufLoader;

    private static GGUFTokenizerLoader gguf() {
        if (ggufLoader == null) ggufLoader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        return ggufLoader;
    }

    @Option(names = {"-h", "--help"}, usageHelp = true, description = "Show this help and exit")
    boolean helpRequested;

    @Option(names = {"-V", "--version"}, versionHelp = true, description = "Print version and exit")
    boolean versionRequested;

    @Option(names = {"-v", "--verbose"}, description = "Print timing breakdown to stderr")
    boolean verbose;

    @Option(names = {"-e", "--encode"}, description = "Encode text to token IDs (default)")
    boolean encode;

    @Option(names = {"-d", "--decode"}, description = "Decode token IDs to text")
    boolean decode;

    @Option(names = {"-c", "--count"}, description = "Print token count only")
    boolean count;

    @Option(
            names = {"-s", "--source"},
            required = true,
            paramLabel = "SOURCE",
            description = {
                "Tokenizer source: [host/]owner/repo[@revision][/path][:quant] or a local path.",
                "  google/gemma-4-e2b-it                      the repository's tokenizer.json",
                "  unsloth/Llama-3.2-1B-Instruct-GGUF:Q8_0    a GGUF's tokenizer, its header only",
                "  modelscope.cn/Qwen/Qwen3-4B                from ModelScope",
                "  ./model.gguf or ./tokenizer-dir            a local GGUF file or tokenizer folder"
            })
    String source;

    @Option(names = {"-i", "--input"}, paramLabel = "INPUT",
            description = "Input text or token IDs (reads stdin by default)")
    String input;

    @Override
    public Integer call() throws Exception {
        if ((encode ? 1 : 0) + (decode ? 1 : 0) + (count ? 1 : 0) > 1) {
            System.err.println("error: -e, -d, -c are mutually exclusive");
            return 1;
        }
        long t0 = System.nanoTime();
        Tokenizer tokenizer;
        try {
            tokenizer = loadTokenizer(source);
        } catch (RuntimeException failed) { // a URL, a typo, the network: the message says it
            System.err.println("error: " + failed.getMessage());
            return 1;
        }
        long t1 = System.nanoTime();

        String text = input != null ? input : readStdin();
        long t2;

        if (decode) {
            String trimmed = text.trim();
            if (trimmed.isEmpty()) {
                System.err.println("error: no token IDs provided for decode");
                return 1;
            }
            int[] ids = Arrays.stream(trimmed.split("\\s+"))
                    .mapToInt(Integer::parseInt).toArray();
            System.out.println(tokenizer.decode(IntSequence.wrap(ids)));
            t2 = System.nanoTime();
            if (verbose) timing(t0, t1, t2, ids.length, "decode");
        } else if (count) {
            IntSequence tokens = tokenizer.encode(text);
            t2 = System.nanoTime();
            System.out.println(tokens.length());
            if (verbose) timing(t0, t1, t2, tokens.length(), "count");
        } else {
            IntSequence tokens = tokenizer.encode(text);
            t2 = System.nanoTime();
            for (int i = 0; i < tokens.length(); i++) System.out.println(tokens.intAt(i));
            if (verbose) timing(t0, t1, t2, tokens.length(), "encode");
        }
        return 0;
    }

    // -- source: jinfer's model reference, or a local path -------------------

    static Tokenizer loadTokenizer(String source) {
        if (!ModelStore.isRef(source) && !source.contains("://")) return local(source);
        ModelRef ref = ModelRef.parse(source); // a URL is refused, naming the ref to pass
        // a quant picks its file from the repository listing, as jinfer does: nothing downloads
        String file = ref.quant() != null ? ModelStore.standard().select(source) : ref.path();
        boolean ms = ref.host().equals("modelscope.cn");
        String owner = ref.owner(), repo = ref.repo(), revision = ref.revision();
        if (file.isEmpty()) // a repository: its tokenizer.json
            return ms
                    ? HuggingFaceTokenizerLoader.fromModelScope(owner, repo, revision, false, false)
                    : HuggingFaceTokenizerLoader.fromHuggingFace(owner, repo, revision, false, false);
        if (!file.endsWith(".gguf"))
            throw new IllegalArgumentException(
                    source + ": a file in a repository is read as a GGUF, name a .gguf");
        return ms // only the GGUF header is fetched, never the weights
                ? gguf().fromModelScope(owner, repo, revision, file, false, false)
                : gguf().fromHuggingFace(owner, repo, revision, file, false, false);
    }

    static Tokenizer local(String source) {
        Path path = Path.of(source);
        return source.endsWith(".gguf")
                ? gguf().fromLocal(path)
                : HuggingFaceTokenizerLoader.fromLocal(path);
    }

    // -- helpers -------------------------------------------------------------

    static void timing(long t0, long t1, long t2, int count, String op) {
        System.err.printf("load= %5.0fms  %s= %5.0fms  tokens= %d%n",
                (t1 - t0) / 1e6, op, (t2 - t1) / 1e6, count);
    }

    static String readStdin() {
        return new BufferedReader(new InputStreamReader(System.in))
                .lines().collect(Collectors.joining("\n"));
    }

    public static void main(String... args) {
        int exitCode = new CommandLine(new toknroll()).execute(args);
        System.exit(exitCode);
    }
}
