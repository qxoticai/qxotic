///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS com.qxotic:toknroll-gguf:0.1.0
//DEPS com.qxotic:toknroll-hf:0.1.0
//DEPS info.picocli:picocli:4.7.7
//DEPS info.picocli:picocli-codegen:4.7.7
//JAVAC_OPTIONS -proc:full
//NATIVE_OPTIONS --no-fallback

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

@Command(
        name = "toknroll",
        version = "toknroll 0.1.0",
        description = "Tokenize text or decode token IDs with HuggingFace / ModelScope / GGUF tokenizers")
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
                "Tokenizer source.",
                "  user/repo                HuggingFace tokenizer (default)",
                "  user/repo/quant          GGUF metadata from HuggingFace",
                "  hf:user/repo[/quant]     Same as above (huggingface: alias)",
                "  ms:user/repo[/quant]     ModelScope (modelscope: alias)",
                "  https://...              Full URL (hf.co / huggingface.co)",
                "  ./path or /path          Local .gguf file or HF tokenizer dir"
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
        Tokenizer tokenizer = loadTokenizer(source);
        long t1 = System.nanoTime();

        String text = input != null ? input : readStdin();
        long t2;

        if (decode) {
            String trimmed = input.trim();
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
            IntSequence tokens = tokenizer.encode(input);
            t2 = System.nanoTime();
            System.out.println(tokens.length());
            if (verbose) timing(t0, t1, t2, tokens.length(), "count");
        } else {
            IntSequence tokens = tokenizer.encode(input);
            t2 = System.nanoTime();
            for (int i = 0; i < tokens.length(); i++) System.out.println(tokens.intAt(i));
            if (verbose) timing(t0, t1, t2, tokens.length(), "encode");
        }
        return 0;
    }

    // -- source parsing ------------------------------------------------------

    static Tokenizer loadTokenizer(String source) throws Exception {
        if (source.contains("://")) return resolveUrl(new URL(source));
        if (source.startsWith("ms:")) return ms(source.substring(3));
        if (source.startsWith("modelscope:")) return ms(source.substring(11));
        if (source.startsWith("hf:")) source = source.substring(3);
        else if (source.startsWith("huggingface:")) source = source.substring(13);
        if (isLocalPath(source)) return local(source);
        return hf(source);
    }

    private static boolean isLocalPath(String s) {
        return s.startsWith("/") || s.startsWith("./") || s.startsWith("../");
    }

    static Tokenizer hf(String s) {
        String[] p = s.split("/");
        if (p.length == 2) return HuggingFaceTokenizerLoader.fromHuggingFace(p[0], p[1]);
        if (p.length == 3) return gguf().fromHuggingFace(p[0], p[1], ggufName(p[1], p[2]));
        throw new IllegalArgumentException("Expected user/repo[/quant], got: " + s);
    }

    static Tokenizer ms(String s) {
        String[] p = s.split("/");
        if (p.length == 2) return HuggingFaceTokenizerLoader.fromModelScope(p[0], p[1]);
        if (p.length == 3) return gguf().fromModelScope(p[0], p[1], ggufName(p[1], p[2]));
        throw new IllegalArgumentException("Expected user/repo[/quant], got: " + s);
    }

    static String ggufName(String repo, String quant) {
        String base = repo.endsWith("-GGUF") ? repo.substring(0, repo.length() - 5) : repo;
        return base + "-" + quant + ".gguf";
    }

    static Tokenizer resolveUrl(URL url) {
        if ("file".equals(url.getProtocol())) {
            String host = url.getHost();
            String path = url.getPath();
            return local((host != null && !host.isEmpty()) ? host + path : path);
        }

        String host = url.getHost();
        boolean ms = host.equals("modelscope.cn") || host.equals("www.modelscope.cn");
        boolean hf = host.endsWith("huggingface.co") || host.equals("hf.co");
        if (!ms && !hf) throw new IllegalArgumentException(
                "Unsupported host: " + host + " (expected huggingface.co, hf.co, or modelscope.cn)");

        String[] p = url.getPath().split("/");
        int off = ms ? 2 : 1; // ModelScope: /models/user/repo/..., HF: /user/repo/...

        if (p.length >= off + 4 && "resolve".equals(p[off + 2])) {
            String file = String.join("/", Arrays.copyOfRange(p, off + 4, p.length));
            if (!file.endsWith(".gguf"))
                throw new IllegalArgumentException("Resolve URL must end in .gguf: " + url);
            return ms
                    ? gguf().fromModelScope(p[off], p[off + 1], p[off + 3], file, false, false)
                    : gguf().fromHuggingFace(p[off], p[off + 1], p[off + 3], file, false, false);
        }
        if (p.length >= off + 2) {
            return ms
                    ? HuggingFaceTokenizerLoader.fromModelScope(p[off], p[off + 1])
                    : HuggingFaceTokenizerLoader.fromHuggingFace(p[off], p[off + 1]);
        }
        throw new IllegalArgumentException("Unable to parse URL: " + url);
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
