///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS com.qxotic:gguf:0.1.0
//DEPS info.picocli:picocli:4.7.7
//DEPS info.picocli:picocli-codegen:4.7.7
//JAVAC_OPTIONS -proc:full
//NATIVE_OPTIONS --no-fallback -H:+ReportExceptionStackTraces

import com.qxotic.format.gguf.GGUF;
import java.io.BufferedInputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

/**
 * GGUF metadata viewer - displays model metadata and tensor information.
 * Everything is a URL: file:// for local files, https:// for remote.
 */
@Command(
    name = "gguf",
    description = "Display GGUF file metadata and tensor information",
    mixinStandardHelpOptions = true,
    subcommands = {gguf.HfCommand.class, gguf.ModelScopeCommand.class, gguf.UrlCommand.class, gguf.FileCommand.class}
)
class gguf implements Callable<Integer> {

    @Mixin
    OutputOptions options;

    @Parameters(paramLabel = "SOURCE", description = "File path or URL to GGUF file", arity = "0..1")
    String source;

    @Override
    public Integer call() throws Exception {
        if (source == null) {
            printQuickHelp();
            return 1;
        }
        return readAndPrint(toUrl(source), options);
    }

    private static void printQuickHelp() {
        System.err.println("Usage: gguf <file|url> [options]");
        System.err.println("       gguf hf <user/repo/quant> [options]");
        System.err.println("       gguf modelscope <user/repo/quant> [options]");
        System.err.println();
        System.err.println("Options: --no-metadata, --no-tensors");
        System.err.println("Run 'gguf --help' for full help.");
    }

    /** Convert any input to URL (file:// for local paths). */
    private static URL toUrl(String source) throws Exception {
        return source.contains("://") ? new URL(source) : new URL("file", "", source);
    }

    /** Unified read - everything comes through here as a URL. */
    static int readAndPrint(URL url, OutputOptions options) throws Exception {
        System.err.println("Reading: " + url);
        try (var channel = Channels.newChannel(new BufferedInputStream(url.openStream(), 1 << 16))) {
            var gguf = GGUF.read(channel);
            System.out.println(gguf.toString(options.printMetadata(), options.printTensors()));
            return 0;
        }
    }

    /** Output control options. */
    static class OutputOptions {
        @Option(names = "--no-metadata", description = "Do not print metadata")
        boolean noMetadata;

        @Option(names = "--no-tensors", description = "Do not print tensors")
        boolean noTensors;

        boolean printMetadata() { return !noMetadata; }
        boolean printTensors() { return !noTensors; }
    }

    @Command(name = "hf", aliases = "huggingface")
    static class HfCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "SHORTHAND") String shorthand;

        @Override
        public Integer call() throws Exception {
            return readAndPrint(toHfUrl(shorthand), options);
        }
    }

    @Command(name = "modelscope")
    static class ModelScopeCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "SHORTHAND") String shorthand;

        @Override
        public Integer call() throws Exception {
            return readAndPrint(toModelScopeUrl(shorthand), options);
        }
    }

    @Command(name = "url")
    static class UrlCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "URL") String url;

        @Override
        public Integer call() throws Exception {
            return readAndPrint(new URL(url), options);
        }
    }

    @Command(name = "file")
    static class FileCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "FILE") String file;

        @Override
        public Integer call() throws Exception {
            return readAndPrint(new URL("file", "", file), options);
        }
    }

    private static URL toHfUrl(String shorthand) throws Exception {
        var parts = shorthand.split("/");
        if (parts.length < 3) throw new IllegalArgumentException("Expected: user/repo/quant");
        String filename = parts[1].replace("-GGUF", "") + "-" + parts[2] + ".gguf";
        return new URL("https://huggingface.co/" + parts[0] + "/" + parts[1] + "/resolve/main/" + filename);
    }

    private static URL toModelScopeUrl(String shorthand) throws Exception {
        var parts = shorthand.split("/");
        if (parts.length < 3) throw new IllegalArgumentException("Expected: user/repo/quant");
        String filename = parts[1].replace("-GGUF", "") + "-" + parts[2] + ".gguf";
        return new URL("https://www.modelscope.cn/models/" + parts[0] + "/" + parts[1] + "/resolve/master/" + filename);
    }

    public static void main(String... args) {
        System.exit(new CommandLine(new gguf()).execute(args));
    }
}
