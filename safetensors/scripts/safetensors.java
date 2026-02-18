package scripts;

//DEPS ai.qxotic:safetensors:0.1-SNAPSHOT
//DEPS info.picocli:picocli:4.7.6
//DEPS info.picocli:picocli-codegen:4.7.6
//JAVAC_OPTIONS -proc:full
//NATIVE_OPTIONS --no-fallback -H:+ReportExceptionStackTraces

import ai.qxotic.format.json.JSON;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.TensorEntry;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

/** Safetensors header viewer that prints pure JSON to stdout. */
@Command(
        name = "safetensors",
        description = "Display Safetensors header metadata and tensor entries as JSON",
        mixinStandardHelpOptions = true,
        subcommands = {
            safetensors.HfCommand.class,
            safetensors.ModelScopeCommand.class,
            safetensors.UrlCommand.class,
            safetensors.FileCommand.class
        })
class safetensors implements Callable<Integer> {

    private static final String SINGLE_FILE = "model.safetensors";
    private static final String INDEX_FILE = "model.safetensors.index.json";
    private static final String WEIGHT_MAP = "weight_map";

    private static final RepoSite HF =
            new RepoSite("hf", "https://huggingface.co/", "main", "resolve", "tree");
    private static final RepoSite MODELSCOPE =
            new RepoSite(
                    "modelscope",
                    "https://www.modelscope.cn/models/",
                    "master",
                    "resolve",
                    "files");

    @Mixin OutputOptions options;

    @Parameters(
            paramLabel = "SOURCE",
            description = "File path, URL, HF repo (org/repo), or HF /tree/main URL",
            arity = "0..1")
    String source;

    @Override
    public Integer call() throws Exception {
        if (source == null) {
            printQuickHelp();
            return 1;
        }

        if (isRepoReference(source, HF)) {
            return readRepo(normalizeRepoId(source, HF), HF, options);
        }
        if (isRepoReference(source, MODELSCOPE)) {
            return readRepo(normalizeRepoId(source, MODELSCOPE), MODELSCOPE, options);
        }

        return readAnySource(source, options);
    }

    private static void printQuickHelp() {
        System.err.println("Usage: safetensors <file|url|hf-repo> [options]");
        System.err.println("       safetensors hf <org/repo|tree-url> [options]");
        System.err.println("       safetensors modelscope <org/repo|modelscope-url> [options]");
        System.err.println();
        System.err.println("Output is pure JSON on stdout.");
    }

    static class OutputOptions {
        @Option(names = "--no-metadata", description = "Do not include metadata")
        boolean noMetadata;

        @Option(names = "--no-tensors", description = "Do not include tensor entries")
        boolean noTensors;

        @Option(names = "--no-summary", description = "Do not include index summary")
        boolean noSummary;

        boolean includeMetadata() {
            return !noMetadata;
        }

        boolean includeTensors() {
            return !noTensors;
        }

        boolean includeSummary() {
            return !noSummary;
        }
    }

    @Command(name = "hf", aliases = "huggingface")
    static class HfCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "REPO") String repo;

        @Override
        public Integer call() throws Exception {
            return readRepo(normalizeRepoId(repo, HF), HF, options);
        }
    }

    @Command(name = "modelscope")
    static class ModelScopeCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "REPO") String repo;

        @Override
        public Integer call() throws Exception {
            return readRepo(normalizeRepoId(repo, MODELSCOPE), MODELSCOPE, options);
        }
    }

    @Command(name = "url")
    static class UrlCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "URL") String url;

        @Override
        public Integer call() throws Exception {
            return readAnySource(url, options);
        }
    }

    @Command(name = "file")
    static class FileCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "FILE") String file;

        @Override
        public Integer call() throws Exception {
            return readAnySource(file, options);
        }
    }

    private static Integer readAnySource(String source, OutputOptions options) throws Exception {
        if (!source.contains("://")) {
            Path path = Path.of(source);
            if (Files.isDirectory(path)) {
                return readLocalDirectory(path, options);
            }
        }

        URL url = toUrl(source);
        if (source.toLowerCase().endsWith(".index.json")) {
            return readIndex(url, options, null, null);
        }
        return readSingle(url, options, null);
    }

    private static Integer readLocalDirectory(Path directory, OutputOptions options) throws Exception {
        Path single = directory.resolve(SINGLE_FILE);
        if (Files.exists(single)) {
            return readSingle(single.toUri().toURL(), options, single.toString());
        }

        Path index = directory.resolve(INDEX_FILE);
        if (Files.exists(index)) {
            return readIndex(index.toUri().toURL(), options, null, null);
        }

        throw new IOException(
                "No safetensors files found in directory: expected '"
                        + SINGLE_FILE
                        + "' or '"
                        + INDEX_FILE
                        + "'");
    }

    private static Integer readRepo(String repoId, RepoSite site, OutputOptions options)
            throws Exception {
        URL single = site.resolve(repoId, SINGLE_FILE);
        try {
            return readSingle(single, options, site.label + ":" + repoId + ":" + SINGLE_FILE);
        } catch (IOException e) {
            URL index = site.resolve(repoId, INDEX_FILE);
            try {
                return readIndex(index, options, repoId, site);
            } catch (IOException indexErr) {
                throw new IOException(
                        "Could not read '"
                                + SINGLE_FILE
                                + "' or '"
                                + INDEX_FILE
                                + "' from "
                                + site.label
                                + " repo "
                                + repoId,
                        indexErr);
            }
        }
    }

    private static Integer readSingle(URL url, OutputOptions options, String name) throws Exception {
        System.err.println("Reading: " + (name == null ? url : name + " -> " + url));
        try (var channel = Channels.newChannel(new BufferedInputStream(url.openStream(), 1 << 16))) {
            Safetensors st = Safetensors.read(channel);
            System.out.println(JSON.stringify(toJson(st, options, name), true));
            return 0;
        }
    }

    private static Integer readIndex(URL indexUrl, OutputOptions options, String repoId, RepoSite site)
            throws Exception {
        System.err.println("Reading index: " + indexUrl);

        Map<String, Object> index = readJsonObject(indexUrl);
        Map<String, String> weightMap = parseWeightMap(index.get(WEIGHT_MAP));

        LinkedHashSet<String> shardNames = new LinkedHashSet<>(weightMap.values());
        List<Object> shardObjects = new ArrayList<>(shardNames.size());

        int i = 1;
        for (String shard : shardNames) {
            URL shardUrl = repoId == null ? new URL(indexUrl, shard) : site.resolve(repoId, shard);
            System.err.println("Reading shard " + i + "/" + shardNames.size() + ": " + shardUrl);
            try (var channel =
                    Channels.newChannel(new BufferedInputStream(shardUrl.openStream(), 1 << 16))) {
                shardObjects.add(toJson(Safetensors.read(channel), options, shard));
            }
            i++;
        }

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("source", indexUrl.toString());
        out.put(WEIGHT_MAP, weightMap);
        if (options.includeSummary()) {
            out.put(
                    "summary",
                    Map.of("tensor_names", weightMap.size(), "shard_files", shardNames.size()));
        }
        out.put("shards", shardObjects);

        System.out.println(JSON.stringify(out, true));
        return 0;
    }

    private static Map<String, Object> toJson(Safetensors st, OutputOptions options, String name) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (name != null) {
            out.put("name", name);
        }
        out.put("tensor_data_offset", st.getTensorDataOffset());
        out.put("alignment", st.getAlignment());

        if (options.includeMetadata()) {
            out.put("__metadata__", st.getMetadata());
        }
        if (options.includeTensors()) {
            out.put("tensors", tensorMap(st));
        }
        return out;
    }

    private static Map<String, Object> tensorMap(Safetensors st) {
        Map<String, Object> tensors = new LinkedHashMap<>();
        for (TensorEntry t : st.getTensors()) {
            tensors.put(
                    t.name(),
                    Map.of(
                            "dtype", t.dtype().toString(),
                            "shape", toList(t.shape()),
                            "data_offsets", List.of(t.byteOffset(), t.byteOffset() + t.byteSize())));
        }
        return tensors;
    }

    private static List<Long> toList(long[] values) {
        List<Long> out = new ArrayList<>(values.length);
        for (long value : values) {
            out.add(value);
        }
        return out;
    }

    private static Map<String, Object> readJsonObject(URL url) throws Exception {
        try (var in = new BufferedInputStream(url.openStream(), 1 << 16)) {
            return JSON.parseObject(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    private static Map<String, String> parseWeightMap(Object value) {
        if (!(value instanceof Map)) {
            throw new IllegalArgumentException("Index JSON must contain object field 'weight_map'");
        }
        Map<?, ?> raw = (Map<?, ?>) value;
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : raw.entrySet()) {
            if (!(e.getKey() instanceof String) || !(e.getValue() instanceof String)) {
                throw new IllegalArgumentException("'weight_map' keys/values must be strings");
            }
            out.put((String) e.getKey(), (String) e.getValue());
        }
        return out;
    }

    private static URL toUrl(String source) throws Exception {
        return source.contains("://") ? new URL(source) : new URL("file", "", source);
    }

    private static boolean isRepoReference(String value, RepoSite site) {
        if (!value.startsWith(site.prefix)) {
            return !value.contains("://") && value.split("/").length == 2;
        }
        return !value.contains("/" + site.resolveSegment + "/")
                && !value.contains("/" + site.treeSegment + "/")
                && !value.endsWith(".safetensors")
                && !value.endsWith(".index.json");
    }

    private static String normalizeRepoId(String value, RepoSite site) {
        if (!value.startsWith(site.prefix)) {
            return value;
        }
        String rest = value.substring(site.prefix.length());
        String[] parts = rest.split("/");
        if (parts.length < 2) {
            throw new IllegalArgumentException("Expected repo id: org/repo");
        }
        return parts[0] + "/" + parts[1];
    }

    private static final class RepoSite {
        final String label;
        final String prefix;
        final String branch;
        final String resolveSegment;
        final String treeSegment;

        RepoSite(String label, String prefix, String branch, String resolveSegment, String treeSegment) {
            this.label = label;
            this.prefix = prefix;
            this.branch = branch;
            this.resolveSegment = resolveSegment;
            this.treeSegment = treeSegment;
        }

        URL resolve(String repoId, String fileName) throws Exception {
            return new URL(prefix + repoId + "/" + resolveSegment + "/" + branch + "/" + fileName);
        }
    }

    public static void main(String... args) {
        System.exit(new CommandLine(new safetensors()).execute(args));
    }
}
