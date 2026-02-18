///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS ai.qxotic:safetensors:0.1-SNAPSHOT
//DEPS ai.qxotic:json:0.1-SNAPSHOT
//DEPS info.picocli:picocli:4.7.7
//DEPS info.picocli:picocli-codegen:4.7.7
//JAVAC_OPTIONS -proc:full
//NATIVE_OPTIONS --no-fallback -H:+ReportExceptionStackTraces

package scripts;

import ai.qxotic.format.json.JSON;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.TensorEntry;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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

    private static final int BUFFER_SIZE = 1 << 16;
    private static final int CONNECT_TIMEOUT_MS = 10_000;
    private static final int READ_TIMEOUT_MS = 120_000;
    private static final String USER_AGENT = "qxotic-safetensors-script/1.0";
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

        if (isExistingLocalPath(source)) {
            return readSource(source, options);
        }

        RepoRef repo = detectRepo(source);
        if (repo != null) {
            return readRepo(repo.repoId, repo.site, options);
        }

        return readSource(source, options);
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

        @Option(names = "--no-weight-map", description = "Do not include index weight_map")
        boolean noWeightMap;

        @Option(
                names = "--no-extras",
                description = "Hide injected fields (alignment, tensor_data_offset)")
        boolean noExtras;

        @Option(names = "--no-summary", description = "Do not include index summary")
        boolean noSummary;

        boolean includeMetadata() {
            return !noMetadata;
        }

        boolean includeTensors() {
            return !noTensors;
        }

        boolean includeWeightMap() {
            return !noWeightMap;
        }

        boolean includeExtras() {
            return !noExtras;
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
            return readSource(url, options);
        }
    }

    @Command(name = "file")
    static class FileCommand implements Callable<Integer> {
        @Mixin OutputOptions options;
        @Parameters(paramLabel = "FILE") String file;

        @Override
        public Integer call() throws Exception {
            return readSource(file, options);
        }
    }

    private static Integer readSource(String source, OutputOptions options) throws Exception {
        if (!hasScheme(source)) {
            Path path = Path.of(source);
            if (Files.isDirectory(path)) {
                return readLocalDirectory(path, options);
            }
        }

        URL url = toUrl(source);
        if (isIndexFile(source)) {
            return readIndex(url, options, null, null);
        }
        return readSingle(url, options, null);
    }

    private static Integer readLocalDirectory(Path directory, OutputOptions options) throws Exception {
        Path single = directory.resolve(SINGLE_FILE);
        if (Files.isRegularFile(single)) {
            return readSingle(single.toUri().toURL(), options, single.toString());
        }

        Path index = directory.resolve(INDEX_FILE);
        if (Files.isRegularFile(index)) {
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
        } catch (IOException singleErr) {
            URL index = site.resolve(repoId, INDEX_FILE);
            try {
                return readIndex(index, options, repoId, site);
            } catch (IOException indexErr) {
                indexErr.addSuppressed(singleErr);
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
        System.err.println("Reading: " + displayName(url, name));
        Safetensors st = readSafetensors(url);
        printJson(toJson(st, options, name));
        return 0;
    }

    private static Integer readIndex(URL indexUrl, OutputOptions options, String repoId, RepoSite site)
            throws Exception {
        System.err.println("Reading index: " + indexUrl);

        Map<String, Object> index = readJsonObject(indexUrl);
        Map<String, String> weightMap = parseWeightMap(index.get(WEIGHT_MAP));

        LinkedHashSet<String> shardNames = new LinkedHashSet<>(weightMap.values());
        List<String> shardList = new ArrayList<>(shardNames);
        List<Object> shardObjects = readShardsInParallel(indexUrl, repoId, site, options, shardList);

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("source", indexUrl.toString());
        if (options.includeWeightMap()) {
            out.put(WEIGHT_MAP, weightMap);
        }
        if (options.includeSummary()) {
            out.put(
                    "summary",
                    Map.of("tensor_names", weightMap.size(), "shard_files", shardNames.size()));
        }
        out.put("shards", shardObjects);

        printJson(out);
        return 0;
    }

    private static List<Object> readShardsInParallel(
            URL indexUrl, String repoId, RepoSite site, OutputOptions options, List<String> shards)
            throws Exception {
        if (shards.isEmpty()) {
            return new ArrayList<>();
        }
        int workers = Math.min(shards.size(), Math.max(1, Runtime.getRuntime().availableProcessors()));
        ExecutorService pool = Executors.newFixedThreadPool(workers);
        try {
            List<Callable<Map<String, Object>>> tasks = new ArrayList<>(shards.size());
            for (int i = 0; i < shards.size(); i++) {
                final int shardIndex = i + 1;
                final String shardName = shards.get(i);
                tasks.add(
                        () -> {
                            URL shardUrl =
                                    repoId == null
                                            ? new URL(indexUrl, shardName)
                                            : site.resolve(repoId, shardName);
                            System.err.println(
                                    "Reading shard "
                                            + shardIndex
                                            + "/"
                                            + shards.size()
                                            + ": "
                                            + shardUrl);
                            return toJson(readSafetensors(shardUrl), options, shardName);
                        });
            }

            List<Future<Map<String, Object>>> futures = pool.invokeAll(tasks);
            List<Object> shardObjects = new ArrayList<>(futures.size());
            for (int i = 0; i < futures.size(); i++) {
                try {
                    shardObjects.add(futures.get(i).get());
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    if (cause instanceof Exception) {
                        throw (Exception) cause;
                    }
                    throw new IOException("Failed reading shard: " + shards.get(i), cause);
                }
            }
            return shardObjects;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while reading shards", e);
        } finally {
            pool.shutdownNow();
        }
    }

    private static Map<String, Object> toJson(Safetensors st, OutputOptions options, String name) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (name != null) {
            out.put("name", name);
        }
        if (options.includeExtras()) {
            out.put("tensor_data_offset", st.getTensorDataOffset());
            out.put("alignment", st.getAlignment());
        }

        if (options.includeMetadata()) {
            out.put("__metadata__", st.getMetadata());
        }
        if (options.includeTensors()) {
            out.put("tensors", tensorObjects(st));
        }
        return out;
    }

    private static List<Map<String, Object>> tensorObjects(Safetensors st) {
        List<TensorEntry> entries = new ArrayList<>(st.getTensors());
        List<Map<String, Object>> tensors = new ArrayList<>(entries.size());
        for (TensorEntry t : entries) {
            Map<String, Object> tensor = new LinkedHashMap<>();
            tensor.put("name", t.name());
            tensor.put("dtype", t.dtype().toString());
            tensor.put("shape", toList(t.shape()));
            tensor.put("data_offsets", List.of(t.byteOffset(), t.byteOffset() + t.byteSize()));
            tensors.add(tensor);
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
        return JSON.parseObject(readUtf8(url));
    }

    private static Map<String, String> parseWeightMap(Object value) {
        if (!(value instanceof Map)) {
            throw new IllegalArgumentException("Index JSON must contain object field 'weight_map'");
        }
        Map<?, ?> raw = (Map<?, ?>) value;
        Map<String, String> out = new LinkedHashMap<>(raw.size());
        for (Map.Entry<?, ?> e : raw.entrySet()) {
            if (!(e.getKey() instanceof String) || !(e.getValue() instanceof String)) {
                throw new IllegalArgumentException("'weight_map' keys/values must be strings");
            }
            out.put((String) e.getKey(), (String) e.getValue());
        }
        return out;
    }

    private static URL toUrl(String source) throws Exception {
        return hasScheme(source) ? new URL(source) : Path.of(source).toUri().toURL();
    }

    private static boolean isRepoReference(String value, RepoSite site) {
        if (value.startsWith(site.prefix)) {
            return !containsResolvedFilePath(value, site) && !looksLikeFileName(value);
        }
        return !hasScheme(value) && looksLikeRepoId(value);
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

    private static RepoRef detectRepo(String source) {
        if (isRepoReference(source, HF)) {
            return new RepoRef(HF, normalizeRepoId(source, HF));
        }
        if (isRepoReference(source, MODELSCOPE)) {
            return new RepoRef(MODELSCOPE, normalizeRepoId(source, MODELSCOPE));
        }
        return null;
    }

    private static final class RepoRef {
        final RepoSite site;
        final String repoId;

        RepoRef(RepoSite site, String repoId) {
            this.site = site;
            this.repoId = repoId;
        }
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

    private static Safetensors readSafetensors(URL url) throws IOException {
        try (var in = openBuffered(url); var channel = Channels.newChannel(in)) {
            return Safetensors.read(channel);
        }
    }

    private static String readUtf8(URL url) throws IOException {
        try (var in = openBuffered(url)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static BufferedInputStream openBuffered(URL url) throws IOException {
        URLConnection connection = url.openConnection();
        connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
        connection.setReadTimeout(READ_TIMEOUT_MS);
        connection.setUseCaches(false);
        connection.setRequestProperty("User-Agent", USER_AGENT);
        return new BufferedInputStream(connection.getInputStream(), BUFFER_SIZE);
    }

    private static void printJson(Object value) {
        String pretty = JSON.stringify(value, true);
        System.out.println(compactTensors(pretty));
    }

    private static String compactTensors(String prettyJson) {
        StringBuilder out = new StringBuilder(prettyJson.length());
        String[] lines = prettyJson.split("\\R", -1);

        boolean inTensorsArray = false;
        StringBuilder tensor = null;
        String tensorIndent = "";
        int depth = 0;

        for (String line : lines) {
            String trimmed = line.trim();

            if (tensor != null) {
                if (!trimmed.isEmpty()) {
                    if (tensor.length() > 0) {
                        tensor.append(' ');
                    }
                    tensor.append(trimmed);
                }
                depth += countChar(trimmed, '{') - countChar(trimmed, '}');
                if (depth == 0) {
                    out.append(tensorIndent).append(tensor).append('\n');
                    tensor = null;
                }
                continue;
            }

            if (!inTensorsArray && isTensorArrayStart(trimmed)) {
                inTensorsArray = true;
                out.append(line).append('\n');
                continue;
            }

            if (inTensorsArray) {
                if (trimmed.equals("]") || trimmed.equals("],")) {
                    inTensorsArray = false;
                    out.append(line).append('\n');
                    continue;
                }
                if (trimmed.startsWith("{")) {
                    tensor = new StringBuilder();
                    tensorIndent = line.substring(0, line.indexOf(trimmed));
                    tensor.append(trimmed);
                    depth = countChar(trimmed, '{') - countChar(trimmed, '}');
                    if (depth == 0) {
                        out.append(tensorIndent).append(tensor).append('\n');
                        tensor = null;
                    }
                    continue;
                }
            }

            out.append(line).append('\n');
        }

        return out.toString();
    }

    private static int countChar(String value, char ch) {
        int count = 0;
        for (int i = 0; i < value.length(); i++) {
            if (value.charAt(i) == ch) {
                count++;
            }
        }
        return count;
    }

    private static boolean isTensorArrayStart(String trimmedLine) {
        if (!trimmedLine.startsWith("\"tensors\"")) {
            return false;
        }
        int colon = trimmedLine.indexOf(':');
        if (colon < 0) {
            return false;
        }
        for (int i = colon + 1; i < trimmedLine.length(); i++) {
            char c = trimmedLine.charAt(i);
            if (!Character.isWhitespace(c)) {
                return c == '[' && i == trimmedLine.length() - 1;
            }
        }
        return false;
    }

    private static String displayName(URL url, String name) {
        return name == null ? url.toString() : name + " -> " + url;
    }

    private static boolean hasScheme(String value) {
        return value.contains("://");
    }

    private static boolean isIndexFile(String value) {
        return endsWithIgnoreCase(value, ".index.json");
    }

    private static boolean isExistingLocalPath(String value) {
        if (hasScheme(value)) {
            return false;
        }
        try {
            return Files.exists(Path.of(value));
        } catch (RuntimeException ignored) {
            return false;
        }
    }

    private static boolean looksLikeRepoId(String value) {
        String[] parts = value.split("/", -1);
        return parts.length == 2 && !parts[0].isEmpty() && !parts[1].isEmpty();
    }

    private static boolean looksLikeFileName(String value) {
        return endsWithIgnoreCase(value, ".safetensors") || endsWithIgnoreCase(value, ".index.json");
    }

    private static boolean endsWithIgnoreCase(String value, String suffix) {
        return value.length() >= suffix.length()
                && value.regionMatches(true, value.length() - suffix.length(), suffix, 0, suffix.length());
    }

    private static boolean containsResolvedFilePath(String value, RepoSite site) {
        return value.contains("/" + site.resolveSegment + "/")
                || value.contains("/" + site.treeSegment + "/");
    }

    public static void main(String... args) {
        System.exit(new CommandLine(new safetensors()).execute(args));
    }
}
