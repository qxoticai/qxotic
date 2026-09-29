package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.hub.ModelStore;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Run after package. Tests what users launch, and proves test providers never ship. */
@Tag("integration")
class DistributionIT {
    @TempDir Path dir;

    private static Path jar() {
        return Path.of(System.getProperty("jinfer.test.jar", "target/jinfer.jar")).toAbsolutePath();
    }

    @Test
    void executableJarContainsOnlyProductionProvidersAndAnEntryPoint() throws Exception {
        try (var jar = new JarFile(jar().toFile())) {
            assertEquals(
                    Main.class.getName(),
                    jar.getManifest().getMainAttributes().getValue("Main-Class"));
            assertNotNull(jar.getManifest().getMainAttributes().getValue("Implementation-Version"));
            assertNull(jar.getEntry(CliModelProvider.class.getName().replace('.', '/') + ".class"));
            String service = "META-INF/services/" + ModelProvider.class.getName();
            try (var in = jar.getInputStream(jar.getJarEntry(service))) {
                Set<String> shipped =
                        new String(in.readAllBytes(), StandardCharsets.UTF_8)
                                .lines()
                                .map(String::strip)
                                .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                                .collect(Collectors.toSet());
                Set<String> expected =
                        ServiceLoader.load(ModelProvider.class).stream()
                                .map(provider -> provider.type().getName())
                                .filter(name -> !name.equals(CliModelProvider.class.getName()))
                                .collect(Collectors.toSet());
                assertEquals(expected, shipped);
            }
        }
    }

    static Stream<Arguments> executables() {
        var java = CliFixtures.javaCommand();
        java.addAll(List.of("-jar", jar().toString()));
        var commands = new ArrayList<Arguments>();
        commands.add(Arguments.of("jar", java));
        String nativeFile = System.getProperty("jinfer.test.executable");
        if (nativeFile != null)
            commands.add(
                    Arguments.of(
                            "native", List.of(Path.of(nativeFile).toAbsolutePath().toString())));
        return commands.stream();
    }

    @ParameterizedTest(name = "{0}: user-facing smoke tests")
    @MethodSource("executables")
    void packagedCommandsHaveHelpAliasesExitCodesAndCleanStreams(String name, List<String> command)
            throws Exception {
        for (String verb :
                List.of(
                        "chat",
                        "instruct",
                        "prompt",
                        "server",
                        "serve",
                        "speak",
                        "transcribe",
                        "pull",
                        "list",
                        "cache-info")) {
            Result help = run(command, "", verb, "--help");
            assertEquals(0, help.status(), help.err());
            assertTrue(help.out().contains("Usage:"));
            assertFalse(help.err().contains("Exception in thread"));
        }
        Result version = run(command, "", "--version");
        assertEquals(0, version.status(), version.err());
        assertTrue(version.out().startsWith("jinfer "));
        for (String removed :
                List.of(
                        "--chat",
                        "--instruct",
                        "--server",
                        "--speak",
                        "--transcribe",
                        "--prompt")) {
            Result rejected = run(command, "", "-m", "missing.gguf", removed);
            assertEquals(2, rejected.status(), rejected.err());
            assertTrue(rejected.err().contains("unknown option: " + removed));
            assertEquals("", rejected.out());
        }
        Result noCommand = run(command, "", "-m", "missing.gguf");
        assertEquals(2, noCommand.status());
        assertTrue(noCommand.err().contains("missing command"));
        Result invalid =
                run(
                        command,
                        "",
                        "speak",
                        "-m",
                        "uncached/repository:Q8_0",
                        "hello",
                        "--speed",
                        "0");
        assertEquals(2, invalid.status(), invalid.err());
        assertEquals("", invalid.out());
        assertTrue(invalid.err().contains("--speed"));
        Result blank = run(command, " \n", "instruct", "-m", "missing.gguf", "-");
        assertEquals(2, blank.status(), blank.err());
        assertTrue(blank.err().contains("non-blank text"));
        Result missing = run(command, "", "cache-info", dir.resolve("missing.jkv").toString());
        assertEquals(1, missing.status());
        assertEquals("", missing.out());
        Path unsupported = dir.resolve("unsupported model 日本語.gguf");
        GGUF.write(
                Builder.newBuilder().putString("general.architecture", "cli_test_language").build(),
                unsupported);
        Result unknownModel = run(command, "", "chat", "-m", unsupported.toString());
        assertEquals(1, unknownModel.status(), unknownModel.err());
        assertTrue(unknownModel.err().contains("cli_test_language"));
        assertFalse(unknownModel.err().contains("Exception in thread"));
        assertFalse(
                Files.exists(dir.resolve("cache")),
                "help and invalid inputs must not create a cache");
    }

    @Test
    void offlineForcePullPreservesSharedHubEntry() throws Exception {
        Path repo = dir.resolve("hf/models--qa--preserve");
        Path blob = repo.resolve("blobs/" + "b".repeat(64));
        Path snapshot = repo.resolve("snapshots/" + "a".repeat(40));
        Files.createDirectories(blob.getParent());
        Files.createDirectories(snapshot);
        Files.createDirectories(repo.resolve("refs"));
        Files.writeString(repo.resolve("refs/main"), "a".repeat(40));
        Files.writeString(blob, "working model");
        Path link = snapshot.resolve("model-Q8_0.gguf");
        Files.createSymbolicLink(link, Path.of("../../blobs/" + blob.getFileName()));
        var command = CliFixtures.javaCommand();
        command.addAll(List.of("-jar", jar().toString()));
        Result failed = run(command, "", "pull", "--force", "qa/preserve:Q8_0");
        assertEquals(1, failed.status(), failed.err());
        assertTrue(failed.err().contains("JINFER_OFFLINE"), failed.err());
        assertFalse(failed.err().contains("not cached"), "the cached model still exists");
        assertEquals("", failed.out());
        assertTrue(Files.isSymbolicLink(link), "refresh removed the snapshot link");
        assertEquals("working model", Files.readString(link));
        assertEquals("working model", Files.readString(blob));
        Result cached =
                run(command, "", "pull", "qa/preserve@" + "a".repeat(40) + "/model-Q8_0.gguf");
        assertEquals(0, cached.status(), cached.err());
        assertEquals(link.toString(), cached.out().strip());

        Path folder = Files.createDirectory(snapshot.resolve("sub"));
        Files.writeString(folder.resolve("sub"), "nested file");
        Result directory = run(command, "", "pull", "qa/preserve@" + "a".repeat(40) + "/sub");
        assertEquals(1, directory.status(), "a directory is not an exact cached filename");
        assertTrue(directory.err().contains("JINFER_OFFLINE"), directory.err());
        Result exact = run(command, "", "pull", "qa/preserve@" + "a".repeat(40) + "/sub/sub");
        assertEquals(0, exact.status(), exact.err());
        assertEquals(folder.resolve("sub").toString(), exact.out().strip());
    }

    @Test
    void packagedPullUpdatesSharedCacheAndHonorsAFlatShadow() throws Exception {
        String oldCommit = "a".repeat(40), newCommit = "b".repeat(40);
        AtomicReference<String> revision = new AtomicReference<>(oldCommit);
        Map<String, String> responses = new HashMap<>();
        Map<String, AtomicInteger> hits = new ConcurrentHashMap<>();
        for (var entry : Map.of(oldCommit, "before", newCommit, "after!").entrySet()) {
            String hash =
                    HexFormat.of()
                            .formatHex(
                                    MessageDigest.getInstance("SHA-256")
                                            .digest(
                                                    entry.getValue()
                                                            .getBytes(StandardCharsets.UTF_8)));
            responses.put(
                    "/api/models/qa/update/tree/" + entry.getKey(),
                    "[{\"type\":\"file\",\"path\":\"model-Q8_0.gguf\",\"size\":6,\"lfs\":{\"oid\":\""
                            + hash
                            + "\"}}]");
            responses.put(
                    "/qa/update/resolve/" + entry.getKey() + "/model-Q8_0.gguf", entry.getValue());
        }
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    String path = exchange.getRequestURI().getPath();
                    hits.computeIfAbsent(path, key -> new AtomicInteger()).incrementAndGet();
                    String body =
                            path.endsWith("/refs")
                                    ? "{\"branches\":[{\"name\":\"main\",\"targetCommit\":\""
                                            + revision.get()
                                            + "\"}]}"
                                    : responses.get(path);
                    byte[] bytes =
                            (body == null ? "not found" : body).getBytes(StandardCharsets.UTF_8);
                    exchange.sendResponseHeaders(body == null ? 404 : 200, bytes.length);
                    try (var out = exchange.getResponseBody()) {
                        out.write(bytes);
                    }
                });
        server.start();
        try {
            Path home = dir.resolve("home");
            Path platform =
                    System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("mac")
                            ? home.resolve("Library/Caches")
                            : dir.resolve("cache-home");
            Path flatRoot = platform.resolve("jinfer");
            Map<String, String> env =
                    Map.of(
                            "JINFER_OFFLINE",
                            "0",
                            "JINFER_MODELS",
                            flatRoot.toString(),
                            "XDG_CACHE_HOME",
                            platform.toString(),
                            "LOCALAPPDATA",
                            platform.toString(),
                            "HF_ENDPOINT",
                            "http://127.0.0.1:" + server.getAddress().getPort());
            var command = CliFixtures.javaCommand();
            command.addAll(List.of("-Duser.home=" + home, "-jar", jar().toString()));
            Result first = run(command, "", env, "pull", "qa/update:Q8_0");
            assertEquals(0, first.status(), first.err());
            Path old = Path.of(first.out().strip());
            assertTrue(old.startsWith(dir.resolve("hf")), "default root must publish in HF cache");
            assertEquals("before", Files.readString(old));
            revision.set(newCommit);
            Result updated = run(command, "", env, "pull", "qa/update:Q8_0");
            assertEquals(0, updated.status(), updated.err());
            Path fresh = Path.of(updated.out().strip());
            assertEquals("after!", Files.readString(fresh));
            assertEquals("before", Files.readString(old));
            assertEquals(
                    newCommit, Files.readString(dir.resolve("hf/models--qa--update/refs/main")));
            String payload = "/qa/update/resolve/" + newCommit + "/model-Q8_0.gguf";
            int fetched = hits.get(payload).get();
            assertEquals(0, run(command, "", env, "pull", "qa/update:Q8_0").status());
            assertEquals(fetched, hits.get(payload).get());
            assertEquals(0, run(command, "", env, "pull", "--force", "qa/update:Q8_0").status());
            assertEquals(fetched + 1, hits.get(payload).get());
            Path branch = dir.resolve("hf/models--qa--update/refs/main");
            Files.write(branch, new byte[] {(byte) 0xff});
            Result repairedRef = run(command, "", env, "pull", "--force", "qa/update:Q8_0");
            assertEquals(0, repairedRef.status(), repairedRef.err());
            assertEquals(newCommit, Files.readString(branch));

            Path shadow = flatRoot.resolve("hf.co/qa/update/model-Q8_0.gguf");
            Files.createDirectories(shadow.getParent());
            Files.writeString(shadow, "stale!");
            Result repaired = run(command, "", env, "pull", "qa/update:Q8_0");
            assertEquals(0, repaired.status(), repaired.err());
            assertEquals(shadow.toString(), repaired.out().strip());
            assertEquals(
                    "after!", Files.readString(ModelStore.of(flatRoot).resolve("qa/update:Q8_0")));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void jarWithoutVectorModuleStillOffersHelpAndExplainsHowToRunModels() throws Exception {
        var java = CliFixtures.javaCommand();
        java.remove("--add-modules");
        java.remove("jdk.incubator.vector");
        java.addAll(List.of("-jar", jar().toString()));
        assertEquals(0, run(java, "", "--help").status());
        Result inference = run(java, "", "chat", "-m", "missing.gguf");
        assertEquals(1, inference.status());
        assertTrue(inference.err().contains("--add-modules jdk.incubator.vector"), inference.err());
        assertFalse(inference.err().contains("NoClassDefFoundError"));
    }

    private record Result(int status, String out, String err) {}

    private Result run(List<String> executable, String input, String... args) throws Exception {
        return run(executable, input, Map.of(), args);
    }

    private Result run(
            List<String> executable, String input, Map<String, String> environment, String... args)
            throws Exception {
        var command = new ArrayList<>(executable);
        command.addAll(List.of(args));
        Path out = dir.resolve("stdout.txt"), err = dir.resolve("stderr.txt");
        ProcessBuilder builder =
                new ProcessBuilder(command)
                        .redirectOutput(out.toFile())
                        .redirectError(err.toFile());
        builder.environment().put("JINFER_OFFLINE", "1");
        builder.environment().put("JINFER_MODELS", dir.resolve("cache").toString());
        builder.environment().put("HF_HUB_CACHE", dir.resolve("hf").toString());
        builder.environment().put("HF_HOME", dir.resolve("hf-home").toString());
        builder.environment().putAll(environment);
        Process process = builder.start();
        try {
            try (var stdin = process.getOutputStream()) {
                stdin.write(input.getBytes(StandardCharsets.UTF_8));
            }
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "CLI timed out: " + command);
            return new Result(process.exitValue(), Files.readString(out), Files.readString(err));
        } finally {
            process.destroyForcibly();
        }
    }
}
