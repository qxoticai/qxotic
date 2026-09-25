package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ModelProvider;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.TimeUnit;
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
            assertTrue(rejected.err().contains("Unknown option: " + removed));
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
        var command = new ArrayList<>(executable);
        command.addAll(List.of(args));
        Path out = dir.resolve("stdout.txt"), err = dir.resolve("stderr.txt");
        ProcessBuilder builder =
                new ProcessBuilder(command)
                        .redirectOutput(out.toFile())
                        .redirectError(err.toFile());
        builder.environment().put("JINFER_OFFLINE", "1");
        builder.environment().put("JINFER_MODELS", dir.resolve("cache").toString());
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
