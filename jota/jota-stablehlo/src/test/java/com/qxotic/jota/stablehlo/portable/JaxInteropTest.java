package com.qxotic.jota.stablehlo.portable;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import com.qxotic.jota.stablehlo.vhlo.VhloException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;

/**
 * Cross-checks the writers against StableHLO itself through jaxlib: artifacts written here are read
 * by {@code jax.extend.mlir.deserialize_portable_artifact}, and programs printed here (pretty form)
 * are parsed and verified by MLIR; both are printed by MLIR, parsed back and compared with the
 * source programs. Enabled with {@code -Dstablehlo.jax.python=/path/to/python} (a Python with
 * {@code jax} and {@code jaxlib} installed); the target version is {@code -Dstablehlo.jax.target}
 * (default 1.0.0: it must not be newer than jaxlib's StableHLO).
 */
@EnabledIfSystemProperty(named = "stablehlo.jax.python", matches = ".+")
class JaxInteropTest {

    private static final String DUMP =
            """
            import sys
            from jax.extend import mlir as jm
            from jax._src.interpreters import mlir as jmlir
            for path in sys.argv[1:]:
                with jmlir.make_ir_context():
                    module = jm.deserialize_portable_artifact(open(path, 'rb').read())
                    open(path + '.mlir', 'w').write(str(module))
            """;

    @Test
    void stableHloReadsWhatWeWrite(@TempDir Path dir) throws Exception {
        StableHloVersion target =
                StableHloVersion.parse(System.getProperty("stablehlo.jax.target", "1.0.0"));
        List<Path> artifacts = new ArrayList<>();
        List<Program> sources = new ArrayList<>();
        for (Path file : Samples.files("text/testdata", ".mlir")) {
            Program p = MlirParser.parse(file);
            try {
                Path out = dir.resolve(file.getFileName() + ".bc");
                Files.write(out, PortableArtifact.serialize(p, target));
                artifacts.add(out);
                sources.add(p);
            } catch (VhloException e) {
                // not expressible at the target
            }
        }
        assertTrue(artifacts.size() > 50, "serialized " + artifacts.size());
        run(dir, DUMP, artifacts);
        for (int i = 0; i < artifacts.size(); i++) {
            Program viaStableHlo =
                    parse(
                            artifacts
                                    .get(i)
                                    .resolveSibling(artifacts.get(i).getFileName() + ".mlir"));
            PortableArtifactSerializeTest.assertSameProgram(sources.get(i), viaStableHlo);
        }
    }

    private static final String PARSE =
            """
            import sys
            from jax._src.interpreters import mlir as jmlir
            from jax._src.lib.mlir import ir
            for path in sys.argv[1:]:
                with jmlir.make_ir_context() as context:
                    context.allow_unregistered_dialects = True
                    data = open(path, 'rb').read()  # text, or bytecode (.mlirbc)
                    module = ir.Module.parse(data if path.endswith('.mlirbc') else data.decode())
                    assert module.operation.verify()
                    open(path + '.mlir', 'w').write(str(module))
            """;

    /** MLIR parses and verifies the pretty form printed here, and means the same program. */
    @Test
    void stableHloParsesOurPrettyForm(@TempDir Path dir) throws Exception {
        checkReadByStableHlo(dir, ".pretty", p -> MlirPrinter.print(p).getBytes());
    }

    /**
     * MLIR reads the plain bytecode written here, both with attribute dictionaries (bytecode
     * version 4) and with native properties in the layouts of jaxlib's StableHLO (version 6).
     */
    @Test
    void stableHloReadsOurPlainBytecode(@TempDir Path dir) throws Exception {
        StableHloVersion jaxlib = StableHloVersion.parse("1.25.3");
        for (long version : new long[] {4, 6}) {
            checkReadByStableHlo(
                    dir.resolve("v" + version),
                    ".mlirbc",
                    p ->
                            com.qxotic.jota.stablehlo.bytecode.MlirBytecodeWriter.writePlain(
                                    p, version, jaxlib, "jota-stablehlo"));
        }
    }

    /**
     * Writes every bundled testdata / interpreter program with {@code writer}, has MLIR (jaxlib)
     * read and verify it and print it, and compares the print with the source program.
     */
    private static void checkReadByStableHlo(
            Path dir, String suffix, java.util.function.Function<Program, byte[]> writer)
            throws Exception {
        Files.createDirectories(dir);
        List<Path> written = new ArrayList<>();
        List<Program> sources = new ArrayList<>();
        for (String samples : List.of("text/testdata", "text/interpret")) {
            for (Path file : Samples.files(samples, ".mlir")) {
                List<String> chunks = MlirParser.splitInputFile(Samples.read(file));
                for (int c = 0; c < chunks.size(); c++) {
                    // jaxlib 0.12.1 rejects the quantized interpreter tests even as written
                    // ("result #0 must be ranked tensor of ... quantized values")
                    if (Samples.isNegativeOrEmpty(chunks.get(c))
                            || chunks.get(c).contains("!quant.")) {
                        continue;
                    }
                    Program p = MlirParser.parse(chunks.get(c), file.toString());
                    Path out = dir.resolve(file.getFileName() + "." + c + suffix);
                    Files.write(out, writer.apply(p));
                    written.add(out);
                    sources.add(p);
                }
            }
        }
        run(dir, PARSE, written);
        for (int i = 0; i < written.size(); i++) {
            Path back = written.get(i).resolveSibling(written.get(i).getFileName() + ".mlir");
            PortableArtifactSerializeTest.assertSameProgram(sources.get(i), parse(back));
        }
    }

    private static void run(Path dir, String python, List<Path> files) throws Exception {
        Path script = dir.resolve("script.py");
        Files.writeString(script, python);
        List<String> command = new ArrayList<>();
        command.add(System.getProperty("stablehlo.jax.python"));
        command.add("-I");
        command.add(script.toString());
        files.forEach(f -> command.add(f.toString()));
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        String output = new String(process.getInputStream().readAllBytes());
        assertTrue(process.waitFor(10, TimeUnit.MINUTES));
        assertEquals(0, process.exitValue(), output);
    }

    private static Program parse(Path file) throws IOException {
        return MlirParser.parse(file);
    }
}
