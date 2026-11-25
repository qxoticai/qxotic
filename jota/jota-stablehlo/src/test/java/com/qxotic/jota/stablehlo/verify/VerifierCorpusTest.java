package com.qxotic.jota.stablehlo.verify;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.Samples;
import com.qxotic.jota.stablehlo.text.MlirParser;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * The verifier against the StableHLO repository: valid programs verify clean, and the {@code
 * expected-error} chunks of {@code stablehlo/tests} are rejected. Runs only when {@code
 * -Dstablehlo.repo=/path/to/openxla/stablehlo} is given.
 */
@EnabledIfSystemProperty(named = "stablehlo.repo", matches = ".+")
class VerifierCorpusTest {

    private static final Pattern EXPECTED =
            Pattern.compile("expected-error(?:@([+-]\\d+|above|below))?\\s*\\{\\{");

    private static Path repo() {
        return Path.of(System.getProperty("stablehlo.repo")).resolve("stablehlo");
    }

    /** Valid programs (testdata, interpreter tests, error-free test chunks) have no diagnostic. */
    @Test
    void validProgramsVerifyClean() {
        List<Path> files = new ArrayList<>();
        files.addAll(Samples.list(repo().resolve("testdata"), ".mlir"));
        files.addAll(Samples.list(repo().resolve("tests"), ".mlir"));
        files.addAll(Samples.list(repo().resolve("tests/interpret"), ".mlir"));
        files.addAll(Samples.list(repo().resolve("tests/vhlo"), ".mlir"));
        int programs = 0;
        List<String> failures = new ArrayList<>();
        for (Path file : files) {
            for (String chunk : MlirParser.splitInputFile(Samples.read(file))) {
                if (chunk.contains("expected-error") || chunk.contains("expected-warning")) {
                    continue;
                }
                var program = tryParse(chunk, file);
                if (program == null) {
                    continue;
                }
                programs++;
                for (Diagnostic d : Verifier.verify(program)) {
                    failures.add(repo().relativize(file) + ": " + d);
                }
            }
        }
        failures.stream()
                .map(s -> s.replaceAll("[0-9]+", "N"))
                .distinct()
                .forEach(System.out::println);
        System.out.println("valid programs: " + programs + ", false positives: " + failures.size());
        assertTrue(programs > 2500, "corpus not found under " + repo());
        assertTrue(failures.isEmpty(), failures.size() + " diagnostics on valid programs");
    }

    /**
     * Return-type inference on the valid programs: every inferred type is compatible with the
     * actual one (also checked by {@link #validProgramsVerifyClean}), and, for ops whose operands
     * all have static shapes, as specific: a static result dimension is inferred static, except
     * where the inputs leave it open (non-constant shape operands, set_dimension_size, branches of
     * different sizes). With dynamic operands a static result is a legal refinement.
     */
    @Test
    void inferenceReproducesResultTypes() {
        List<Path> files = new ArrayList<>();
        files.addAll(Samples.list(repo().resolve("testdata"), ".mlir"));
        files.addAll(Samples.list(repo().resolve("tests"), ".mlir"));
        files.addAll(Samples.list(repo().resolve("tests/interpret"), ".mlir"));
        Map<String, int[]> perOp = new TreeMap<>();
        List<String> lessSpecific = new ArrayList<>();
        for (Path file : files) {
            for (String chunk : MlirParser.splitInputFile(Samples.read(file))) {
                if (chunk.contains("expected-error") || chunk.contains("expected-warning")) {
                    continue;
                }
                var program = tryParse(chunk, file);
                if (program == null) {
                    continue;
                }
                program.operation()
                        .walk(
                                op -> {
                                    var inferred = TypeInference.infer(op).orElse(null);
                                    if (inferred == null) {
                                        return;
                                    }
                                    int[] c = perOp.computeIfAbsent(op.name(), k -> new int[2]);
                                    c[0]++;
                                    boolean staticInputs =
                                            op.operandTypes().stream()
                                                    .allMatch(
                                                            t ->
                                                                    !(t
                                                                                    instanceof
                                                                                    com.qxotic.jota
                                                                                                    .stablehlo
                                                                                                    .type
                                                                                                    .TensorType
                                                                                            tt)
                                                                            || tt.hasStaticShape());
                                    for (int i = 0; staticInputs && i < inferred.size(); i++) {
                                        if (inferred.get(i).type()
                                                        instanceof
                                                        com.qxotic.jota.stablehlo.type.TensorType t
                                                && op.resultTypes().get(i)
                                                        instanceof
                                                        com.qxotic.jota.stablehlo.type.TensorType a
                                                && !asSpecific(t, a)) {
                                            c[1]++;
                                            lessSpecific.add(
                                                    op.name() + ": " + t.mlir() + " vs "
                                                            + a.mlir());
                                        }
                                    }
                                });
            }
        }
        int ops = perOp.values().stream().mapToInt(c -> c[0]).sum();
        System.out.println("inferred ops: " + ops + " in " + perOp.size() + " kinds");
        perOp.forEach(
                (k, v) -> {
                    if (v[1] > 0) {
                        System.out.println("  less specific " + v[1] + "/" + v[0] + " " + k);
                    }
                });
        lessSpecific.stream()
                .map(s -> s.replaceAll("[0-9]+", "N"))
                .distinct()
                .limit(40)
                .forEach(s -> System.out.println("    " + s));
        assertTrue(ops > 20_000, "only " + ops + " ops inferred");
        assertTrue(
                lessSpecific.stream()
                        .allMatch(
                                s ->
                                        s.startsWith("stablehlo.set_dimension_size")
                                                || s.startsWith("stablehlo.dynamic_gather")
                                                || s.startsWith("stablehlo.rng")
                                                || s.startsWith("stablehlo.if")
                                                || s.startsWith("stablehlo.case")),
                "inference less specific than the result type");
    }

    private static boolean asSpecific(
            com.qxotic.jota.stablehlo.type.TensorType inferred,
            com.qxotic.jota.stablehlo.type.TensorType actual) {
        for (int d = 0; d < actual.rank(); d++) {
            if (actual.dim(d) != com.qxotic.jota.stablehlo.type.TensorType.DYNAMIC
                    && inferred.dim(d) != actual.dim(d)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Chunks marked {@code expected-error} on a StableHLO operation that the parser accepts: the
     * verifier rejects at least {@link #MIN_CAUGHT} of them (and reports how many on the very line
     * the test marks). {@code -Dstablehlo.verify.misses} prints the expected message of each miss.
     */
    @Test
    void invalidProgramsAreRejected() {
        int total = 0;
        int caught = 0;
        int caughtOnTarget = 0;
        Map<String, int[]> perOp = new TreeMap<>();
        List<String> wrongOp = new ArrayList<>();
        for (Path file : Samples.list(repo().resolve("tests"), ".mlir")) {
            for (String chunk : MlirParser.splitInputFile(Samples.read(file))) {
                String[] lines = chunk.split("\n", -1);
                int target = -1;
                for (int i = 0; i < lines.length; i++) {
                    Matcher m = EXPECTED.matcher(lines[i]);
                    if (m.find()) {
                        target = i + 1 + offset(m.group(1));
                        break;
                    }
                }
                if (target < 1 || target > lines.length) {
                    continue;
                }
                Matcher name =
                        Pattern.compile("\"?(stablehlo\\.[a-z_0-9]+)").matcher(lines[target - 1]);
                if (!name.find()) {
                    continue;
                }
                MlirParser.Parsed parsed;
                try {
                    parsed = MlirParser.parseWithLines(chunk, file.toString());
                } catch (RuntimeException e) {
                    continue; // rejected while parsing
                }
                total++;
                int[] counts = perOp.computeIfAbsent(name.group(1), k -> new int[2]);
                counts[0]++;
                List<Diagnostic> diags = Verifier.verify(parsed.program());
                final int t = target;
                boolean onTarget = diags.stream().anyMatch(d -> parsed.line(d.operation()) == t);
                if (!diags.isEmpty()) {
                    caught++;
                }
                if (onTarget) {
                    caughtOnTarget++;
                    counts[1]++;
                } else {
                    Matcher msg = Pattern.compile("\\{\\{(.*?)\\}\\}").matcher(chunk);
                    if (System.getProperty("stablehlo.verify.misses") != null && msg.find()) {
                        System.out.println(
                                "MISS "
                                        + name.group(1)
                                        + " "
                                        + repo().relativize(file)
                                        + ":"
                                        + t
                                        + " | "
                                        + msg.group(1));
                    }
                }
                if (!onTarget && !diags.isEmpty()) {
                    wrongOp.add(repo().relativize(file) + ":" + t + " " + diags.getFirst());
                }
            }
        }
        System.out.println(
                "invalid programs: "
                        + total
                        + ", rejected: "
                        + caught
                        + ", on the marked line: "
                        + caughtOnTarget);
        perOp.forEach(
                (k, v) -> {
                    if (v[1] < v[0])
                        System.out.println("  missed " + (v[0] - v[1]) + "/" + v[0] + " " + k);
                });
        wrongOp.stream().limit(30).forEach(s -> System.out.println("  elsewhere: " + s));
        assertTrue(caught >= MIN_CAUGHT * total, caught + " of " + total + " caught");
    }

    /**
     * Fraction rejected (StableHLO at 1.27: 559 of 591, the rest are mostly parser-level or
     * tuple-layout rules).
     */
    static final double MIN_CAUGHT = 1.0; // every invalid program the parser accepts

    private static int offset(String g) {
        if (g == null) {
            return 0;
        }
        return switch (g) {
            case "above" -> -1;
            case "below" -> 1;
            default -> Integer.parseInt(g);
        };
    }

    private static com.qxotic.jota.stablehlo.ir.Program tryParse(String chunk, Path file) {
        try {
            return MlirParser.parse(chunk, file.toString());
        } catch (RuntimeException e) {
            return null;
        }
    }
}
