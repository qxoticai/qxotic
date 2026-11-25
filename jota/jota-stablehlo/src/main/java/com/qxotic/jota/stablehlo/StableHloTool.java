package com.qxotic.jota.stablehlo;

import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeReader;
import com.qxotic.jota.stablehlo.bytecode.MlirBytecodeWriter;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.portable.BytecodeHeader;
import com.qxotic.jota.stablehlo.portable.PortableArtifact;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import com.qxotic.jota.stablehlo.tir.StableHloToTirConverter;
import com.qxotic.jota.stablehlo.tir.TirConversion;
import com.qxotic.jota.stablehlo.verify.Diagnostic;
import com.qxotic.jota.stablehlo.verify.Verifier;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Command-line tool over the readers, writers, verifier and TIR converter of this module.
 *
 * <pre>
 * info      FILE                                   format, versions, functions, op counts
 * print     FILE [--generic] [-o OUT]              MLIR text (pretty, or generic form)
 * convert   FILE --to FORMAT -o OUT [--target X.Y.Z] [--bytecode-version N]
 *           FORMAT: text | generic | portable (VHLO artifact) | bytecode (plain MLIR bytecode)
 * verify    FILE                                   spec constraints; exit status 1 on a diagnostic
 * to-tir    FILE [--function NAME]                 the function as a jota TIR graph
 * </pre>
 *
 * The input format is detected from its content: a StableHLO portable artifact, plain MLIR
 * bytecode, or MLIR text. Exit status: 0 on success, 1 on a failure or diagnostics, 2 on a usage
 * error.
 */
public final class StableHloTool {

    static final String USAGE =
            """
            usage: stablehlo-tool COMMAND FILE [options]
              info      FILE                       format, versions, functions, op counts
              print     FILE [--generic] [-o OUT]  MLIR text, pretty or generic form
              convert   FILE --to FORMAT -o OUT [--target X.Y.Z] [--bytecode-version N]
                        FORMAT: text | generic | portable | bytecode
                        portable: a VHLO artifact for StableHLO --target (default %s)
                        bytecode: plain MLIR bytecode for StableHLO --target; version 4 or older
                                  is readable by any StableHLO, 5 and later (default %d) only by
                                  one whose op property layouts match the target's
              verify    FILE                       check the spec constraints (exit 1 on errors)
              to-tir    FILE [--function NAME]     convert a function (default main) to jota TIR
            """
                    .formatted(StableHloVersion.CURRENT, MlirBytecodeReader.VERSION_CURRENT);

    private final PrintStream out;
    private final PrintStream err;

    private StableHloTool(PrintStream out, PrintStream err) {
        this.out = out;
        this.err = err;
    }

    public static void main(String[] args) {
        System.exit(run(args, System.out, System.err));
    }

    /** Runs one command; returns the exit status. */
    public static int run(String[] args, PrintStream out, PrintStream err) {
        StableHloTool tool = new StableHloTool(out, err);
        try {
            return tool.dispatch(args);
        } catch (UsageException e) {
            err.println("stablehlo-tool: " + e.getMessage());
            err.print(USAGE);
            return 2;
        } catch (java.nio.file.NoSuchFileException e) {
            err.println("stablehlo-tool: no such file: " + e.getFile());
            return 1;
        } catch (IOException | RuntimeException e) {
            err.println("stablehlo-tool: " + e);
            return 1;
        }
    }

    private static final class UsageException extends RuntimeException {
        UsageException(String message) {
            super(message);
        }
    }

    /** {@code COMMAND FILE} plus {@code --key value} / {@code --flag} options. */
    private record Args(String command, Path file, Map<String, String> options) {

        static Args parse(String[] args) {
            if (args.length == 0 || args[0].equals("-h") || args[0].equals("--help")) {
                throw new UsageException("missing command");
            }
            Path file = null;
            Map<String, String> options = new LinkedHashMap<>();
            for (int i = 1; i < args.length; i++) {
                String a = args[i];
                if (a.equals("--generic")) {
                    options.put(a, "true");
                } else if (a.startsWith("-")) {
                    if (i + 1 >= args.length) {
                        throw new UsageException("missing value for " + a);
                    }
                    options.put(a, args[++i]);
                } else if (file == null) {
                    file = Path.of(a);
                } else {
                    throw new UsageException("unexpected argument " + a);
                }
            }
            if (file == null) {
                throw new UsageException("missing input file");
            }
            return new Args(args[0], file, options);
        }

        String option(String name, String fallback) {
            return options.getOrDefault(name, fallback);
        }

        void allow(String... names) {
            for (String o : options.keySet()) {
                if (!List.of(names).contains(o)) {
                    throw new UsageException("unknown option " + o + " for " + command);
                }
            }
        }
    }

    private int dispatch(String[] argv) throws IOException {
        Args args = Args.parse(argv);
        return switch (args.command()) {
            case "info" -> {
                args.allow();
                yield info(args);
            }
            case "print" -> {
                args.allow("--generic", "-o");
                yield print(args);
            }
            case "convert" -> {
                args.allow("--to", "-o", "--target", "--bytecode-version");
                yield convert(args);
            }
            case "verify" -> {
                args.allow();
                yield verify(args);
            }
            case "to-tir" -> {
                args.allow("--function");
                yield toTir(args);
            }
            default -> throw new UsageException("unknown command " + args.command());
        };
    }

    // ================================================================== reading

    enum Format {
        TEXT,
        PORTABLE,
        BYTECODE
    }

    private record Input(Format format, Program program, MlirParser.Parsed parsed, byte[] data) {}

    private static Input read(Path file) throws IOException {
        byte[] data = Files.readAllBytes(file);
        if (BytecodeHeader.isBytecode(data)) {
            if (BytecodeHeader.read(data).isPortableArtifact()) {
                return new Input(Format.PORTABLE, PortableArtifact.deserialize(data), null, data);
            }
            return new Input(Format.BYTECODE, MlirBytecodeReader.read(data).program(), null, data);
        }
        MlirParser.Parsed parsed =
                MlirParser.parseWithLines(
                        new String(data, StandardCharsets.UTF_8), file.toString());
        return new Input(Format.TEXT, parsed.program(), parsed, data);
    }

    // ================================================================== commands

    private int info(Args args) throws IOException {
        Input in = read(args.file());
        out.println("format: " + describe(in));
        out.println("module: " + in.program().name().orElse("(unnamed)"));
        for (var f : in.program().functions()) {
            out.println("function: @" + f.symName());
        }
        Map<String, Integer> counts = new TreeMap<>();
        in.program().operation().walk(op -> counts.merge(op.name(), 1, Integer::sum));
        counts.forEach((name, n) -> out.println("  " + n + " " + name));
        return 0;
    }

    private static String describe(Input in) {
        return switch (in.format()) {
            case TEXT -> "MLIR text";
            case PORTABLE -> {
                BytecodeHeader h = BytecodeHeader.read(in.data());
                yield "StableHLO portable artifact (StableHLO "
                        + PortableArtifact.version(in.data())
                        + ", bytecode v"
                        + h.bytecodeVersion()
                        + ")";
            }
            case BYTECODE -> {
                MlirBytecodeReader.Result r = MlirBytecodeReader.read(in.data());
                byte[] v = r.dialectVersions().get("stablehlo");
                yield "MLIR bytecode v"
                        + r.bytecodeVersion()
                        + " (producer "
                        + r.producer()
                        + (v != null ? ", stablehlo dialect " + dialectVersion(v) : "")
                        + ")";
            }
        };
    }

    /** {@code major.minor.patch} of a stablehlo dialect version blob (three varints). */
    private static String dialectVersion(byte[] v) {
        List<Long> parts = new ArrayList<>();
        int i = 0;
        while (i < v.length && parts.size() < 3) {
            int b = v[i] & 0xFF;
            int bytes = Integer.numberOfTrailingZeros(b | 0x100) + 1;
            long value = 0;
            if (bytes == 9) {
                for (int k = 8; k >= 1; k--) {
                    value = (value << 8) | (v[i + k] & 0xFF);
                }
            } else {
                for (int k = bytes - 1; k >= 0; k--) {
                    value = (value << 8) | (v[i + k] & 0xFF);
                }
                value >>>= bytes;
            }
            parts.add(value);
            i += bytes;
        }
        return parts.stream().map(String::valueOf).reduce((a, b) -> a + "." + b).orElse("?");
    }

    private int print(Args args) throws IOException {
        Program p = read(args.file()).program();
        String text =
                args.options().containsKey("--generic")
                        ? MlirPrinter.printGeneric(p)
                        : MlirPrinter.print(p);
        String output = args.option("-o", null);
        if (output == null) {
            out.print(text);
        } else {
            Files.writeString(Path.of(output), text);
        }
        return 0;
    }

    private int convert(Args args) throws IOException {
        String to = args.option("--to", null);
        String output = args.option("-o", null);
        if (to == null || output == null) {
            throw new UsageException("convert needs --to FORMAT and -o OUT");
        }
        StableHloVersion target =
                StableHloVersion.parse(
                        args.option("--target", StableHloVersion.CURRENT.toString()));
        Program p = read(args.file()).program();
        byte[] bytes =
                switch (to) {
                    case "text" -> MlirPrinter.print(p).getBytes(StandardCharsets.UTF_8);
                    case "generic" -> MlirPrinter.printGeneric(p).getBytes(StandardCharsets.UTF_8);
                    case "portable" -> PortableArtifact.serialize(p, target);
                    case "bytecode" ->
                            MlirBytecodeWriter.writePlain(
                                    p,
                                    Long.parseLong(
                                            args.option(
                                                    "--bytecode-version",
                                                    Long.toString(
                                                            MlirBytecodeReader.VERSION_CURRENT))),
                                    target,
                                    "jota-stablehlo");
                    default -> throw new UsageException("unknown format " + to);
                };
        Files.write(Path.of(output), bytes);
        return 0;
    }

    private int verify(Args args) throws IOException {
        Input in = read(args.file());
        List<Diagnostic> diagnostics = Verifier.verify(in.program());
        for (Diagnostic d : diagnostics) {
            Operation op = d.operation();
            int line = in.parsed() == null || op == null ? -1 : in.parsed().line(op);
            out.println(args.file() + (line > 0 ? ":" + line : "") + ": " + d);
        }
        if (diagnostics.isEmpty()) {
            out.println(args.file() + ": ok");
            return 0;
        }
        return 1;
    }

    private int toTir(Args args) throws IOException {
        Program p = read(args.file()).program();
        TirConversion c = StableHloToTirConverter.convert(p, args.option("--function", "main"));
        out.println(c.graph());
        out.println("inputs: " + c.inputs());
        out.println("results: " + c.resultTypes());
        return 0;
    }
}
