package com.qxotic.jota.runtime.mojo.codegen;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/** CLI tool that writes generated Mojo source to disk. */
public final class MojoKernelSourceCli {

    private MojoKernelSourceCli() {}

    public static void main(String[] args) throws IOException {
        Map<String, String> values = parseArgs(args);
        String operation = require(values, "operation").toLowerCase(Locale.ROOT);
        String dtype = values.getOrDefault("dtype", "f32").toLowerCase(Locale.ROOT);
        int rank = Integer.parseInt(values.getOrDefault("rank", "1"));
        String cacheKey = values.getOrDefault("cache-key", "mojo_codegen_v1");
        Path output = Path.of(require(values, "output"));

        if (!"f32".equals(dtype)) {
            throw new IllegalArgumentException("Unsupported dtype: " + dtype + " (only f32)");
        }
        boolean lhsB0 = parseBoolean(values.getOrDefault("lhs-broadcast0", "false"));
        boolean lhsB1 = parseBoolean(values.getOrDefault("lhs-broadcast1", "false"));
        boolean rhsB0 = parseBoolean(values.getOrDefault("rhs-broadcast0", "false"));
        boolean rhsB1 = parseBoolean(values.getOrDefault("rhs-broadcast1", "false"));

        String kernelName = kernelNameFor(operation, rank, cacheKey);
        String source =
                rank == 1
                        ? renderRank1(operation, kernelName)
                        : renderRank2(operation, kernelName, lhsB0, lhsB1, rhsB0, rhsB1);

        Files.createDirectories(output.getParent());
        Files.writeString(output, source);
    }

    private static boolean parseBoolean(String value) {
        return switch (value.toLowerCase()) {
            case "1", "true", "yes", "on" -> true;
            case "0", "false", "no", "off" -> false;
            default -> throw new IllegalArgumentException("Invalid boolean value: " + value);
        };
    }

    private static String kernelNameFor(String operation, int rank, String cacheKey) {
        String prefix =
                sanitizeIdentifier((operation + "_f32_" + rank + "d").toLowerCase(Locale.ROOT));
        String hash = cacheKey.length() <= 12 ? cacheKey : cacheKey.substring(0, 12);
        return "jota_" + prefix + "_" + sanitizeIdentifier(hash);
    }

    private static String sanitizeIdentifier(String text) {
        StringBuilder out = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if ((c >= 'a' && c <= 'z')
                    || (c >= 'A' && c <= 'Z')
                    || (c >= '0' && c <= '9')
                    || c == '_') {
                out.append(c);
            } else {
                out.append('_');
            }
        }
        return out.toString();
    }

    private static String renderRank1(String operation, String kernelName) {
        String expr = expression(operation, "idx", "idx");
        boolean unary = "relu".equals(operation);
        return "from std.gpu import global_idx\n\n\n"
                + "fn "
                + kernelName
                + "(\n"
                + "    lhs: UnsafePointer[Float32, MutAnyOrigin],\n"
                + (unary ? "" : "    rhs: UnsafePointer[Float32, MutAnyOrigin],\n")
                + "    out_ptr: UnsafePointer[Float32, MutAnyOrigin],\n"
                + "    n: Int,\n"
                + "):"
                + "\n"
                + "    idx = global_idx.x\n"
                + "    if idx < UInt(n):\n"
                + "        out_ptr[idx] = "
                + expr
                + "\n";
    }

    private static String renderRank2(
            String operation,
            String kernelName,
            boolean lhsB0,
            boolean lhsB1,
            boolean rhsB0,
            boolean rhsB1) {
        boolean unary = "relu".equals(operation);
        String expr = expression(operation, "lhs_offset", "rhs_offset");
        return "from std.gpu import global_idx\n\n\n"
                + "fn "
                + kernelName
                + "(\n"
                + "    lhs: UnsafePointer[Float32, MutAnyOrigin],\n"
                + (unary ? "" : "    rhs: UnsafePointer[Float32, MutAnyOrigin],\n")
                + "    out_ptr: UnsafePointer[Float32, MutAnyOrigin],\n"
                + "    rows: Int,\n"
                + "    cols: Int,\n"
                + "    lhs_stride0: Int,\n"
                + "    lhs_stride1: Int,\n"
                + (unary ? "" : "    rhs_stride0: Int,\n")
                + (unary ? "" : "    rhs_stride1: Int,\n")
                + "    out_stride0: Int,\n"
                + "    out_stride1: Int,\n"
                + "):"
                + "\n"
                + "    idx = global_idx.x\n"
                + "    total = UInt(rows) * UInt(cols)\n"
                + "    if idx < total:\n"
                + "        col_count = UInt(cols)\n"
                + "        row = idx / col_count\n"
                + "        col = idx % col_count\n"
                + "        lhs_row = UInt(0) if "
                + boolLiteral(lhsB0)
                + " else row\n"
                + "        lhs_col = UInt(0) if "
                + boolLiteral(lhsB1)
                + " else col\n"
                + "        lhs_offset = lhs_row * UInt(lhs_stride0) + lhs_col * UInt(lhs_stride1)\n"
                + "        out_offset = row * UInt(out_stride0) + col * UInt(out_stride1)\n"
                + (unary
                        ? ""
                        : ("        rhs_row = UInt(0) if "
                                + boolLiteral(rhsB0)
                                + " else row\n"
                                + "        rhs_col = UInt(0) if "
                                + boolLiteral(rhsB1)
                                + " else col\n"
                                + "        rhs_offset = rhs_row * UInt(rhs_stride0) + rhs_col *"
                                + " UInt(rhs_stride1)\n"))
                + "        out_ptr[out_offset] = "
                + expr
                + "\n";
    }

    private static String expression(String operation, String lhsIndex, String rhsIndex) {
        return switch (operation) {
            case "add" -> "lhs[" + lhsIndex + "] + rhs[" + rhsIndex + "]";
            case "sub" -> "lhs[" + lhsIndex + "] - rhs[" + rhsIndex + "]";
            case "mul" -> "lhs[" + lhsIndex + "] * rhs[" + rhsIndex + "]";
            case "div" -> "lhs[" + lhsIndex + "] / rhs[" + rhsIndex + "]";
            case "relu" -> "max(lhs[" + lhsIndex + "], 0.0)";
            default -> throw new IllegalArgumentException("Unsupported operation: " + operation);
        };
    }

    private static String boolLiteral(boolean value) {
        return value ? "True" : "False";
    }

    private static String require(Map<String, String> values, String key) {
        String value = values.get(key);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Missing required arg --" + key);
        }
        return value;
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> values = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (!arg.startsWith("--")) {
                throw new IllegalArgumentException("Expected option starting with --, got: " + arg);
            }
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("Missing value for option: " + arg);
            }
            values.put(arg.substring(2), args[++i]);
        }
        return values;
    }
}
