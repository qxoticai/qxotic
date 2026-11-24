package com.qxotic.jota.runtime.mojo.codegen;

import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The Mojo language variant that generated sources must target.
 *
 * <p>Mojo changed incompatibly between the releases this backend supports, so the generators pick
 * their syntax from the detected compiler version:
 *
 * <ul>
 *   <li>{@link #V0_26_1}: kernels declared with {@code fn}, {@code def main()} without {@code
 *       raises} (a {@code def} already raises), {@code compile_function[f, f]}.
 *   <li>{@link #V0_26_2}: {@code def main()} must say {@code raises} to call raising functions.
 *   <li>{@link #V1}: Mojo 1.x removed {@code fn} (use {@code def}), moved the GPU modules from
 *       {@code std.gpu} to {@code max.gpu} (shipped by the {@code max-mojo-libs} package), and
 *       {@code compile_function} takes the kernel once.
 * </ul>
 *
 * <p>All dialects declare locals with {@code var} and import math through {@code std.math}; every
 * supported release accepts both.
 */
public enum MojoDialect {
    V0_26_1("std.gpu", "fn", "def main():", false),
    V0_26_2("std.gpu", "fn", "def main() raises:", false),
    V1("max.gpu", "def", "def main() raises:", true);

    /** The dialect used when the compiler version is unknown. */
    public static final MojoDialect LATEST = V1;

    private static final Pattern VERSION = Pattern.compile("(?i)\\bmojo\\s+(\\d+)\\.(\\d+)");
    private static final Pattern NUMERIC = Pattern.compile("^\\s*(\\d+)\\.(\\d+)");

    private final String gpuModule;
    private final String kernelKeyword;
    private final String mainSignature;
    private final boolean singleCompileFunctionParameter;

    MojoDialect(
            String gpuModule,
            String kernelKeyword,
            String mainSignature,
            boolean singleCompileFunctionParameter) {
        this.gpuModule = gpuModule;
        this.kernelKeyword = kernelKeyword;
        this.mainSignature = mainSignature;
        this.singleCompileFunctionParameter = singleCompileFunctionParameter;
    }

    /**
     * Module holding {@code thread_idx}, {@code block_idx}, ... ({@code std.gpu} or {@code
     * max.gpu}).
     */
    public String gpuModule() {
        return gpuModule;
    }

    /** Keyword that declares a GPU kernel function ({@code fn} or {@code def}). */
    public String kernelKeyword() {
        return kernelKeyword;
    }

    /** Header line of the host {@code main} in the compile wrapper. */
    public String mainSignature() {
        return mainSignature;
    }

    /** The {@code DeviceContext.compile_function} call that compiles {@code fnName}. */
    public String compileFunctionCall(String contextVar, String fnName) {
        String params = singleCompileFunctionParameter ? fnName : fnName + ", " + fnName;
        return contextVar + ".compile_function[" + params + "]()";
    }

    /**
     * Picks the dialect from {@code mojo --version} output (for example {@code "Mojo 0.26.1.0
     * (156d3ac6)"}) or from a bare version such as {@code "1.2"}. Returns empty when no version can
     * be read.
     */
    public static Optional<MojoDialect> fromVersionText(String text) {
        if (text == null) {
            return Optional.empty();
        }
        Matcher m = VERSION.matcher(text);
        if (!m.find()) {
            m = NUMERIC.matcher(text);
            if (!m.find()) {
                return Optional.empty();
            }
        }
        return Optional.of(
                forVersion(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)), text));
    }

    private static MojoDialect forVersion(int major, int minor, String text) {
        if (major == 0) {
            if (minor > 26) {
                return V0_26_2;
            }
            if (minor == 26) {
                return isAtLeastPatch(text, 2) ? V0_26_2 : V0_26_1;
            }
            return V0_26_1;
        }
        if (major < 20) {
            return V1;
        }
        // Year-based releases (25.x and earlier) predate 0.26; the closest syntax is the oldest.
        return V0_26_1;
    }

    private static boolean isAtLeastPatch(String text, int patch) {
        Matcher m = Pattern.compile("\\b0\\.26\\.(\\d+)").matcher(text);
        return m.find() && Integer.parseInt(m.group(1)) >= patch;
    }
}
