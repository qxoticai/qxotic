package com.qxotic.jinfer.libjinfer;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.graalvm.nativeimage.c.CContext;
import org.graalvm.nativeimage.c.struct.CField;
import org.graalvm.nativeimage.c.struct.CStruct;
import org.graalvm.nativeimage.c.struct.SizeOf;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CCharPointerPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.word.PointerBase;
import org.graalvm.word.UnsignedWord;

/** jinfer_model_options, read through the header itself so the layout can never drift from it. */
@CContext(Options.Directives.class)
final class Options {

    /**
     * The header is the single source of the struct layout; the build names its directory and,
     * since jinfer.h includes toknroll.h, that header's as well (path-separated).
     */
    public static final class Directives implements CContext.Directives {
        private static String[] includes() {
            String include = System.getProperty("jinfer.native.include");
            if (include == null)
                throw new IllegalStateException(
                        "-Djinfer.native.include=<dir of jinfer.h>:<dir of toknroll.h> is"
                                + " required");
            return include.split(java.io.File.pathSeparator);
        }

        @Override
        public List<String> getHeaderFiles() {
            return Collections.singletonList("\"" + includes()[0] + "/jinfer.h\"");
        }

        @Override
        public List<String> getOptions() {
            List<String> options = new java.util.ArrayList<>();
            for (String dir : includes()) options.add("-I" + dir);
            return options;
        }
    }

    @CStruct("jinfer_model_options")
    interface Struct extends PointerBase {
        @CField("struct_size")
        UnsignedWord structSize();

        @CField("companions")
        CCharPointerPointer companions();

        @CField("companion_count")
        int companionCount();

        @CField("prompt_cache")
        CCharPointer promptCache();

        @CField("context_capacity")
        int contextCapacity();
    }

    /** The Java-side copy; raw pointers because word types may not live in object fields. */
    record Values(Map<String, String> companions, String promptCache, int contextCapacity) {
        static final Values DEFAULTS = new Values(Map.of(), null, 0);
    }

    static Values read(Struct s) {
        if (s.isNull()) return Values.DEFAULTS;
        long size = s.structSize().rawValue();
        if (size < SizeOf.get(Struct.class)) {
            throw Errors.invalid(
                    "options.struct_size "
                            + size
                            + " is too small: call jinfer_model_options_init first");
        }
        Map<String, String> companions = new LinkedHashMap<>();
        CCharPointerPointer list = s.companions();
        int count = s.companionCount();
        if (count < 0 || (count > 0 && list.isNull())) {
            throw Errors.invalid(
                    "options.companion_count " + count + " does not match options.companions");
        }
        for (int i = 0; i < count; i++) {
            if (list.read(i).isNull()) throw Errors.invalid("companions[" + i + "] is NULL");
            String entry = CTypeConversion.toJavaString(list.read(i));
            int eq = entry.indexOf('=');
            if (eq <= 0 || eq == entry.length() - 1) {
                throw Errors.invalid(
                        "companions["
                                + i
                                + "] must read \"<capability>=<ref>\", got \""
                                + entry
                                + "\"");
            }
            companions.put(entry.substring(0, eq), entry.substring(eq + 1));
        }
        String promptCache =
                s.promptCache().isNull() ? null : CTypeConversion.toJavaString(s.promptCache());
        return new Values(companions, promptCache, s.contextCapacity());
    }

    private Options() {}
}
