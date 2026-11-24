package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.vhlo.VhloOpSchemas;
import java.util.List;
import java.util.Map;

/**
 * Layout of operation properties in bytecode: ODS-generated {@code readProperties} reads the
 * inherent attributes sorted by name, required ones as an attribute index, optional ones as a
 * flagged index (absent when the flag is clear).
 */
final class PropertySchemas {

    record Field(String name, boolean optional) {}

    private PropertySchemas() {}

    private static Field req(String name) {
        return new Field(name, false);
    }

    private static Field opt(String name) {
        return new Field(name, true);
    }

    /**
     * A trailing optional property seen in bytecode written by jaxlib 0.12.1 after the {@code
     * func.func} / {@code builtin.module} attributes (ODS appends native properties after the
     * sorted attributes); its name and meaning are unknown, so it is accepted only when absent.
     */
    static final String UNKNOWN = "<unknown trailing property>";

    /**
     * Builtin and func ops, newest layout first: the LLVM of jaxlib 0.12.1 writes one more (absent)
     * trailing property than older ones. Only layouts seen in real files are listed: a layout that
     * decodes cleanly can still give the values the wrong names (newer LLVM sources declare an
     * optional {@code dlti} attribute sorted among the others, which reads jaxlib's {@code
     * sym_name} as {@code dlti}), so add one only together with a sample written by that LLVM.
     */
    private static final Map<String, List<List<Field>>> BUILTIN_AND_FUNC =
            Map.of(
                    "builtin.module",
                    List.of(
                            List.of(opt("sym_name"), opt("sym_visibility"), opt(UNKNOWN)),
                            List.of(opt("sym_name"), opt("sym_visibility"))),
                    "func.func",
                    List.of(
                            List.of(
                                    opt("arg_attrs"),
                                    req("function_type"),
                                    opt("no_inline"),
                                    opt("res_attrs"),
                                    req("sym_name"),
                                    opt("sym_visibility"),
                                    opt(UNKNOWN)),
                            List.of(
                                    opt("arg_attrs"),
                                    req("function_type"),
                                    opt("no_inline"),
                                    opt("res_attrs"),
                                    req("sym_name"),
                                    opt("sym_visibility"))),
                    "func.call",
                    List.of(
                            List.of(
                                    opt("arg_attrs"),
                                    req("callee"),
                                    opt("no_inline"),
                                    opt("res_attrs"))),
                    "func.return",
                    List.of(List.of()));

    /**
     * The layout the writer uses for {@code opName}, or null when unknown: the oldest one, which
     * every reader of the formats written here (portable artifacts) understands.
     */
    static List<Field> of(String opName) {
        List<List<Field>> all = layouts(opName);
        return all == null ? null : all.getLast();
    }

    /** Every known layout of {@code opName}, newest first, or null when unknown. */
    static List<List<Field>> layouts(String opName) {
        return layouts(opName, null);
    }

    /**
     * The layouts to try for {@code opName}, newest first, or null when unknown: for a {@code
     * stablehlo} op written by a known StableHLO version, the layout of that version only.
     */
    static List<List<Field>> layouts(
            String opName, com.qxotic.jota.stablehlo.portable.StableHloVersion stablehlo) {
        List<List<Field>> fixed = BUILTIN_AND_FUNC.get(opName);
        if (fixed != null) {
            return fixed;
        }
        if (opName.startsWith("vhlo.")) {
            VhloOpSchemas.Schema s = VhloOpSchemas.get(opName.substring(5));
            if (s != null) {
                return List.of(s.attributes().stream().map(a -> new Field(a, false)).toList());
            }
        }
        if (opName.startsWith("stablehlo.") || opName.startsWith("chlo.")) {
            if (stablehlo == null) {
                return StablehloPropertySchemas.all(opName);
            }
            List<Field> s = StablehloPropertySchemas.get(opName, stablehlo);
            if (s != null) {
                return List.of(s);
            }
        }
        return null;
    }
}
