package com.qxotic.jota.stablehlo.text;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * The SSA value names MLIR's printer gives ({@code SSANameState} of {@code AsmPrinter.cpp}).
 *
 * <p>Each region is numbered as a whole before the regions nested in it, which start from the
 * counters (value ids, argument ids, name-conflict ids) reached at the end of their parent region
 * and see the names used there. Entry block arguments are {@code %argN}, other values {@code %N}; a
 * name already used gets the suffix {@code _<conflict id>}.
 *
 * <p>In the pretty form, the {@code OpAsmOpInterface} names of the dialects apply: {@code
 * stablehlo.constant} results are {@code %c} (integer elements) or {@code %cst}, {@code
 * stablehlo.while} arguments {@code %iterArg}, {@code chlo.scan} arguments {@code %input} / {@code
 * %carry}, and ops with several named ODS results (rng_bit_generator, batch_norm_training /
 * batch_norm_grad, chlo.top_k) name each result.
 */
final class SsaNames {

    /** ODS result names of the ops whose tablegen'd {@code getAsmResultNames} names them. */
    private static final Map<String, List<String>> RESULT_NAMES =
            Map.of(
                    "stablehlo.rng_bit_generator", List.of("output_state", "output"),
                    "stablehlo.batch_norm_training", List.of("output", "batch_mean", "batch_var"),
                    "stablehlo.batch_norm_grad",
                            List.of("grad_operand", "grad_scale", "grad_offset"),
                    "chlo.top_k", List.of("values", "indices"));

    private final boolean interfaces;
    private final Map<Value, String> valueNames = new IdentityHashMap<>();
    private final Map<Operation, String> resultLists = new IdentityHashMap<>();

    /** Names introduced in a region and in the regions enclosing it. */
    private static final class Scope {
        final Scope parent;
        final Set<String> names = new HashSet<>();

        Scope(Scope parent) {
            this.parent = parent;
        }

        boolean contains(String name) {
            for (Scope s = this; s != null; s = s.parent) {
                if (s.names.contains(name)) {
                    return true;
                }
            }
            return false;
        }
    }

    private static final class Counters {
        int nextValue;
        int nextArgument;
        int nextConflict;

        Counters copy() {
            Counters c = new Counters();
            c.nextValue = nextValue;
            c.nextArgument = nextArgument;
            c.nextConflict = nextConflict;
            return c;
        }
    }

    /** Per value: its name, or null for a numbered value (its id is in {@code ids}). */
    private final Map<Value, String> named = new IdentityHashMap<>();

    private final Map<Value, Integer> ids = new IdentityHashMap<>();
    private final Map<Operation, List<Integer>> groups = new IdentityHashMap<>();

    private SsaNames(boolean interfaces) {
        this.interfaces = interfaces;
    }

    /**
     * Names the values of {@code top} and of every operation nested in it, as MLIR prints {@code
     * top}; {@code pretty} selects the pretty form's interface names.
     */
    static SsaNames of(Operation top, boolean pretty) {
        SsaNames n = new SsaNames(pretty);
        Counters counters = new Counters();
        Scope scope = new Scope(null);
        // MLIR saves the counters for the top-level regions before numbering the op itself
        Counters forRegions = counters.copy();
        n.numberOp(top, counters, scope);
        for (Region r : top.regions()) {
            n.numberRegion(r, top, forRegions.copy(), scope);
        }
        n.finish(top);
        return n;
    }

    /** How a use of {@code v} is printed: {@code %cst}, {@code %3}, {@code %2#1}. */
    String name(Value v) {
        return valueNames.get(v);
    }

    /** The results as printed before {@code =}: {@code %0}, {@code %2:2}, {@code %a, %b}. */
    String resultList(Operation op) {
        return resultLists.get(op);
    }

    private void numberRegion(Region region, Operation parent, Counters c, Scope parentScope) {
        Scope scope = new Scope(parentScope);
        if (interfaces) {
            blockArgumentNames(region, parent, c, scope);
        }
        for (Block block : region.blocks()) {
            boolean entry = block == region.blocks().get(0);
            for (Value arg : block.arguments()) {
                if (named.containsKey(arg) || ids.containsKey(arg)) {
                    continue;
                }
                if (entry) {
                    setName(arg, "arg" + c.nextArgument++, c, scope);
                } else {
                    ids.put(arg, c.nextValue++);
                }
            }
            for (Operation op : block.operations()) {
                numberOp(op, c, scope);
            }
        }
        for (Block block : region.blocks()) {
            for (Operation op : block.operations()) {
                for (Region nested : op.regions()) {
                    numberRegion(nested, op, c.copy(), scope);
                }
            }
        }
    }

    /** {@code getAsmBlockArgumentNames} of the ops that define it. */
    private void blockArgumentNames(Region region, Operation parent, Counters c, Scope scope) {
        if (region.blocks().isEmpty()) {
            return;
        }
        List<Value> args = region.blocks().get(0).arguments();
        switch (parent.name()) {
            case "stablehlo.while" -> {
                for (Value a : args) {
                    setName(a, "iterArg", c, scope);
                }
            }
            case "chlo.scan" -> {
                int inputs = scanInputs(parent);
                for (int i = 0; i < args.size(); i++) {
                    setName(args.get(i), i < inputs ? "input" : "carry", c, scope);
                }
            }
            default -> {}
        }
    }

    private static int scanInputs(Operation scan) {
        Attribute sizes = scan.attribute("operandSegmentSizes").orElse(null);
        if (sizes instanceof DenseArrayAttr d && d.values().length > 0) {
            return Math.toIntExact(d.values()[0]);
        }
        return scan.numOperands() / 2;
    }

    private void numberOp(Operation op, Counters c, Scope scope) {
        List<Integer> resultGroups = new ArrayList<>(List.of(0));
        if (interfaces && op.numResults() > 0) {
            List<String> resultNames = RESULT_NAMES.get(op.name());
            if (op.name().equals("stablehlo.constant") && op.numResults() == 1) {
                boolean integer =
                        op.result(0).type() instanceof TensorType t
                                && t.elementType() instanceof IntegerType;
                setName(op.result(0), integer ? "c" : "cst", c, scope);
            } else if (resultNames != null && resultNames.size() == op.numResults()) {
                for (int i = 0; i < resultNames.size(); i++) {
                    setName(op.result(i), resultNames.get(i), c, scope);
                    if (i > 0) {
                        resultGroups.add(i);
                    }
                }
            }
        }
        if (op.numResults() == 0) {
            return;
        }
        Value first = op.result(0);
        if (!named.containsKey(first) && !ids.containsKey(first)) {
            ids.put(first, c.nextValue++);
        }
        if (resultGroups.size() != 1) {
            groups.put(op, resultGroups);
        }
    }

    private void setName(Value v, String name, Counters c, Scope scope) {
        String unique = name;
        if (scope.contains(name)) {
            do {
                unique = name + "_" + c.nextConflict++;
            } while (scope.contains(unique));
        }
        scope.names.add(unique);
        named.put(v, unique);
    }

    /** The printed spelling of the first value of a result group. */
    private String base(Value v) {
        String n = named.get(v);
        return n != null ? "%" + n : "%" + ids.get(v);
    }

    /** Computes the printed spellings of every value numbered above. */
    private void finish(Operation top) {
        spell(top);
    }

    private void spell(Operation op) {
        int n = op.numResults();
        if (n > 0) {
            List<Integer> starts = groups.getOrDefault(op, List.of(0));
            StringJoiner list = new StringJoiner(", ");
            for (int g = 0; g < starts.size(); g++) {
                int start = starts.get(g);
                int size = (g + 1 < starts.size() ? starts.get(g + 1) : n) - start;
                String base = base(op.result(start));
                list.add(size > 1 ? base + ":" + size : base);
                for (int i = 0; i < size; i++) {
                    valueNames.put(
                            op.result(start + i), n == 1 || size == 1 ? base : base + "#" + i);
                }
            }
            resultLists.put(op, list.toString());
        }
        for (Region r : op.regions()) {
            for (Block b : r.blocks()) {
                for (Value a : b.arguments()) {
                    valueNames.put(a, base(a));
                }
                for (Operation inner : b.operations()) {
                    spell(inner);
                }
            }
        }
    }
}
