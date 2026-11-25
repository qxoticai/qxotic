package com.qxotic.jota.stablehlo.ir;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Compares two IR trees modulo value names: same operations, attributes (properties and discardable
 * attributes merged), types, operand wiring and regions.
 */
public final class StructuralEquality {

    private final Map<Value, Value> mapping = new IdentityHashMap<>();

    private StructuralEquality() {}

    /** Empty when equal, otherwise a description of the first difference. */
    public static Optional<String> diff(Program a, Program b) {
        return diff(a.operation(), b.operation());
    }

    public static Optional<String> diff(Operation a, Operation b) {
        return Optional.ofNullable(new StructuralEquality().compare(a, b, ""));
    }

    private String compare(Operation a, Operation b, String path) {
        String here = path + "/" + a.name();
        if (!a.name().equals(b.name())) {
            return here + ": name " + a.name() + " vs " + b.name();
        }
        if (a.numOperands() != b.numOperands()) {
            return here + ": operand count " + a.numOperands() + " vs " + b.numOperands();
        }
        for (int i = 0; i < a.numOperands(); i++) {
            if (mapping.get(a.operand(i)) != b.operand(i)) {
                return here + ": operand #" + i + " is wired differently";
            }
        }
        Map<String, Attribute> attrsA = merged(a.allAttributes());
        Map<String, Attribute> attrsB = merged(b.allAttributes());
        if (!attrsA.equals(attrsB)) {
            return here + ": attributes " + attrsA + " vs " + attrsB;
        }
        if (a.numResults() != b.numResults()) {
            return here + ": result count " + a.numResults() + " vs " + b.numResults();
        }
        for (int i = 0; i < a.numResults(); i++) {
            Value resIa = a.result(i);
            Value resIb = b.result(i);
            if (!resIa.type().equals(resIb.type())) {
                return here
                        + ": result #"
                        + i
                        + " type "
                        + resIa.type().mlir()
                        + " vs "
                        + resIb.type().mlir();
            }
            mapping.put(resIa, resIb);
        }
        if (a.regions().size() != b.regions().size()) {
            return here + ": region count";
        }
        for (int r = 0; r < a.regions().size(); r++) {
            List<Block> ba = a.region(r).blocks();
            List<Block> bb = b.region(r).blocks();
            if (ba.size() != bb.size()) {
                return here + ": region #" + r + " block count " + ba.size() + " vs " + bb.size();
            }
            for (int k = 0; k < ba.size(); k++) {
                String d = compare(ba.get(k), bb.get(k), here + "[" + r + "." + k + "]");
                if (d != null) {
                    return d;
                }
            }
        }
        return null;
    }

    private String compare(Block a, Block b, String path) {
        if (a.arguments().size() != b.arguments().size()) {
            return path + ": block argument count";
        }
        for (int i = 0; i < a.arguments().size(); i++) {
            if (!a.argument(i).type().equals(b.argument(i).type())) {
                return path + ": block argument #" + i + " type";
            }
            mapping.put(a.argument(i), b.argument(i));
        }
        if (a.operations().size() != b.operations().size()) {
            return path
                    + ": operation count "
                    + a.operations().size()
                    + " vs "
                    + b.operations().size();
        }
        for (int i = 0; i < a.operations().size(); i++) {
            String d = compare(a.operations().get(i), b.operations().get(i), path);
            if (d != null) {
                return d;
            }
        }
        return null;
    }

    private static Map<String, Attribute> merged(List<NamedAttribute> attrs) {
        Map<String, Attribute> m = new LinkedHashMap<>();
        for (NamedAttribute a : attrs) {
            m.put(a.name(), a.value());
        }
        return m;
    }
}
