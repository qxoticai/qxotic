package com.qxotic.jota.stablehlo.transform;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.ir.IrRewriter;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Drops StableHLO attributes whose value equals the op's default (e.g. {@code api_version = 1 :
 * i32} on {@code custom_call}, an all-{@code DEFAULT} {@code precision_config}, the default {@code
 * result_accuracy}), as MLIR's printer and the VHLO legalization do. Two programs that differ only
 * in spelled-out defaults are equal after this pass.
 */
public final class CanonicalizeDefaults {

    private static final Set<String> RESULT_ACCURACY_OPS =
            Set.of(
                    "cbrt",
                    "cosine",
                    "exponential",
                    "exponential_minus_one",
                    "exp2",
                    "log",
                    "log_plus_one",
                    "log2",
                    "logistic",
                    "rsqrt",
                    "sine",
                    "sqrt",
                    "tan",
                    "tanh");

    private static final Set<String> COLLECTIVES =
            Set.of(
                    "all_gather",
                    "all_reduce",
                    "all_to_all",
                    "collective_broadcast",
                    "collective_permute",
                    "collective_reduce",
                    "reduce_scatter");

    private CanonicalizeDefaults() {}

    public static Program apply(Program program) {
        IrRewriter.AttributeRewrite strip =
                (op, attrs) -> attrs.stream().filter(a -> !isDefault(op, a)).toList();
        return IrRewriter.rewriteAttributes(program, strip, strip);
    }

    /** True when {@code attribute} of {@code op} holds the op's default value. */
    public static boolean isDefault(Operation op, NamedAttribute attribute) {
        if (!op.dialect().equals("stablehlo")) {
            return false;
        }
        String o = op.mnemonic();
        String n = attribute.name();
        Attribute v = attribute.value();
        if (RESULT_ACCURACY_OPS.contains(o) && n.equals("result_accuracy")) {
            return isDefaultResultAccuracy(v);
        }
        if (COLLECTIVES.contains(o)) {
            if (n.equals("channel_handle")) {
                return isChannel(v, 0, 0);
            }
            if (n.equals("use_global_device_ids") || n.equals("has_dynamic_root")) {
                return isBool(v, false);
            }
        }
        if (Set.of("convolution", "dynamic_conv", "dot_general", "dot").contains(o)
                && n.equals("precision_config")) {
            return v instanceof ArrayAttr a
                    && a.elements().stream()
                            .allMatch(
                                    e -> e instanceof EnumAttr en && en.value().equals("DEFAULT"));
        }
        if (Set.of("gather", "dynamic_gather", "scatter").contains(o)
                && n.equals("indices_are_sorted")) {
            return isBool(v, false);
        }
        return switch (o + "." + n) {
            case "custom_call.api_version" -> isInteger(v, 1);
            case "custom_call.has_side_effect" -> isBool(v, false);
            case "custom_call.backend_config" ->
                    isString(v, "") || (v instanceof DictionaryAttr d && d.isEmpty());
            case "custom_call.called_computations",
                    "custom_call.output_operand_aliases",
                    "custom_call.result_tilings" ->
                    isEmptyArray(v);
            case "compare.compare_type" -> v instanceof EnumAttr e && e.value().equals("NOTYPE");
            case "cholesky.lower" -> isBool(v, false);
            case "composite.version" -> isInteger(v, 0);
            case "composite.composite_attributes" -> v instanceof DictionaryAttr d && d.isEmpty();
            case "convolution.window_strides",
                    "convolution.lhs_dilation",
                    "convolution.rhs_dilation",
                    "dynamic_conv.window_strides",
                    "dynamic_conv.lhs_dilation",
                    "dynamic_conv.rhs_dilation",
                    "reduce_window.window_strides",
                    "reduce_window.base_dilations",
                    "reduce_window.window_dilations",
                    "select_and_scatter.window_strides" ->
                    allEqual(v, 1);
            case "convolution.padding", "reduce_window.padding", "select_and_scatter.padding" ->
                    allEqual(v, 0);
            case "convolution.window_reversal", "dynamic_conv.window_reversal" -> allEqual(v, 0);
            case "scatter.unique_indices", "send.is_host_transfer", "recv.is_host_transfer" ->
                    isBool(v, false);
            case "sort.dimension" -> isInteger(v, -1);
            case "sort.is_stable" -> isBool(v, false);
            case "infeed.infeed_config", "outfeed.outfeed_config" -> isString(v, "");
            case "infeed.layout" -> isEmptyArray(v);
            case "send.source_target_pairs",
                    "recv.source_target_pairs",
                    "dynamic_broadcast_in_dim.known_expanding_dimensions",
                    "dynamic_broadcast_in_dim.known_nonexpanding_dimensions" ->
                    isEmpty(v);
            default -> false;
        };
    }

    private static boolean isBool(Attribute v, boolean value) {
        return v instanceof BoolAttr b && b.value() == value;
    }

    private static boolean isInteger(Attribute v, long value) {
        return v instanceof IntegerAttr i && i.value() == value;
    }

    private static boolean isString(Attribute v, String value) {
        return v instanceof StringAttr s && s.value().equals(value);
    }

    private static boolean isEmptyArray(Attribute v) {
        return v instanceof ArrayAttr a && a.elements().isEmpty();
    }

    private static boolean isEmpty(Attribute v) {
        return (v instanceof DenseElementsAttr d && d.numElements() == 0)
                || (v instanceof DenseArrayAttr a && a.size() == 0);
    }

    /** Every element equals {@code value} (and there is at least one element). */
    private static boolean allEqual(Attribute v, long value) {
        Predicate<Long> eq = x -> x == value;
        if (v instanceof DenseArrayAttr a) {
            return a.size() > 0 && java.util.Arrays.stream(a.values()).allMatch(x -> eq.test(x));
        }
        if (v instanceof DenseElementsAttr d) {
            if (d.numElements() == 0) {
                return false;
            }
            for (long i = 0; i < d.numElements(); i++) {
                if (d.getLong(i) != value) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    private static boolean isChannel(Attribute v, long handle, long type) {
        return v instanceof DialectAttr d
                && d.mnemonic().equals("channel_handle")
                && d.param("handle").map(h -> isInteger(h, handle)).orElse(false)
                && d.param("type").map(t -> isInteger(t, type)).orElse(false);
    }

    private static boolean isDefaultResultAccuracy(Attribute v) {
        if (!(v instanceof DialectAttr d) || !d.hasParams()) {
            return false;
        }
        boolean zero =
                d.param("atol").map(a -> a instanceof FloatAttr f && f.value() == 0.0).orElse(true)
                        && d.param("rtol")
                                .map(a -> a instanceof FloatAttr f && f.value() == 0.0)
                                .orElse(true)
                        && d.param("ulps").map(a -> isInteger(a, 0)).orElse(true);
        boolean defaultMode =
                d.param("mode")
                        .map(
                                m ->
                                        m instanceof DialectAttr dm
                                                && !dm.hasParams()
                                                && dm.rawBody().trim().equals("DEFAULT"))
                        .orElse(true);
        return zero && defaultMode;
    }
}
