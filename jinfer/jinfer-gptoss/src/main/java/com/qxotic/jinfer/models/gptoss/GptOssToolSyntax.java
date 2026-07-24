package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import java.util.List;
import java.util.Map;

/**
 * Byte-exact port of the gpt-oss template's tool-declaration macros ({@code render_tool_namespace}
 * / {@code render_typescript_type}): the TypeScript-flavored namespace block the developer message
 * embeds under {@code # Tools}. Pure text - Harmony declarations contain no special tokens, so this
 * renders plain strings and the template encodes them in one contiguous run.
 *
 * <p>Faithful to the template's quirks on purpose (they are what the model was trained on): every
 * parameter line ends {@code ,\n} (last included), enum defaults concatenate raw while plain
 * defaults go through {@code tojson}, and the nested-object / oneOf branches keep the template's
 * literal indentation runs (untrimmed {@code {{ ... }}} tags). One deviation: a missing tool
 * description renders as an empty {@code //} comment where Jinja would error on the concat.
 */
final class GptOssToolSyntax {

    private GptOssToolSyntax() {}

    /** {@code render_tool_namespace("functions", tools)}: the whole declaration block. */
    static String namespace(List<Tool> tools) {
        StringBuilder sb = new StringBuilder();
        sb.append("## functions\n\n");
        sb.append("namespace functions {\n\n");
        for (Tool tool : tools) {
            Map<String, Object> fn = function(tool);
            sb.append("// ").append(str(fn.get("description"))).append('\n');
            sb.append("type ").append(str(fn.get("name"))).append(" = ");
            Map<String, Object> parameters = map(fn.get("parameters"));
            Map<String, Object> properties =
                    parameters == null ? null : map(parameters.get("properties"));
            if (properties != null && !properties.isEmpty()) {
                sb.append("(_: {\n");
                List<?> required = list(parameters.get("required"));
                for (Map.Entry<String, Object> e : properties.entrySet()) {
                    Map<String, Object> spec = map(e.getValue());
                    if (truthy(spec.get("description"))) {
                        sb.append("// ").append(spec.get("description")).append('\n');
                    }
                    sb.append(e.getKey());
                    if (!required.contains(e.getKey())) sb.append('?');
                    sb.append(": ").append(tsType(spec));
                    if (spec.containsKey("default")) {
                        Object d = spec.get("default");
                        if (truthy(spec.get("enum"))) sb.append(", // default: ").append(d);
                        else if (truthy(spec.get("oneOf"))) sb.append("// default: ").append(d);
                        else sb.append(", // default: ").append(ToolCallSyntax.jinjaJson(d));
                    }
                    sb.append(",\n"); // the template emits ",\n" on the last parameter too
                }
                sb.append("}) => any;\n\n");
            } else {
                sb.append("() => any;\n\n");
            }
        }
        sb.append("} // namespace functions");
        return sb.toString();
    }

    /** {@code render_typescript_type(param_spec, ...)} - the schema-to-TS-type lowering. */
    static String tsType(Map<String, Object> spec) {
        Object type = spec.get("type");
        if ("array".equals(type)) {
            String nullable = truthy(spec.get("nullable")) ? " | null" : "";
            Map<String, Object> items = map(spec.get("items"));
            if (items != null && !items.isEmpty()) {
                Object itemType = items.get("type");
                if ("string".equals(itemType)) return "string[]" + nullable;
                if ("number".equals(itemType) || "integer".equals(itemType))
                    return "number[]" + nullable;
                if ("boolean".equals(itemType)) return "boolean[]" + nullable;
                String inner = tsType(items);
                if (inner.equals("object | object") || inner.length() > 50)
                    return "any[]" + nullable;
                return inner + "[]" + nullable;
            }
            return "any[]" + nullable;
        }
        if (type instanceof List<?> union && !union.isEmpty()) {
            // Union[dict, list] shapes arrive as a type array: joined, or the single entry
            if (union.size() == 1) return String.valueOf(union.get(0));
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < union.size(); i++) {
                if (i > 0) sb.append(" | ");
                sb.append(union.get(i));
            }
            return sb.toString();
        }
        List<?> oneOf = list(spec.get("oneOf"));
        if (!oneOf.isEmpty()) {
            int objects = 0;
            for (Object v : oneOf) {
                Map<String, Object> variant = map(v);
                if (variant != null && "object".equals(variant.get("type"))) objects++;
            }
            if (objects > 0 && oneOf.size() > 1) return "any";
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < oneOf.size(); i++) {
                Map<String, Object> variant = map(oneOf.get(i));
                sb.append(tsType(variant));
                if (truthy(variant.get("description")))
                    sb.append("// ").append(variant.get("description"));
                if (variant.containsKey("default")) {
                    // the template's untrimmed "{{ ... }}" tag: literal newline + indent emitted
                    sb.append("\n                    // default: ")
                            .append(ToolCallSyntax.jinjaJson(variant.get("default")));
                }
                if (i < oneOf.size() - 1) sb.append(" | \n                ");
            }
            return sb.toString();
        }
        if ("string".equals(type)) {
            List<?> enums = list(spec.get("enum"));
            if (!enums.isEmpty()) {
                StringBuilder sb = new StringBuilder("\"");
                for (int i = 0; i < enums.size(); i++) {
                    if (i > 0) sb.append("\" | \"");
                    sb.append(enums.get(i));
                }
                return sb.append('"').toString();
            }
            return truthy(spec.get("nullable")) ? "string | null" : "string";
        }
        if ("number".equals(type) || "integer".equals(type)) return "number";
        if ("boolean".equals(type)) return "boolean";
        if ("object".equals(type)) {
            Map<String, Object> properties = map(spec.get("properties"));
            if (properties != null && !properties.isEmpty()) {
                List<?> required = list(spec.get("required"));
                StringBuilder sb = new StringBuilder("{\n");
                int i = 0;
                for (Map.Entry<String, Object> e : properties.entrySet()) {
                    sb.append(e.getKey());
                    if (!required.contains(e.getKey())) sb.append('?');
                    // the template's untrimmed "{{ render... }}" tag: newline + indent before it
                    sb.append(": \n                ").append(tsType(map(e.getValue())));
                    if (++i < properties.size()) sb.append(", ");
                }
                return sb.append('}').toString();
            }
            return "object";
        }
        return "any";
    }

    /** The tool's inner {@code function} object; lenient when rawJson IS already that object. */
    private static Map<String, Object> function(Tool tool) {
        Map<String, Object> raw = map(JsonCodec.parse(tool.rawJson()));
        Map<String, Object> fn = map(raw.get("function"));
        return fn != null ? fn : raw;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object v) {
        return v instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
    }

    private static List<?> list(Object v) {
        return v instanceof List<?> l ? l : List.of();
    }

    private static String str(Object v) {
        return v == null ? "" : String.valueOf(v);
    }

    /** Jinja truthiness for the template's bare {@code if x} checks. */
    private static boolean truthy(Object v) {
        return switch (v) {
            case null -> false;
            case Boolean b -> b;
            case String s -> !s.isEmpty();
            case Map<?, ?> m -> !m.isEmpty();
            case List<?> l -> !l.isEmpty();
            case Number n -> n.doubleValue() != 0;
            default -> true;
        };
    }
}
