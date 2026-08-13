package com.qxotic.jinfer.x.models.nemotronh;

import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.Tool;
import com.qxotic.jinfer.x.chat.ToolCallSyntax;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Byte-exact port of the Nemotron template's tool rendering: the XML declarations block the system
 * turn embeds under {@code # Tools}, and the {@code <function=...>} call body. Pure text - the only
 * special tokens in the tool flow ({@code <tool_call>}, {@code </tool_call>}, {@code
 * <tool_response>}, {@code </tool_response>}) are spliced by the template class; everything here is
 * plain-encoded, so request-authored names/descriptions can never mint control tokens.
 *
 * <p>The template's {@code render_extra_keys} macro is load-bearing: schema fields beyond the
 * handled sets (array {@code items}, nested-object {@code properties}, {@code default}, ...) render
 * as {@code \n<key>value</key>} with {@code tojson} for maps/lists and the Jinja string form
 * otherwise (booleans {@code True}/{@code False}).
 */
final class NemotronToolSyntax {

    private NemotronToolSyntax() {}

    /** {@code # Tools\n\n...<tools>...</tools>} - the declarations block, instructions excluded. */
    static String declarations(List<Tool> tools) {
        StringBuilder sb = new StringBuilder();
        sb.append("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (Tool tool : tools) {
            Map<String, Object> fn = ToolCallSyntax.functionObject(tool);
            sb.append("\n<function>\n<name>").append(str(fn.get("name"))).append("</name>");
            if (fn.containsKey("description")) {
                sb.append("\n<description>")
                        .append(str(fn.get("description")).strip())
                        .append("</description>");
            }
            sb.append("\n<parameters>");
            Map<String, Object> parameters = map(fn.get("parameters"));
            Map<String, Object> properties =
                    parameters == null ? null : map(parameters.get("properties"));
            if (properties != null) {
                for (Map.Entry<String, Object> e : properties.entrySet()) {
                    Map<String, Object> field = map(e.getValue());
                    sb.append("\n<parameter>");
                    sb.append("\n<name>").append(e.getKey()).append("</name>");
                    if (field.containsKey("type")) {
                        sb.append("\n<type>").append(str(field.get("type"))).append("</type>");
                    }
                    if (field.containsKey("description")) {
                        sb.append("\n<description>")
                                .append(str(field.get("description")).strip())
                                .append("</description>");
                    }
                    if (field.containsKey("enum")) {
                        sb.append("\n<enum>")
                                .append(ToolCallSyntax.jinjaJson(field.get("enum")))
                                .append("</enum>");
                    }
                    extraKeys(field, Set.of("name", "type", "description", "enum"), sb);
                    sb.append("\n</parameter>");
                }
            }
            extraKeys(parameters, Set.of("type", "properties", "required"), sb);
            if (parameters != null && parameters.containsKey("required")) {
                sb.append("\n<required>")
                        .append(ToolCallSyntax.jinjaJson(parameters.get("required")))
                        .append("</required>");
            }
            sb.append("\n</parameters>");
            extraKeys(fn, Set.of("type", "name", "description", "parameters"), sb);
            sb.append("\n</function>");
        }
        sb.append("\n</tools>");
        return sb.toString();
    }

    /**
     * The call body between the trusted {@code <tool_call>} / {@code </tool_call>} ids: {@code
     * \n<function=NAME>\n<parameter=K>\nV\n</parameter>...\n</function>\n}.
     */
    static String call(Content.ToolCall call) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n<function=").append(call.name()).append(">\n");
        for (Map.Entry<String, Object> arg : call.arguments().entrySet()) {
            sb.append("<parameter=").append(arg.getKey()).append(">\n");
            sb.append(ToolCallSyntax.jinjaValue(arg.getValue()));
            sb.append("\n</parameter>\n");
        }
        sb.append("</function>\n");
        return sb.toString();
    }

    /** {@code render_extra_keys}: every unhandled schema key as {@code \n<key>value</key>}. */
    private static void extraKeys(Map<String, Object> dict, Set<String> handled, StringBuilder sb) {
        if (dict == null) return;
        for (Map.Entry<String, Object> e : dict.entrySet()) {
            if (handled.contains(e.getKey())) continue;
            sb.append('\n')
                    .append('<')
                    .append(e.getKey())
                    .append('>')
                    .append(ToolCallSyntax.jinjaValue(e.getValue()))
                    .append("</")
                    .append(e.getKey())
                    .append('>');
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object v) {
        return v instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
    }

    /** Jinja's {@code |string}: Python spellings for booleans and none. */
    private static String str(Object v) {
        if (v instanceof Boolean b) return b ? "True" : "False";
        if (v == null) return "None";
        return String.valueOf(v);
    }
}
