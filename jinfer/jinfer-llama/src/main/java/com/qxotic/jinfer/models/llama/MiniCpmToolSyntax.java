package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * MiniCPM5's XML function-call payload: the span between the trusted {@code <function} and {@code
 * </function>} ids carries {@code name="NAME"><param name="K">V</param>...}, values optionally
 * CDATA-wrapped ({@code <![CDATA[...]]>}) when they contain {@code <}, {@code &} or newlines.
 * Argument values stay STRINGS (the wire is untyped; consumers coerce against the schema), so an
 * echo re-renders to the exact bytes the model emitted.
 */
final class MiniCpmToolSyntax {

    private MiniCpmToolSyntax() {}

    /** Parse one function span's payload (everything between the two trusted ids). */
    static List<Part.ToolCall> parsePayload(String block) {
        int nameAt = block.indexOf("name=\"");
        if (nameAt < 0) return List.of();
        int nameEnd = block.indexOf('"', nameAt + 6);
        if (nameEnd < 0) return List.of();
        String name = block.substring(nameAt + 6, nameEnd);
        if (name.isEmpty()) return List.of();
        Map<String, Object> args = new LinkedHashMap<>();
        int p = block.indexOf("<param", nameEnd);
        while (p >= 0) {
            int kAt = block.indexOf("name=\"", p);
            if (kAt < 0) break;
            int kEnd = block.indexOf('"', kAt + 6);
            if (kEnd < 0) break;
            String key = block.substring(kAt + 6, kEnd);
            int vAt = block.indexOf('>', kEnd);
            if (vAt < 0) break;
            int close = block.indexOf("</param>", vAt);
            if (close < 0) break;
            String value = block.substring(vAt + 1, close);
            if (value.startsWith("<![CDATA[") && value.endsWith("]]>")) {
                value = value.substring("<![CDATA[".length(), value.length() - "]]>".length());
            }
            if (!key.isEmpty()) args.put(key, value);
            p = block.indexOf("<param", close);
        }
        return List.of(new Part.ToolCall("", name, args));
    }

    /**
     * One parameter's rendered value, the template's rule: strings containing {@code <}, {@code &}
     * or newlines wrap in CDATA; other values render as Jinja prints them (numbers plain, {@code
     * True}/{@code False}/{@code None}, python-repr for structures).
     */
    static String paramValue(Object v) {
        if (v instanceof String s) {
            return s.indexOf('<') >= 0 || s.indexOf('&') >= 0 || s.indexOf('\n') >= 0
                    ? "<![CDATA[" + s + "]]>"
                    : s;
        }
        if (v instanceof Map || v instanceof List) return ToolCallSyntax.pythonRepr(v);
        if (v instanceof Boolean b) return b ? "True" : "False";
        if (v == null) return "None";
        return String.valueOf(v);
    }
}
