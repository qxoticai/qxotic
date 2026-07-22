// The plumbing shared by OracleScenario and CodecOracleScenario: the GGUF tokenizer +
// compiled-chat-template load, the Message-to-OpenAI-map lowering the Jinja side reads, and the
// token-diff failure printer. The scenarios keep their distinct drive logic (per-turn vs codec).
package com.qxotic.jinfer.testkit;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.jinja.CompiledTemplate;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.toknroll.Tokenizer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class OracleSupport {

    final Tokenizer tokenizer;
    final CompiledTemplate jinja;

    /** Loads the GGUF's tokenizer and compiles its own chat template (never the weights). */
    OracleSupport(Path gguf) throws Exception {
        GGUF g;
        try (FileChannel channel = FileChannel.open(gguf, StandardOpenOption.READ)) {
            g = ModelLoader.readGguf(channel, gguf.toString());
        }
        this.tokenizer = Tokenizers.fromGGUF(g);
        String source = Tokenizers.chatTemplateSource(g);
        this.jinja = source.isEmpty() ? null : JinjaRenderer.template(source);
        if (jinja == null) throw new IllegalStateException("GGUF chat_template failed to compile");
    }

    /** A Message to the OpenAI-shaped map the Jinja template reads (text + tool_calls). */
    static Map<String, Object> oracleMessage(Message m) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("role", m.role().name());
        StringBuilder text = new StringBuilder();
        List<Object> toolCalls = new ArrayList<>();
        for (Part p : m.content()) {
            if (p instanceof Part.Text t) text.append(t.text());
            else if (p instanceof Part.ToolResult r) text.append(r.text());
            else if (p instanceof Part.ToolCall c) {
                Map<String, Object> fn = new LinkedHashMap<>();
                fn.put("name", c.name());
                fn.put("arguments", c.arguments());
                Map<String, Object> call = new LinkedHashMap<>();
                call.put("type", "function");
                call.put("function", fn);
                toolCalls.add(call);
            }
        }
        map.put("content", text.toString());
        if (!toolCalls.isEmpty()) map.put("tool_calls", toolCalls);
        return map;
    }

    /**
     * The token-exact check plus, on failure, the divergence window (oracle vs ours, ids and
     * decoded text) and the rendered reference when the caller has one ({@code rendered} nullable).
     */
    void diff(
            Checks checks, String name, List<Integer> oracle, List<Integer> ours, String rendered) {
        boolean equal = oracle.equals(ours);
        checks.check(equal, name + " (" + ours.size() + " tokens)");
        if (equal) return;
        int at = 0;
        while (at < Math.min(oracle.size(), ours.size()) && oracle.get(at).equals(ours.get(at)))
            at++;
        System.out.println(
                "  diverge at " + at + "/" + oracle.size() + " (ours " + ours.size() + ")");
        System.out.println(
                "  oracle: " + window(oracle, at) + "  |" + decode(window(oracle, at)) + "|");
        System.out.println(
                "  ours:   " + window(ours, at) + "  |" + decode(window(ours, at)) + "|");
        if (rendered != null) System.out.println("  rendered: " + rendered.replace("\n", "\\n"));
    }

    int special(String name) {
        return SpecialTokens.find(tokenizer, name).getAsInt();
    }

    long count(List<Integer> ids, int id) {
        return java.util.Collections.frequency(ids, id);
    }

    String decode(List<Integer> ids) {
        return tokenizer.decode(com.qxotic.toknroll.IntSequence.wrap(ids)).replace("\n", "\\n");
    }

    private static List<Integer> window(List<Integer> ids, int at) {
        return ids.subList(Math.max(0, at - 2), Math.min(ids.size(), at + 6));
    }
}
