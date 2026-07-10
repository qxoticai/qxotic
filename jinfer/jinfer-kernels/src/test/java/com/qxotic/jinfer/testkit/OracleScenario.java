// The TurnTemplate-vs-Jinja oracle: a hand-written TurnTemplate must be token-exact with the
// GGUF's own chat_template rendered by jinfer-jinja and re-scanned with encodeWithSpecialTokens.
// Model-agnostic core: per-model mains supply the battery (conversation shapes + render vars) and
// their injection-inertness check via the id/decode helpers here. Loads only the tokenizer, never
// the weights.
package com.qxotic.jinfer.testkit;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.CompiledTemplate;
import com.qxotic.jinfer.llm.GgufTokenizer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class OracleScenario {

    public final GgufTokenizer tokenizer;
    private final CompiledTemplate jinja;
    private final TurnTemplate mine;
    private final Map<String, Object> renderVars;
    private final Checks checks = new Checks();

    /**
     * Loads the GGUF's tokenizer + compiled chat template. {@code renderVars} are the extra
     * template variables (bos/eos/date) that pin the render deterministic.
     */
    public OracleScenario(
            Path gguf,
            java.util.function.Function<GgufTokenizer, TurnTemplate> template,
            Map<String, Object> renderVars)
            throws Exception {
        GGUF g;
        try (FileChannel channel = FileChannel.open(gguf, StandardOpenOption.READ)) {
            g = ModelLoader.readGguf(channel, gguf.toString());
        }
        this.tokenizer = new GgufTokenizer(g, JinjaRenderer::template);
        this.jinja = tokenizer.chatTemplate();
        if (jinja == null) throw new IllegalStateException("GGUF chat_template failed to compile");
        this.mine = template.apply(tokenizer);
        this.renderVars = renderVars;
    }

    /** Token-exact comparison of the hand-written encode vs the rendered+rescanned template. */
    public void compare(String name, boolean generationPrompt, List<Message> conversation) {
        compare(name, generationPrompt, true, Map.of(), conversation);
    }

    /**
     * Comparison with per-case overrides: {@code thinking} selects the hand-written
     * generation-prompt scaffold, {@code extraVars} merge over the instance render vars (e.g.
     * {@code enable_thinking=false} to pin a template's non-thinking branch).
     */
    public void compare(
            String name,
            boolean generationPrompt,
            boolean thinking,
            Map<String, Object> extraVars,
            List<Message> conversation) {
        List<Object> maps = new ArrayList<>();
        for (Message m : conversation) {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("role", m.role().name());
            map.put("content", m.text());
            maps.add(map);
        }
        Map<String, Object> vars = new HashMap<>(renderVars);
        vars.putAll(extraVars);
        vars.put("messages", maps);
        vars.put("add_generation_prompt", generationPrompt);
        String rendered = jinja.render(vars);
        List<Integer> oracle = tokenizer.encodeWithSpecialTokens(rendered);

        List<Batch> batches = new ArrayList<>(mine.encode(conversation));
        if (generationPrompt) batches.addAll(mine.generationPrompt(thinking));
        List<Integer> ours = new ArrayList<>();
        for (int id : Batch.tokenIds(batches)) ours.add(id);

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
        System.out.println("  rendered: " + rendered.replace("\n", "\\n"));
    }

    /**
     * Tool-aware comparison: the conversation may carry {@link
     * com.qxotic.jinfer.chat.Part.ToolCall} parts (assistant turns) and {@link
     * com.qxotic.jinfer.chat.Role#TOOL} turns, and {@code tools} is the offered tool list. The
     * Jinja side gets {@code tools} (parsed from each Tool's rawJson, so its own {@code tojson}
     * renders them) plus messages carrying {@code tool_calls}; our side runs the preamble path:
     * {@code conversationStart(Preamble)} folds the leading system message and tools, then the
     * remaining turns, then the generation prompt.
     */
    public void compareTools(
            String name,
            boolean generationPrompt,
            List<com.qxotic.jinfer.chat.Tool> tools,
            List<Message> conversation) {
        List<Object> maps = new ArrayList<>();
        for (Message m : conversation) maps.add(oracleMessage(m));
        List<Object> toolVars = new ArrayList<>();
        for (com.qxotic.jinfer.chat.Tool t : tools)
            toolVars.add(com.qxotic.format.json.Json.parse(t.rawJson()));

        Map<String, Object> vars = new HashMap<>(renderVars);
        vars.put("messages", maps);
        vars.put("tools", toolVars);
        vars.put("add_generation_prompt", generationPrompt);
        String rendered = jinja.render(vars);
        List<Integer> oracle = tokenizer.encodeWithSpecialTokens(rendered);
        List<Integer> ours = encodeWithPreamble(tools, conversation, generationPrompt);

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
        System.out.println("  rendered: " + rendered.replace("\n", "\\n"));
    }

    /**
     * Like {@link #compareTools} but against an explicit expected rendered string instead of the
     * Jinja engine. Needed for the tool-CALL turns: jinfer-jinja cannot evaluate LFM2's {@code
     * render_tool_calls} macro (nested namespace mutation + a macro call inside the loop), so the
     * reference is the known-correct string the trained format produces, rescanned with
     * encodeWithSpecialTokens - the same rescan the Jinja path uses.
     */
    public void compareToolsExpected(
            String name,
            String expectedRendered,
            boolean generationPrompt,
            List<com.qxotic.jinfer.chat.Tool> tools,
            List<Message> conversation) {
        List<Integer> oracle = tokenizer.encodeWithSpecialTokens(expectedRendered);
        List<Integer> ours = encodeWithPreamble(tools, conversation, generationPrompt);
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
    }

    /** The preamble-path encoding: conversationStart(Preamble) + non-system turns + gen prompt. */
    private List<Integer> encodeWithPreamble(
            List<com.qxotic.jinfer.chat.Tool> tools,
            List<Message> conversation,
            boolean generationPrompt) {
        java.util.Optional<Message> system =
                !conversation.isEmpty() && conversation.get(0).role().equals(Role.SYSTEM)
                        ? java.util.Optional.of(conversation.get(0))
                        : java.util.Optional.empty();
        List<Message> turns =
                system.isPresent() ? conversation.subList(1, conversation.size()) : conversation;
        List<Batch> batches =
                new ArrayList<>(mine.conversationStart(new TurnTemplate.Preamble(system, tools)));
        for (Message m : turns) batches.addAll(mine.encodeTurn(m));
        if (generationPrompt) batches.addAll(mine.generationPrompt(true));
        List<Integer> ours = new ArrayList<>();
        for (int id : Batch.tokenIds(batches)) ours.add(id);
        return ours;
    }

    /** A Message to the OpenAI-shaped map the Jinja template reads (text + tool_calls). */
    private static Map<String, Object> oracleMessage(Message m) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("role", m.role().name());
        StringBuilder text = new StringBuilder();
        List<Object> toolCalls = new ArrayList<>();
        for (com.qxotic.jinfer.chat.Part p : m.content()) {
            if (p instanceof com.qxotic.jinfer.chat.Part.Text t) text.append(t.text());
            else if (p instanceof com.qxotic.jinfer.chat.Part.ToolCall c) {
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

    // ---- helpers for the per-model injection-inertness checks ----

    /** The hand-written encoding of one turn, flattened to ids. */
    public List<Integer> encodeTurnIds(Message m) {
        List<Integer> ids = new ArrayList<>();
        for (int id : Batch.tokenIds(mine.encodeTurn(m))) ids.add(id);
        return ids;
    }

    public void check(boolean ok, String what) {
        checks.check(ok, what);
    }

    /** Prints the verdict and exits non-zero on any failed case. */
    public void finish(String name) {
        checks.finish(name, "all cases token-exact");
    }

    public int special(String name) {
        return tokenizer.getSpecialTokens().get(name);
    }

    public long count(List<Integer> ids, int id) {
        return java.util.Collections.frequency(ids, id);
    }

    public String decode(List<Integer> ids) {
        return tokenizer.decode(ids).replace("\n", "\\n");
    }

    private static List<Integer> window(List<Integer> ids, int at) {
        return ids.subList(Math.max(0, at - 2), Math.min(ids.size(), at + 6));
    }
}
