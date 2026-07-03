// The TurnTemplate-vs-Jinja oracle: a hand-written TurnTemplate must be token-exact with the
// GGUF's own chat_template rendered by jinfer-jinja and re-scanned with encodeWithSpecialTokens.
// Model-agnostic core: per-model mains supply the battery (conversation shapes + render vars) and
// their injection-inertness check via the id/decode helpers here. Loads only the tokenizer, never
// the weights.
package com.qxotic.jinfer.testkit;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CompiledTemplate;
import com.qxotic.jinfer.GgufTokenizer;
import com.qxotic.jinfer.ModelLoader;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.jinja.JinjaRenderer;

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

    /** Loads the GGUF's tokenizer + compiled chat template. {@code renderVars} are the extra
     *  template variables (bos/eos/date) that pin the render deterministic. */
    public OracleScenario(Path gguf, java.util.function.Function<GgufTokenizer, TurnTemplate> template,
                          Map<String, Object> renderVars) throws Exception {
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

    /** Comparison with per-case overrides: {@code thinking} selects the hand-written
     *  generation-prompt scaffold, {@code extraVars} merge over the instance render vars (e.g.
     *  {@code enable_thinking=false} to pin a template's non-thinking branch). */
    public void compare(String name, boolean generationPrompt, boolean thinking,
                        Map<String, Object> extraVars, List<Message> conversation) {
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
        while (at < Math.min(oracle.size(), ours.size()) && oracle.get(at).equals(ours.get(at))) at++;
        System.out.println("  diverge at " + at + "/" + oracle.size() + " (ours " + ours.size() + ")");
        System.out.println("  oracle: " + window(oracle, at) + "  |" + decode(window(oracle, at)) + "|");
        System.out.println("  ours:   " + window(ours, at) + "  |" + decode(window(ours, at)) + "|");
        System.out.println("  rendered: " + rendered.replace("\n", "\\n"));
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
