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
    private int failures;

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
        List<Object> maps = new ArrayList<>();
        for (Message m : conversation) {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("role", m.role().name());
            map.put("content", m.text());
            maps.add(map);
        }
        Map<String, Object> vars = new HashMap<>(renderVars);
        vars.put("messages", maps);
        vars.put("add_generation_prompt", generationPrompt);
        String rendered = jinja.render(vars);
        List<Integer> oracle = tokenizer.encodeWithSpecialTokens(rendered);

        List<Batch> batches = new ArrayList<>(mine.encode(conversation));
        if (generationPrompt) batches.addAll(mine.generationPrompt(true));
        List<Integer> ours = new ArrayList<>();
        for (Batch b : batches) {
            for (int id : ((Batch.Input.Tokens) b.input()).ids()) ours.add(id);
        }

        if (oracle.equals(ours)) {
            System.out.println("ok:   " + name + " (" + ours.size() + " tokens)");
            return;
        }
        failures++;
        System.out.println("FAIL: " + name);
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
        for (Batch b : mine.encodeTurn(m)) {
            for (int id : ((Batch.Input.Tokens) b.input()).ids()) ids.add(id);
        }
        return ids;
    }

    public int special(String name) {
        return tokenizer.getSpecialTokens().get(name);
    }

    public long count(List<Integer> ids, int id) {
        return ids.stream().filter(x -> x == id).count();
    }

    public String decode(List<Integer> ids) {
        return tokenizer.decode(ids).replace("\n", "\\n");
    }

    public void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }

    public void finish(String name) {
        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println(name + ": all cases token-exact");
    }

    private static List<Integer> window(List<Integer> ids, int at) {
        return ids.subList(Math.max(0, at - 2), Math.min(ids.size(), at + 6));
    }
}
