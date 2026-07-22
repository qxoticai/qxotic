// The TurnTemplate-vs-Jinja oracle: a hand-written TurnTemplate must be token-exact with the
// GGUF's own chat_template rendered by jinfer-jinja and re-scanned with encodeWithSpecialTokens.
// Model-agnostic core: per-model mains supply the battery (conversation shapes + render vars) and
// their injection-inertness check via the id/decode helpers here. Loads only the tokenizer, never
// the weights.
package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class OracleScenario {

    public final Tokenizer tokenizer;
    private final com.qxotic.toknroll.Specials specials;
    private final OracleSupport support;
    private final TurnTemplate mine;
    private final Map<String, Object> renderVars;
    private final Checks checks = new Checks();

    /**
     * Loads the GGUF's tokenizer + compiled chat template. {@code renderVars} are the extra
     * template variables (bos/eos/date) that pin the render deterministic.
     */
    public OracleScenario(
            Path gguf,
            java.util.function.Function<Tokenizer, TurnTemplate> template,
            Map<String, Object> renderVars)
            throws Exception {
        this.support = new OracleSupport(gguf);
        this.tokenizer = support.tokenizer;
        this.specials = SpecialTokens.encoder(tokenizer);
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
        for (Message m : conversation) maps.add(OracleSupport.oracleMessage(m));
        Map<String, Object> vars = new HashMap<>(renderVars);
        vars.putAll(extraVars);
        vars.put("messages", maps);
        vars.put("add_generation_prompt", generationPrompt);
        String rendered = support.jinja.render(vars);
        List<Integer> oracle = specials.encode(tokenizer, rendered).toList();

        List<Batch> batches = new ArrayList<>(mine.conversationStart());
        for (Message m : mine.normalize(conversation)) batches.addAll(mine.encodeTurn(m));
        if (generationPrompt) batches.addAll(mine.generationPrompt(thinking));
        List<Integer> ours = new ArrayList<>();
        for (int id : Batch.tokenIds(batches)) ours.add(id);

        support.diff(checks, name, oracle, ours, rendered);
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
        return support.special(name);
    }

    public long count(List<Integer> ids, int id) {
        return support.count(ids, id);
    }

    public String decode(List<Integer> ids) {
        return support.decode(ids);
    }
}
