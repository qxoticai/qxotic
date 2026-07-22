// The ChatTemplate-vs-Jinja oracle: a native codec's encode(Conversation) must be token-exact
// with the GGUF's own chat_template rendered by jinfer-jinja (add_generation_prompt=true - the
// codec always ends generate-ready) and rescanned with encodeWithSpecialTokens. The codec sibling
// of OracleScenario (which serves the per-turn TurnTemplate ports). Loads only the tokenizer,
// never the weights.
package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class CodecOracleScenario {

    public final Tokenizer tokenizer;
    private final com.qxotic.toknroll.Specials specials;
    public final ChatTemplate template;
    private final OracleSupport support;
    private final Map<String, Object> renderVars;
    private final Checks checks = new Checks();

    /**
     * Loads the GGUF's tokenizer + compiled chat template. {@code renderVars} are the extra
     * template variables (bos/eos/date) that pin the render deterministic.
     */
    public CodecOracleScenario(
            Path gguf,
            java.util.function.Function<Tokenizer, ChatTemplate> template,
            Map<String, Object> renderVars)
            throws Exception {
        this.support = new OracleSupport(gguf);
        this.tokenizer = support.tokenizer;
        this.specials = SpecialTokens.encoder(tokenizer);
        this.template = template.apply(tokenizer);
        this.renderVars = renderVars;
    }

    /** Token-exact comparison of the codec encode vs the rendered+rescanned template. */
    public void compare(String name, List<Message> conversation) {
        compare(name, true, Map.of(), conversation);
    }

    /**
     * Comparison with per-case overrides: {@code thinking} feeds the Conversation (and should be
     * mirrored in {@code extraVars} as {@code enable_thinking} for templates that branch on it).
     */
    public void compare(
            String name,
            boolean thinking,
            Map<String, Object> extraVars,
            List<Message> conversation) {
        List<Object> maps = new ArrayList<>();
        for (Message m : conversation) maps.add(OracleSupport.oracleMessage(m));
        Map<String, Object> vars = new HashMap<>(renderVars);
        vars.putAll(extraVars);
        vars.put("messages", maps);
        vars.put("add_generation_prompt", true);
        String rendered = support.jinja.render(vars);
        List<Integer> oracle = specials.encode(tokenizer, rendered).toList();
        List<Integer> ours = encodeIds(new Conversation(conversation, List.of(), thinking, ""));
        support.diff(checks, name, oracle, ours, rendered);
    }

    /**
     * Tool-aware comparison: the conversation may carry {@link
     * com.qxotic.jinfer.chat.Part.ToolCall} parts and tool-role turns, and {@code tools} is the
     * offered tool list (each Tool's rawJson parsed so Jinja's own {@code tojson} renders it).
     */
    public void compareTools(String name, List<Tool> tools, List<Message> conversation) {
        List<Object> maps = new ArrayList<>();
        for (Message m : conversation) maps.add(OracleSupport.oracleMessage(m));
        List<Object> toolVars = new ArrayList<>();
        for (Tool t : tools) toolVars.add(com.qxotic.format.json.Json.parse(t.rawJson()));
        Map<String, Object> vars = new HashMap<>(renderVars);
        vars.put("messages", maps);
        vars.put("tools", toolVars);
        vars.put("add_generation_prompt", true);
        String rendered = support.jinja.render(vars);
        List<Integer> oracle = specials.encode(tokenizer, rendered).toList();
        List<Integer> ours = encodeIds(new Conversation(conversation, tools, true, ""));
        support.diff(checks, name, oracle, ours, rendered);
    }

    /**
     * Like {@link #compareTools} but against an explicit expected rendered string (must include the
     * trailing generation prompt) instead of the Jinja engine - for constructs jinfer-jinja cannot
     * evaluate (LFM2's render_tool_calls macro).
     */
    public void compareToolsExpected(
            String name, String expectedRendered, List<Tool> tools, List<Message> conversation) {
        List<Integer> oracle = specials.encode(tokenizer, expectedRendered).toList();
        List<Integer> ours = encodeIds(new Conversation(conversation, tools, true, ""));
        support.diff(checks, name, oracle, ours, expectedRendered);
    }

    /** The codec's whole-conversation encoding, flattened to ids. */
    public List<Integer> encodeIds(Conversation conversation) {
        List<Integer> ids = new ArrayList<>();
        for (int id : Batch.tokenIds(template.encode(conversation))) ids.add(id);
        return ids;
    }

    // ---- helpers for the per-model injection-inertness checks ----

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
