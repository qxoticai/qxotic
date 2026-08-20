package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.jinja.CompiledTemplate;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * Whole-render chat renderer: renders the model's own Jinja chat_template over the engine-built
 * request maps and re-scans the string into tokens - the fallback when a conversation cannot be
 * framed turn-stably by the model's native {@link ChatTemplate} (no hand-written codec, an
 * unsupported conversation shape, or {@code chat_template_kwargs}). Unlike a native codec it
 * re-scans a rendered String and bakes in the generation prompt, so it is one prompt, not turn
 * groups - no incremental caching. Constructed per model (parse once, render many times).
 *
 * <p>Reduced on purpose: per-request kwargs only (no global property), no raw-prompt seam. Media
 * never reaches here - the engine fails it loudly first.
 */
final class JinjaChatTemplate {

    private final Tokenizer tokenizer;
    private final CompiledTemplate template; // null: GGUF carries none (a parse failure throws)
    private final Specials specials; // compiled once per model
    private final List<String> specialNames; // longest-first, for the content scrub

    JinjaChatTemplate(Tokenizer tokenizer, String source) {
        this.tokenizer = tokenizer;
        // A template that fails to parse throws here, at load, naming the offending construct -
        // better a loud failure than a model silently chatting in foreign (ChatML) framing.
        this.template = source.isEmpty() ? null : JinjaRenderer.template(source);
        this.specials = SpecialTokens.encoder(tokenizer);
        // Think markers are exempt from the scrub: templates legitimately PROCESS them as text in
        // echoed history (content.split("</think>")), and a content-minted think id toggles
        // reasoning display, not roles - the injection vectors that matter are the turn/role
        // scaffold.
        this.specialNames =
                specials.tokens().stream()
                        .filter(n -> !n.equals("<think>") && !n.equals("</think>"))
                        .sorted((a, b) -> b.length() - a.length())
                        .toList();
    }

    /**
     * Renders the request maps to prompt tokens. {@code kwargs} merges extra template variables
     * ({@code chat_template_kwargs}); per-request keys win over the engine's defaults. Falls back
     * to a best-effort ChatML framing only when the GGUF carries no template at all — a template
     * the parser rejects fails the model load instead (see {@link JinjaRenderer#template}).
     */
    IntSequence render(
            List<Object> messages,
            List<Object> tools,
            boolean addGenerationPrompt,
            boolean enableThinking,
            Map<String, Object> kwargs) {
        // The whole-render path re-scans the rendered string with special-token awareness, so
        // content could otherwise mint control ids (llama.cpp ships this hole unmitigated).
        // Scrub special-token strings out of every request-supplied string first; the template's
        // own scaffold is added AFTER the scrub and re-scans as intended.
        messages = scrubbed(messages);
        tools = tools == null ? null : scrubbed(tools);
        CompiledTemplate tpl = template;
        if (tpl == null) {
            return chatMl(messages, tools, addGenerationPrompt);
        }
        var vars = new LinkedHashMap<String, Object>();
        vars.put("messages", preprocessToolCalls(messages));
        vars.put("add_generation_prompt", addGenerationPrompt);
        // A template that opens with {{ bos_token }} - Llama 3's does - printed the literal string
        // "None" when this bound null, putting four characters of garbage at the very front of
        // every whole-render prompt. TWO things fixed that: the spellings live in one table in
        // SpecialTokens (which is what taught it <|begin_of_text|>), and an absent token binds ""
        // rather than null, so a vocabulary this build has never seen renders NOTHING instead of
        // garbage. "" is falsy in Jinja exactly as None is, so {% if bos_token %} is unaffected.
        vars.put("bos_token", specialString(SpecialTokens.bos(tokenizer)));
        vars.put("eos_token", specialString(SpecialTokens.eos(tokenizer)));
        vars.put("tools", tools);
        vars.put("enable_thinking", enableThinking);
        vars.put("preserve_thinking", false);
        if (kwargs != null) vars.putAll(kwargs);
        String rendered = tpl.render(vars);
        // Prompt-opened thinking for whole-render families: a /think scaffold that does NOT open
        // the span itself leaves the model to mint <think> from a bare role header - and a
        // degenerate checkpoint instead closes the turn on its FIRST token (SmolLM3 Q4_K_M,
        // greedy: one <|im_end|>, an empty reply). Open it for them; the engine arms the thinking
        // cap and seeds the parser from exactly this tail. Skip when the render already left a
        // span open (Qwen3-style scaffolds end with "<think>\n").
        if (enableThinking
                && addGenerationPrompt
                && SpecialTokens.find(tokenizer, Thinking.OPEN).isPresent()
                && rendered.lastIndexOf(Thinking.OPEN) <= rendered.lastIndexOf(Thinking.CLOSE)) {
            rendered += Thinking.OPEN;
        }
        return specials.encode(tokenizer, rendered);
    }

    /**
     * Best-effort ChatML fallback for GGUFs without a chat_template: {@code
     * <|im_start|>role\ncontent<|im_end|>\n} per message, tools flattened into the system turn,
     * tool results/calls rendered as text. The string is re-scanned with special-token awareness,
     * so the turn markers become real ids when the vocab has them. The maps are engine-built, so
     * message content is always a plain String here.
     */
    private IntSequence chatMl(
            List<Object> messages, List<Object> tools, boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();
        StringBuilder system = new StringBuilder();
        if (tools != null && !tools.isEmpty()) {
            system.append("List of tools: ").append(JsonCodec.stringify(tools));
        }
        List<Object> body = new ArrayList<>();
        for (Object raw : messages) {
            if (raw instanceof Map<?, ?> m && "system".equals(m.get("role"))) {
                String text = contentText(m.get("content"));
                if (!text.isEmpty()) system.insert(0, system.isEmpty() ? text : text + "\n");
                continue;
            }
            body.add(raw);
        }
        if (!system.isEmpty()) {
            sb.append("<|im_start|>system\n").append(system).append("<|im_end|>\n");
        }
        for (Object raw : body) {
            if (!(raw instanceof Map<?, ?> m)) continue;
            String role = m.get("role") instanceof String s ? s : "user";
            String content = contentText(m.get("content"));
            if ("tool".equals(role)) {
                Object name = m.get("name") != null ? m.get("name") : m.get("tool_call_id");
                role = "user";
                content = "Tool result from " + (name != null ? name : "tool") + ":\n" + content;
            } else if (m.get("tool_calls") instanceof List<?> calls && !calls.isEmpty()) {
                String callsText = "Tool calls made:\n" + JsonCodec.stringify(calls);
                content = content.isEmpty() ? callsText : content + "\n" + callsText;
            }
            sb.append("<|im_start|>")
                    .append(role)
                    .append('\n')
                    .append(content)
                    .append("<|im_end|>\n");
        }
        if (addGenerationPrompt) sb.append("<|im_start|>assistant\n");
        return specials.encode(tokenizer, sb.toString());
    }

    private static String contentText(Object content) {
        return content instanceof String s ? s : "";
    }

    /**
     * HuggingFace {@code apply_chat_template} pre-processes tool-call arguments from JSON strings
     * into dicts so Jinja templates can call {@code .items()} on them. This mirrors that
     * normalization: every {@code tool_calls[*].function.arguments} string is parsed into a {@code
     * Map<String,Object>} (non-strings and null are left alone).
     */
    static List<Object> preprocessToolCalls(List<Object> messages) {
        var out = new ArrayList<Object>(messages.size());
        for (Object raw : messages) {
            if (!(raw instanceof Map<?, ?> m)) {
                out.add(raw);
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> msg = new LinkedHashMap<>((Map<String, Object>) m);
            Object tc = msg.get("tool_calls");
            if (tc instanceof List<?> calls) {
                var parsed = new ArrayList<>(calls.size());
                for (Object c : calls) {
                    if (!(c instanceof Map<?, ?> cm)) {
                        parsed.add(c);
                        continue;
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, Object> call = new LinkedHashMap<>((Map<String, Object>) cm);
                    Object fn = call.get("function");
                    if (fn instanceof Map<?, ?> fm) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> func = new LinkedHashMap<>((Map<String, Object>) fm);
                        Object args = func.get("arguments");
                        if (args instanceof String s && !s.isEmpty()) {
                            try {
                                Object parsedArgs = JsonCodec.parse(s);
                                if (parsedArgs instanceof Map<?, ?> pm) {
                                    @SuppressWarnings("unchecked")
                                    Map<String, Object> argsMap = (Map<String, Object>) pm;
                                    func.put("arguments", argsMap);
                                }
                            } catch (RuntimeException ignored) {
                                /* leave as string */
                            }
                        }
                        call.put("function", func);
                    }
                    parsed.add(call);
                }
                msg.put("tool_calls", parsed);
            }
            out.add(msg);
        }
        return out;
    }

    /** The text of a resolved special token, or "" when this vocabulary has none. */
    private String specialString(OptionalInt id) {
        return id.isPresent() ? tokenizer.decode(new int[] {id.getAsInt()}) : "";
    }

    /**
     * A deep copy of {@code values} with every String scrubbed: any embedded special-token string
     * gets a zero-width space after its first character, breaking the longest-match rescan without
     * visibly changing the text. Content can then never mint control ids through the whole-render
     * path. Keys and non-string scalars pass through untouched.
     */
    @SuppressWarnings("unchecked")
    private <T> List<T> scrubbed(List<T> values) {
        return (List<T>) scrubValue(values, specialNames);
    }

    static Object scrubValue(Object value, List<String> names) {
        // Identity fast path: the common clean request allocates nothing - a node is copied only
        // when a descendant string actually changed.
        if (value instanceof String s) return scrub(s, names);
        if (value instanceof List<?> list) {
            ArrayList<Object> out = null;
            for (int i = 0; i < list.size(); i++) {
                Object scrubbed = scrubValue(list.get(i), names);
                if (out == null && scrubbed != list.get(i)) {
                    out = new ArrayList<>(list.subList(0, i));
                }
                if (out != null) out.add(scrubbed);
            }
            return out != null ? out : value;
        }
        if (value instanceof Map<?, ?> map) {
            LinkedHashMap<Object, Object> out = null;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                Object scrubbed = scrubValue(e.getValue(), names);
                if (out == null && scrubbed != e.getValue()) {
                    out = new LinkedHashMap<>();
                    for (Map.Entry<?, ?> prior : map.entrySet()) {
                        if (prior.getKey().equals(e.getKey())) break;
                        out.put(prior.getKey(), prior.getValue());
                    }
                }
                if (out != null) out.put(e.getKey(), scrubbed);
            }
            return out != null ? out : value;
        }
        return value;
    }

    static String scrub(String text, List<String> names) {
        String out = text;
        for (String name : names) {
            if (out.contains(name)) {
                out = out.replace(name, name.charAt(0) + "\u200b" + name.substring(1));
            }
        }
        return out; // == text (same reference) when nothing matched
    }
}
