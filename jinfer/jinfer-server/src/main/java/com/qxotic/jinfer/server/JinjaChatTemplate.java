// Whole-render chat renderer: renders the model's own Jinja chat_template over the OpenAI request
// and re-scans the string into tokens. The server's fallback when a request cannot be framed
// turn-stably by the model's TurnTemplate - a model with no hand-written template, tools on a model
// without supportsTools(), a forced tool_choice, or arbitrary chat_template_kwargs. Unlike a
// TurnTemplate it re-scans a rendered String and bakes in the generation prompt, so it is one
// prompt, not turn groups - no incremental caching. Plain chat on covered models goes through the
// model's TurnTemplate instead.
package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.jinja.CompiledTemplate;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Renders a chat request to prompt tokens through the model's Jinja chat_template - the one place a
 * rendered String is re-scanned into tokens (encodeWithSpecialTokens); the {@code TurnTemplate}
 * path emits token ids directly. Constructed per model with its tokenizer; compiles the GGUF's
 * template source once at construction (parse once, render many times) - hold one instance per
 * model, not per request.
 */
final class JinjaChatTemplate {

    private final Tokenizer tokenizer;
    private final CompiledTemplate template; // null: GGUF carries none or it failed to parse
    private final com.qxotic.toknroll.Specials specials; // compiled once per model
    private final List<String> specialNames; // longest-first, for the content scrub

    JinjaChatTemplate(Tokenizer tokenizer, String source) {
        this.tokenizer = tokenizer;
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
     * Render the OpenAI request maps to prompt tokens. {@code messages}/{@code tools} stay as raw
     * maps because Jinja templates read arbitrary fields; {@code kwargs} merges extra template
     * variables (chat_template_kwargs). Falls back to a best-effort ChatML framing when the GGUF
     * has no compilable template.
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
        vars.put("bos_token", firstSpecialString(tokenizer, "<bos>", "<|startoftext|>"));
        vars.put("eos_token", firstSpecialString(tokenizer, "<eos>", "<|endoftext|>"));
        vars.put("tools", tools);
        vars.put("enable_thinking", enableThinking);
        vars.put("preserve_thinking", false);
        if (PROPERTY_KWARGS != null) vars.putAll(PROPERTY_KWARGS);
        if (kwargs != null) vars.putAll(kwargs); // per-request kwargs win over the property
        return specials.encode(tokenizer, tpl.render(vars));
    }

    /** Raw-prompt authoring: specials map to ids (the compiled per-model encoder, reused). */
    IntSequence encodeRaw(String text) {
        return specials.encode(tokenizer, text);
    }

    /**
     * Extra template variables from {@code -Djinfer.chatTemplateKwargs} (a JSON object, e.g. {@code
     * {"keep_past_thinking": true}}), applied to every whole-render request under any per-request
     * {@code chat_template_kwargs}. Parsed once; a malformed value fails fast at startup.
     */
    private static final Map<String, Object> PROPERTY_KWARGS = propertyKwargs();

    @SuppressWarnings("unchecked")
    private static Map<String, Object> propertyKwargs() {
        String json = System.getProperty("jinfer.chatTemplateKwargs");
        if (json == null || json.isBlank()) return null;
        Object parsed = JsonCodec.parse(json);
        if (!(parsed instanceof Map))
            throw new IllegalArgumentException(
                    "-Djinfer.chatTemplateKwargs must be a JSON object: " + json);
        return (Map<String, Object>) parsed;
    }

    /**
     * Best-effort ChatML fallback for GGUFs without a compilable chat_template (the framing the
     * pre-refactor server used): {@code <|im_start|>role\ncontent<|im_end|>\n} per message, tools
     * flattened into the system turn, tool results/calls rendered as text. The string is re-scanned
     * with special-token awareness, so the turn markers become real ids when the vocab has them.
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
                String text = Values.messageContent(m.get("content"));
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
            String role = Values.stringValue(m.get("role"), "user");
            String content = Values.messageContent(m.get("content"));
            if ("tool".equals(role)) {
                String name =
                        Values.stringValue(
                                m.get("name"), Values.stringValue(m.get("tool_call_id"), "tool"));
                role = "user";
                content = "Tool result from " + name + ":\n" + content;
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

    /**
     * The text of the first present special token among {@code names} (preferred name first), or
     * null if none exist — e.g. {@code <bos>} with a {@code <|startoftext|>} fallback.
     */
    private static String firstSpecialString(Tokenizer t, String... names) {
        java.util.OptionalInt id = SpecialTokens.findFirst(t, names);
        return id.isPresent() ? t.decode(new int[] {id.getAsInt()}) : null;
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
