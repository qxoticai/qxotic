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

    private final GgufTokenizer tokenizer;
    private final CompiledTemplate template; // null: GGUF carries none or it failed to parse

    JinjaChatTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        String source = tokenizer.chatTemplateSource();
        this.template = source.isEmpty() ? null : JinjaRenderer.template(source);
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
        if (kwargs != null) vars.putAll(kwargs);
        return tokenizer.encodeWithSpecialTokens(tpl.render(vars));
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
        return tokenizer.encodeWithSpecialTokens(sb.toString());
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

    /** The text representation of a special token, or null if absent. */
    private static String specialTokenString(GgufTokenizer t, String name) {
        Integer id = t.getSpecialTokens().get(name);
        return id != null ? t.decode(id) : null;
    }

    /**
     * The text of the first present special token among {@code names} (preferred name first), or
     * null if none exist — e.g. {@code <bos>} with a {@code <|startoftext|>} fallback.
     */
    private static String firstSpecialString(GgufTokenizer t, String... names) {
        for (String name : names) {
            String text = specialTokenString(t, name);
            if (text != null) return text;
        }
        return null;
    }
}
