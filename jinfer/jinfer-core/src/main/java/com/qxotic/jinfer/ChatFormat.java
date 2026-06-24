// Chat prompt formatting: turns a chat request into prompt tokens, either via the model's Jinja
// chat_template or the built-in ChatML fallback. One opaque, token-returning seam (ChatFormat)
// so the prompt source is a swappable detail — a hand-written Java format can be added here
// without touching the call sites. Independent of HTTP transport and generation.
package com.qxotic.jinfer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Builds prompt tokens for a chat request. Internal — never exposed by the OpenAI layer. */
interface ChatFormat {
    List<Integer> encode(ChatContext ctx);
}

/** The normalized inputs a ChatFormat needs, built once per request. {@code messages} and
 *  {@code tools} stay as raw OpenAI maps because Jinja templates read arbitrary fields. */
record ChatContext(List<Object> messages, List<Object> tools, Object toolChoice,
                   boolean addGenerationPrompt, boolean enableThinking,
                   Map<String, Object> kwargs) {
    boolean hasTools() {
        return tools != null && !tools.isEmpty();
    }
}

/** Selects the chat format for a model and holds the message-rendering helpers shared between the
 *  formats and the server's session/cache logic. */
final class ChatFormats {
    private ChatFormats() {
    }

    static ChatFormat forModel(LFMTokenizer tokenizer) {
        ChatTemplate tpl = tokenizer.chatTemplate();
        return tpl != null ? new JinjaChatFormat(tokenizer, tpl) : new ChatMLChatFormat(tokenizer);
    }

    /** Flattens one OpenAI message map to the model-facing text the ChatML format and the
     *  prompt-cache session key both use (so they stay byte-identical). */
    static String chatMessageContent(Map<String, Object> message) {
        String role = Values.stringValue(message.get("role"), "user");
        if ("tool".equals(role)) {
            String name = Values.stringValue(message.get("name"), Values.stringValue(message.get("tool_call_id"), "tool"));
            return "Tool result from " + name + ":\n" + Values.messageContent(message.get("content"));
        }
        String content = Values.messageContent(message.get("content"));
        Object toolCalls = message.get("tool_calls");
        if (toolCalls instanceof List<?> calls && !calls.isEmpty()) {
            String callsText = "Tool calls made:\n" + modelFacingJson(calls);
            return content.isEmpty() ? callsText : content + "\n" + callsText;
        }
        Object functionCall = message.get("function_call");
        if (functionCall instanceof Map<?, ?> call) {
            String callText = "Function call made:\n" + modelFacingJson(call);
            return content.isEmpty() ? callText : content + "\n" + callText;
        }
        return content;
    }

    /** Encodes one OpenAI message map as a ChatML turn (role header + flattened content). Shared
     *  by the ChatML format and incremental session resume so both stay byte-identical. */
    static List<Integer> encodeChatTurn(LFMChatFormat format, Object message) {
        Map<String, Object> map = Values.asObject(message, "message");
        LFMChatFormat.Role role = LFMChatFormat.Role.of(Values.stringValue(map.get("role"), "user"));
        return format.encodeMessage(new LFMChatFormat.Message(role, chatMessageContent(map)));
    }

    /** HuggingFace {@code apply_chat_template} pre-processes tool-call arguments from JSON
     *  strings into dicts so Jinja templates can call {@code .items()} on them.  This mirrors
     *  that normalization: every {@code tool_calls[*].function.arguments} string is parsed
     *  into a {@code Map<String,Object>} (non-strings and null are left alone). */
    static List<Object> preprocessToolCalls(List<Object> messages) {
        var out = new ArrayList<Object>(messages.size());
        for (Object raw : messages) {
            if (!(raw instanceof Map<?, ?> m)) { out.add(raw); continue; }
            @SuppressWarnings("unchecked")
            Map<String, Object> msg = new LinkedHashMap<>((Map<String, Object>) m);
            Object tc = msg.get("tool_calls");
            if (tc instanceof List<?> calls) {
                var parsed = new ArrayList<>(calls.size());
                for (Object c : calls) {
                    if (!(c instanceof Map<?, ?> cm)) { parsed.add(c); continue; }
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
                            } catch (RuntimeException ignored) { /* leave as string */ }
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
     * json.dumps-style JSON (", " and ": " separators) for text the MODEL reads. LFM2.5's
     * training data renders tool lists through Python/Jinja tojson, which spaces separators —
     * feeding the same content COMPACT puts it out of distribution and the model disowns its
     * tools ("I don't have access to real-time data"). API responses stay compact (Json).
     */
    private static String modelFacingJson(Object value) {
        StringBuilder sb = new StringBuilder();
        writeModelFacingJson(sb, value);
        return sb.toString();
    }

    private static void writeModelFacingJson(StringBuilder sb, Object value) {
        switch (value) {
            case null -> sb.append("null");
            case String s -> sb.append(JsonCodec.stringify(s)); // a bare string stringifies quoted+escaped
            case Number n -> sb.append(n);
            case Boolean b -> sb.append(b);
            case Map<?, ?> map -> {
                sb.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) sb.append(", ");
                    first = false;
                    writeModelFacingJson(sb, String.valueOf(entry.getKey()));
                    sb.append(": ");
                    writeModelFacingJson(sb, entry.getValue());
                }
                sb.append('}');
            }
            case Iterable<?> iterable -> {
                sb.append('[');
                boolean first = true;
                for (Object item : iterable) {
                    if (!first) sb.append(", ");
                    first = false;
                    writeModelFacingJson(sb, item);
                }
                sb.append(']');
            }
            default -> writeModelFacingJson(sb, String.valueOf(value));
        }
    }

    /** The text representation of a special token, or null if absent. */
    private static String specialTokenString(LFMTokenizer t, String name) {
        Integer id = t.getSpecialTokens().get(name);
        return id != null ? t.decode(id) : null;
    }

    /** The text of the first present special token among {@code names} (preferred name first),
     *  or null if none exist — e.g. {@code <bos>} with a {@code <|startoftext|>} fallback. */
    private static String firstSpecialString(LFMTokenizer t, String... names) {
        for (String name : names) {
            String text = specialTokenString(t, name);
            if (text != null) return text;
        }
        return null;
    }

    /** Renders the model's Jinja chat_template. The one place a rendered String is re-scanned
     *  into tokens (encodeWithSpecialTokens); every other format emits token ids directly. */
    private record JinjaChatFormat(LFMTokenizer tokenizer, ChatTemplate tpl) implements ChatFormat {
        public List<Integer> encode(ChatContext ctx) {
            var vars = new LinkedHashMap<String, Object>();
            vars.put("messages", preprocessToolCalls(ctx.messages()));
            vars.put("add_generation_prompt", ctx.addGenerationPrompt());
            vars.put("bos_token", firstSpecialString(tokenizer, "<bos>", "<|startoftext|>"));
            vars.put("eos_token", firstSpecialString(tokenizer, "<eos>", "<|endoftext|>"));
            vars.put("tools", ctx.tools());
            vars.put("enable_thinking", ctx.enableThinking());
            vars.put("preserve_thinking", false);
            if (ctx.kwargs() != null) vars.putAll(ctx.kwargs());
            return tokenizer.encodeWithSpecialTokens(tpl.render(vars));
        }
    }

    /** The built-in ChatML format for models without a Jinja template. Control tokens are written
     *  as ids by LFMChatFormat; only message content crosses through tokenizer.encode. */
    private static final class ChatMLChatFormat implements ChatFormat {
        private final LFMChatFormat chatml;

        ChatMLChatFormat(LFMTokenizer tokenizer) {
            this.chatml = new LFMChatFormat(tokenizer);
        }

        public List<Integer> encode(ChatContext ctx) {
            List<Integer> tokens = new ArrayList<>();
            tokens.add(chatml.beginOfSentence);
            List<Object> turns = ctx.messages();
            String systemText = null;
            if (!turns.isEmpty()) {
                Map<String, Object> first = Values.asObject(turns.getFirst(), "message");
                if ("system".equals(Values.stringValue(first.get("role"), ""))) {
                    systemText = chatMessageContent(first);
                    turns = turns.subList(1, turns.size());
                }
            }
            if (ctx.hasTools()) {
                tokens.addAll(encodeToolsSystemMessage(systemText, ctx));
            } else if (systemText != null) {
                tokens.addAll(chatml.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.SYSTEM, systemText)));
            }
            for (Object value : turns) {
                tokens.addAll(encodeChatTurn(chatml, value));
            }
            if (ctx.addGenerationPrompt()) {
                tokens.addAll(chatml.encodeGenerationPrompt());
                if (!ctx.enableThinking()) chatml.appendThinkSurrogate(tokens);
            }
            return tokens;
        }

        /** Per LFM2.5's training, tools render as a JSON array in the system message —
         *  {@code List of tools: [{...}, {...}]} — without any special-token markers. */
        private List<Integer> encodeToolsSystemMessage(String systemText, ChatContext ctx) {
            String lead = (systemText == null || systemText.isBlank()) ? "" : systemText + "\n\n";
            StringBuilder sb = new StringBuilder();
            sb.append(lead).append("List of tools: [");
            List<Object> tools = ctx.tools();
            for (int i = 0; i < tools.size(); i++) {
                if (i > 0) sb.append(", ");
                sb.append(modelFacingJson(tools.get(i)));
            }
            sb.append("]");
            String hints = toolChoiceHints(ctx.toolChoice());
            if (!hints.isEmpty()) sb.append("\n").append(hints);
            return chatml.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.SYSTEM, sb.toString()));
        }

        private static String toolChoiceHints(Object choice) {
            if (choice instanceof String s && "required".equals(s)) {
                return "A tool call is required.";
            }
            if (choice instanceof Map<?, ?> map && map.get("function") instanceof Map<?, ?> fn && fn.get("name") != null) {
                return "Call the tool named \"" + fn.get("name") + "\".";
            }
            return "";
        }
    }
}
