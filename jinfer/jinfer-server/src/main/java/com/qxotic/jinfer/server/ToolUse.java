package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Server-side tool-calling policy: reads the OpenAI {@code tools}/{@code tool_choice} fields to
 * decide whether (and which) tool a request offers or forces, seeds a forced call into the prompt,
 * and parses the model's reply back into tool-call objects (via {@link ToolCalls}). The wire policy
 * around the tokenizer-level parsing — one cohesive place for "does this request use tools".
 */
final class ToolUse {

    private ToolUse() {}

    /** True when the request offers at least one tool and has not disabled tool use. */
    static boolean offered(Map<String, Object> request) {
        Object choice = request.get("tool_choice");
        if (choice instanceof String s && "none".equals(s)) return false;
        Object tools = request.get("tools");
        return tools instanceof List<?> list && !list.isEmpty();
    }

    /**
     * The function name a {@code tool_choice} forces ("" = any function via "required"), or null
     * when the request does not force a call.
     */
    static String forced(Map<String, Object> request) {
        if (!offered(request)) return null;
        Object choice = request.get("tool_choice");
        if (choice instanceof String s && "required".equals(s)) return "";
        if (choice instanceof Map<?, ?> map
                && map.get("function") instanceof Map<?, ?> fn
                && fn.get("name") instanceof String name) {
            return name;
        }
        return null;
    }

    /**
     * The assistant-turn text seeded by {@link #seedForced}; re-attached to the reply before
     * parsing so the seeded call parses whole.
     */
    static String forcedPrefix(Map<String, Object> request) {
        String choice = forced(request);
        if (choice == null) return "";
        return choice.isEmpty() ? ToolCalls.TC_START : ToolCalls.TC_START + "[" + choice;
    }

    /**
     * tool_choice "required"/named function: seed the assistant turn with {@code
     * <|tool_call_start|>} (plus "[name" for a named choice) so the model can only complete a tool
     * call instead of merely being asked to make one. The open paren is deliberately NOT seeded: a
     * bare trailing "(" lands on a tokenization boundary the model never saw (it merges with the
     * first argument in training data) and greedy decoding stops dead. No-op when the vocabulary
     * lacks the marker (the prompted fallback keeps its text hint). Pure: returns the (possibly
     * extended) prompt.
     */
    static IntSequence seedForced(
            Tokenizer tokenizer, Map<String, Object> request, IntSequence promptTokens) {
        String choice = forced(request);
        if (choice == null) return promptTokens;
        java.util.OptionalInt markerId = SpecialTokens.find(tokenizer, "<|tool_call_start|>");
        Integer marker = markerId.isPresent() ? markerId.getAsInt() : null;
        if (marker == null) return promptTokens;
        IntSequence seeded = promptTokens.concat(IntSequence.of(marker));
        if (!choice.isEmpty()) seeded = seeded.concat(tokenizer.encode("[" + choice));
        return seeded;
    }

    /**
     * Parses tool calls out of a finished reply, returning the reply re-tagged with them (or the
     * original reply when none parsed).
     */
    static Reply parse(LoadedModel<?> model, Reply reply, Map<String, Object> request) {
        // The model's own decoder already ran during generation (structured calls on token ids);
        // when it produced calls they are authoritative, so this string-scan fallback is only for
        // models with no native tool-call format (whole-render / no template).
        if (!reply.toolCalls().isEmpty()) return reply;
        // Parse from the FULL generated text (think span included). reply.text() is the
        // think-STRIPPED content, so a call the model emits before it closes </think> (or in an
        // unterminated think span) would be deleted before we ever see it. Decoding the raw
        // response tokens renders special tokens (<|tool_call_start|>, <think>) as literal text.
        String text = forcedPrefix(request);
        String decoded = model.tokenizer().decode(reply.tokens());
        text +=
                !decoded.strip().isEmpty()
                        ? decoded
                        : (reply.reasoning() != null ? reply.reasoning() + "\n" : "")
                                + reply.text();
        boolean debug = System.getProperty("jinfer.debugToolCalls") != null;
        List<Map<String, Object>> toolCalls = ToolCalls.parseToolCalls(text, names(request));
        if (toolCalls.isEmpty()) {
            // A reply that smells like an attempted call (markers or a "name" key) but parsed to
            // nothing is the diagnostic we care about; surface it even without the debug flag.
            String t = text.strip();
            if (t.contains(ToolCalls.TC_START) || t.contains("\"name\"")) {
                System.err.println(
                        "[tool-parse] tools offered but parsed 0 calls from reply: "
                                + t.replace("\n", "\\n"));
            }
            return reply;
        }
        if (debug)
            System.err.println(
                    "[tool-parse] found "
                            + toolCalls.size()
                            + " call(s) in: "
                            + text.strip().replace("\n", "\\n"));
        int marker = text.indexOf(ToolCalls.TC_START);
        return reply.asToolCalls(
                ToolCalls.fromWire(toolCalls), marker > 0 ? text.substring(0, marker).strip() : "");
    }

    /** The function names a request offers; calls naming anything else are dropped. */
    private static Set<String> names(Map<String, Object> request) {
        if (!(request.get("tools") instanceof List<?> tools)) return Set.of();
        Set<String> names = new HashSet<>();
        for (Object tool : tools) {
            if (tool instanceof Map<?, ?> t
                    && t.get("function") instanceof Map<?, ?> fn
                    && fn.get("name") instanceof String name) {
                names.add(name);
            }
        }
        return names;
    }
}
