package com.qxotic.jinfer.x.server;

import java.util.List;
import java.util.Map;

/** The two pieces of OpenAI tool-choice policy needed before lowering to ChatEngine. */
final class ToolUse {
    private ToolUse() {}

    static boolean offered(Map<String, Object> request) {
        if ("none".equals(request.get("tool_choice"))) return false;
        return request.get("tools") instanceof List<?> tools && !tools.isEmpty();
    }

    /** Null = not forced, empty = any offered tool, otherwise the forced function name. */
    static String forced(Map<String, Object> request) {
        if (!offered(request)) return null;
        Object choice = request.get("tool_choice");
        if ("required".equals(choice)) return "";
        if (choice instanceof Map<?, ?> map
                && map.get("function") instanceof Map<?, ?> function
                && function.get("name") instanceof String name) return name;
        return null;
    }
}
