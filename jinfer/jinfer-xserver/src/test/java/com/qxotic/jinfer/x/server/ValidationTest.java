package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ValidationTest {

    private static Map<String, Object> user(String text) {
        return Map.of("messages", List.of(Map.of("role", "user", "content", text)));
    }

    @Test
    void acceptsAWellFormedConversation() {
        Validation.validateChatRequest(user("hello"));
    }

    @Test
    void rejectsEmptyTextAndUnknownRoles() {
        assertThrows(IllegalArgumentException.class, () -> Validation.validateChatRequest(user(" ")));
        Map<String, Object> bad =
                Map.of("messages", List.of(Map.of("role", "wizard", "content", "hi")));
        assertTrue(
                assertThrows(
                                IllegalArgumentException.class,
                                () -> Validation.validateChatRequest(bad))
                        .getMessage()
                        .contains("wizard"));
    }

    @Test
    void validatesOnlyValuesTheClientProvided() {
        ServerConfig config = ServerConfig.local(0);
        Map<String, Object> request = new HashMap<>(user("hi"));
        Validation.validateGenerationParams(request, "model", config);
        request.put("max_tokens", -1);
        Validation.validateGenerationParams(request, "model", config);
        request.put("max_tokens", -2);
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(request, "model", config));
    }

    @Test
    void modelIsOptionalButCannotNameAnotherModel() {
        ServerConfig config = ServerConfig.local(0);
        Map<String, Object> request = new HashMap<>(user("hi"));
        Validation.validateGenerationParams(request, "maple", config);
        assertEquals("maple", Requests.modelId(request, "maple"));
        request.put("model", "other");
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(request, "maple", config));
    }

    @Test
    void nullLegacyBudgetDefersToNewSpelling() {
        Map<String, Object> request = new HashMap<>();
        request.put("max_tokens", null);
        request.put("max_completion_tokens", 12);
        assertEquals(12, Requests.budget(request));
    }

    @Test
    void forcedToolMustBeOffered() {
        Map<String, Object> required = new HashMap<>(user("hi"));
        required.put("tool_choice", "required");
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateChatRequest(required));

        Map<String, Object> named = new HashMap<>(user("hi"));
        named.put(
                "tools",
                List.of(
                        Map.of(
                                "type",
                                "function",
                                "function",
                                Map.of("name", "weather"))));
        named.put(
                "tool_choice",
                Map.of("type", "function", "function", Map.of("name", "missing")));
        assertThrows(IllegalArgumentException.class, () -> Validation.validateChatRequest(named));
    }
}
