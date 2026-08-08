package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Validation runs on the handler thread, BEFORE a request is queued, so everything it rejects costs
 * no model time and never reaches the worker. That makes it the server's cheapest and most
 * load-bearing defence, and it needs no model to test - which matters, because the battery that
 * covers the rest of this module is opt-in and does not run in a normal build.
 *
 * <p>Every rejection must be an {@link IllegalArgumentException}: the request handler maps exactly
 * that (and {@code UnsupportedOperationException}) to 400, and anything else to a 500 that reports
 * a server defect. A validator throwing the wrong type would answer a malformed request with
 * "Internal server error".
 */
class ValidationTest {

    private static Map<String, Object> user(String content) {
        return Map.of("messages", List.of(Map.of("role", "user", "content", content)));
    }

    private static String rejects(Map<String, Object> request) {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Validation.validateChatRequest(request),
                        "expected a 400-mapped rejection");
        assertTrue(e.getMessage() != null && !e.getMessage().isBlank(), "rejection needs a reason");
        return e.getMessage();
    }

    @Test
    void aWellFormedRequestPasses() {
        Validation.validateChatRequest(user("hello"));
    }

    @Test
    void emptyOrSubstanceLessConversationsAreRejected() {
        rejects(Map.of("messages", List.of()));
        // whitespace is not substance: an all-blank conversation would prefill and generate from
        // nothing, burning a slot to produce noise
        rejects(user("   "));
        rejects(user(""));
    }

    @Test
    void unknownRolesAreRejectedAndNamed() {
        String message =
                rejects(Map.of("messages", List.of(Map.of("role", "wizard", "content", "hi"))));
        assertTrue(
                message.contains("wizard"), "the message must name the offending role: " + message);
    }

    @Test
    void aToolCallCountsAsSubstanceEvenWithEmptyContent() {
        // an assistant turn replaying a tool call carries no text, and must not read as empty
        Validation.validateChatRequest(
                Map.of(
                        "messages",
                        List.of(
                                Map.of("role", "user", "content", "hi"),
                                Map.of(
                                        "role",
                                        "assistant",
                                        "content",
                                        "",
                                        "tool_calls",
                                        List.of(Map.of("id", "c1"))))));
    }

    @Test
    void unsupportedResponseFormatsAreRejected() {
        Map<String, Object> request =
                Map.of(
                        "messages",
                        List.of(Map.of("role", "user", "content", "hi")),
                        "response_format",
                        Map.of("type", "json_schema"));
        assertTrue(rejects(request).contains("json_schema"));
    }

    /**
     * -1 means context-bounded everywhere in jinfer (ChatEngine.Request, the CLI's -n), and it is
     * the CLI's default when -n is not given - so it must pass here too, sent explicitly OR as the
     * fallback for an omitted max_tokens. This guard once required {@code 0 <=} of the RESOLVED
     * value, which made a stock {@code --server} reject every request that did not name a budget.
     */
    @Test
    void contextBoundedMaxTokensIsAccepted() {
        ServerConfig config = ServerTestSupport.config(Path.of("model.gguf"));
        Map<String, Object> request = new HashMap<>(user("hi"));
        request.put("model", config.modelName());
        request.put("max_tokens", -1);
        Validation.validateGenerationParams(request, config);

        // omitted max_tokens resolves to the server's own default, which may itself be -1
        ServerConfig unbounded =
                new ServerConfig(
                        config.modelName(),
                        config.bind(),
                        new ServerConfig.Defaults(config.defaults().sampling(), -1, true, false),
                        config.limits(),
                        config.cache());
        Map<String, Object> omitted = new HashMap<>(user("hi"));
        omitted.put("model", config.modelName());
        Validation.validateGenerationParams(omitted, unbounded);

        request.put("max_tokens", -2); // below the sentinel stays malformed
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Validation.validateGenerationParams(request, config));
        assertTrue(e.getMessage().contains("max_tokens"), e.getMessage());
    }

    /**
     * This server has ONE model, so an absent "model" can only mean that one - the field is
     * optional and {@link Requests#modelId} resolves it. Naming the wrong model is still refused:
     * that is a client pointed at the wrong server, which is worth saying out loud.
     */
    @Test
    void theModelFieldIsOptionalButNeverWrong() {
        ServerConfig config = ServerTestSupport.config(Path.of("model.gguf"));
        Validation.validateGenerationParams(new HashMap<>(user("hi")), config);
        assertEquals(
                config.modelName(),
                Requests.modelId(new HashMap<>(user("hi")), config),
                "an absent model resolves to the served one");

        Map<String, Object> blank = new HashMap<>(user("hi"));
        blank.put("model", "");
        Validation.validateGenerationParams(blank, config);
        assertEquals(
                config.modelName(),
                Requests.modelId(blank, config),
                "blank resolves too - echoing \"model\": \"\" back helps nobody");

        Map<String, Object> wrong = new HashMap<>(user("hi"));
        wrong.put("model", "some-other-model");
        String message =
                assertThrows(
                                IllegalArgumentException.class,
                                () -> Validation.validateGenerationParams(wrong, config))
                        .getMessage();
        assertTrue(message.contains("some-other-model"), message);
        assertTrue(
                message.contains(config.modelName()),
                "the refusal names what IS served: " + message);

        Map<String, Object> typed = new HashMap<>(user("hi"));
        typed.put("model", 42);
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(typed, config),
                "a non-string model is a malformed request, not a missing one");
    }

    /**
     * A range rule here is a rule about what a CLIENT may ask for. Checking the RESOLVED value
     * instead turned the operator's configuration into the client's error: a server started with
     * {@code --temp 9} (legal for the engine - Sampling caps only at 0) answered "temperature must
     * be within [0, 2]" to every request that did not override it. Same shape as the max_tokens
     * default that made a stock --server refuse everything.
     */
    @Test
    void serverDefaultsAreNeverTheClientsFault() {
        ServerConfig base = ServerTestSupport.config(Path.of("model.gguf"));
        ServerConfig odd =
                new ServerConfig(
                        base.modelName(),
                        base.bind(),
                        new ServerConfig.Defaults(
                                new com.qxotic.jinfer.llm.Sampling(9f, 1f, 0, 0f, 42L),
                                -1,
                                true,
                                false),
                        base.limits(),
                        base.cache());
        // says nothing about sampling: the request is well formed whatever the server's defaults
        Validation.validateGenerationParams(new HashMap<>(user("hi")), odd);

        // ... and the caps still bind what the request DOES carry
        for (Map.Entry<String, Object> bad :
                Map.of("temperature", (Object) 2.5, "top_p", 1.5, "top_k", -1, "min_p", 1.5)
                        .entrySet()) {
            Map<String, Object> request = new HashMap<>(user("hi"));
            request.put(bad.getKey(), bad.getValue());
            String message =
                    assertThrows(
                                    IllegalArgumentException.class,
                                    () -> Validation.validateGenerationParams(request, odd),
                                    bad.getKey() + " out of range must still be refused")
                            .getMessage();
            assertTrue(message.contains(bad.getKey()), message);
        }

        // an explicit null is "unset", which is how the OpenAI SDKs serialise an omitted field
        Map<String, Object> nulled = new HashMap<>(user("hi"));
        nulled.put("temperature", null);
        nulled.put("max_tokens", null);
        Validation.validateGenerationParams(nulled, odd);
    }

    /**
     * The OpenAI SDKs serialise an unset field as an explicit null, so {@code {"max_tokens": null,
     * "max_completion_tokens": 100}} is what a client that only set the newer name actually sends.
     * {@code getOrDefault} returns the STORED null for a present key, so both the validator and the
     * resolver fell through to the server's default and threw the client's 100 away.
     */
    @Test
    void anExplicitNullBudgetDefersToTheOtherSpelling() {
        Map<String, Object> both = new HashMap<>();
        both.put("max_tokens", null);
        both.put("max_completion_tokens", 100);
        assertEquals(100, Requests.budget(both), "the spelling that carries a value wins");

        Map<String, Object> legacy = new HashMap<>();
        legacy.put("max_tokens", 8);
        legacy.put("max_completion_tokens", 99);
        assertEquals(8, Requests.budget(legacy), "max_tokens wins when both are set");

        Map<String, Object> unset = new HashMap<>();
        unset.put("max_tokens", null);
        unset.put("max_completion_tokens", null);
        assertEquals(null, Requests.budget(unset), "both null is no budget at all");
        assertEquals(null, Requests.budget(new HashMap<>()), "absent is no budget either");

        // and a bad value in the SURVIVING spelling is still refused
        ServerConfig config = ServerTestSupport.config(Path.of("model.gguf"));
        Map<String, Object> bad = new HashMap<>(user("hi"));
        bad.put("max_tokens", null);
        bad.put("max_completion_tokens", -5);
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(bad, config));
    }

    @Test
    void everyRejectionIsTheTypeTheHandlerMapsTo400() {
        // the contract in one place: if this list grows a case that throws something else, the
        // server answers a bad request with "Internal server error" instead of an explanation
        List<Map<String, Object>> bad =
                List.of(
                        Map.of("messages", List.of()),
                        user("   "),
                        Map.of("messages", List.of(Map.of("role", "wizard", "content", "hi"))));
        for (Map<String, Object> request : bad) {
            Throwable thrown =
                    assertThrows(Throwable.class, () -> Validation.validateChatRequest(request));
            assertEquals(
                    IllegalArgumentException.class,
                    thrown.getClass(),
                    "validation must throw the 400-mapped type, not " + thrown.getClass());
        }
    }
}
