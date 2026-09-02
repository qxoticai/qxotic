package com.qxotic.jinfer.server;

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
        assertThrows(
                IllegalArgumentException.class, () -> Validation.validateChatRequest(user(" ")));
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
    void topPRangeIsTheEnginesRange() {
        // the validator used to admit 0, which Sampling refuses after the request was accepted
        ServerConfig config = ServerConfig.local(0);
        for (Object bad : List.of(0, 0.0, -0.1, 1.5)) {
            Map<String, Object> request = new HashMap<>(user("hi"));
            request.put("top_p", bad);
            assertTrue(
                    assertThrows(
                                    IllegalArgumentException.class,
                                    () ->
                                            Validation.validateGenerationParams(
                                                    request, "model", config))
                            .getMessage()
                            .contains("top_p must be within (0, 1]"),
                    "top_p=" + bad);
        }
        for (Object fine : List.of(1e-6, 0.5, 1, 1.0)) {
            Map<String, Object> request = new HashMap<>(user("hi"));
            request.put("top_p", fine);
            Validation.validateGenerationParams(request, "model", config);
        }
    }

    @Test
    void rejectsEmptyStopStrings() {
        // "" would stop every reply at its first fragment, an empty answer with finish "stop"
        ServerConfig config = ServerConfig.local(0);
        for (Object stop : List.of("", List.of(""), List.of("###", ""))) {
            Map<String, Object> request = new HashMap<>(user("hi"));
            request.put("stop", stop);
            assertTrue(
                    assertThrows(
                                    IllegalArgumentException.class,
                                    () ->
                                            Validation.validateGenerationParams(
                                                    request, "model", config))
                            .getMessage()
                            .contains("stop strings must not be empty"),
                    "stop=" + stop);
        }
        Map<String, Object> fine = new HashMap<>(user("hi"));
        fine.put("stop", List.of("###"));
        Validation.validateGenerationParams(fine, "model", config);
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
    void rejectsTypeConfusionAtTheHttpBoundary() {
        ServerConfig config = ServerConfig.local(0);
        for (String field : List.of("n", "top_k", "max_tokens", "seed")) {
            Map<String, Object> request = new HashMap<>(user("hi"));
            request.put(field, 1.5);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Validation.validateGenerationParams(request, "model", config),
                    field);
            request.put(field, true);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Validation.validateGenerationParams(request, "model", config),
                    field);
        }

        Map<String, Object> temperature = new HashMap<>(user("hi"));
        temperature.put("temperature", true);
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(temperature, "model", config));

        Map<String, Object> stream = new HashMap<>(user("hi"));
        stream.put("stream", "true");
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(stream, "model", config));

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Validation.validateChatRequest(
                                Map.of(
                                        "messages",
                                        List.of(Map.of("role", "user", "content", 123)))));
    }

    @Test
    void treatsExplicitNullOptionsAsUnset() {
        Map<String, Object> request = new HashMap<>(user("hi"));
        for (String field :
                List.of(
                        "grammar",
                        "response_format",
                        "logprobs",
                        "top_logprobs",
                        "logit_bias",
                        "frequency_penalty",
                        "presence_penalty")) {
            request.put(field, null);
        }
        ServerConfig local = ServerConfig.local(0);
        ServerConfig.Limits defaults = local.limits();
        ServerConfig config =
                new ServerConfig(
                        local.bind(),
                        local.defaults(),
                        new ServerConfig.Limits(
                                defaults.threads(),
                                defaults.queueCapacity(),
                                defaults.maxBodyBytes(),
                                false,
                                defaults.writeTimeout(),
                                defaults.requestTimeout(),
                                defaults.shutdownTimeout()),
                        local.access());

        Validation.validateGenerationParams(request, "model", config);

        request.put("grammar", "");
        request.put("response_format", Map.of("type", "text"));
        Validation.validateGenerationParams(request, "model", config);

        request.put("response_format", Map.of("type", "json_object"));
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(request, "model", config));
    }

    @Test
    void forcedToolMustBeOffered() {
        Map<String, Object> required = new HashMap<>(user("hi"));
        required.put("tool_choice", "required");
        assertThrows(
                IllegalArgumentException.class, () -> Validation.validateChatRequest(required));

        Map<String, Object> named = new HashMap<>(user("hi"));
        named.put(
                "tools",
                List.of(Map.of("type", "function", "function", Map.of("name", "weather"))));
        named.put("tool_choice", Map.of("type", "function", "function", Map.of("name", "missing")));
        assertThrows(IllegalArgumentException.class, () -> Validation.validateChatRequest(named));
    }

    @Test
    void toolsAndConstrainedOutputAreMutuallyExclusive() {
        Map<String, Object> request = new HashMap<>(user("return JSON"));
        request.put(
                "tools",
                List.of(Map.of("type", "function", "function", Map.of("name", "weather"))));
        request.put("response_format", Map.of("type", "json_object"));
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Validation.validateGenerationParams(
                                        request, "model", ServerConfig.local(0)));
        assertTrue(error.getMessage().contains("cannot be used"), error.getMessage());

        request.put("tool_choice", "none");
        Validation.validateGenerationParams(request, "model", ServerConfig.local(0));
    }

    @Test
    void acceptsEverySupportedMediaInputWithoutKnowingTheModel() {
        Map<String, Object> request =
                Map.of(
                        "messages",
                        List.of(
                                Map.of(
                                        "role",
                                        "user",
                                        "content",
                                        List.of(
                                                Map.of(
                                                        "type",
                                                        "input_image",
                                                        "image_url",
                                                        "data:image/png;base64,AA=="),
                                                Map.of(
                                                        "type",
                                                        "input_audio",
                                                        "input_audio",
                                                        Map.of("data", "AA==", "format", "wav")),
                                                Map.of(
                                                        "type",
                                                        "video_url",
                                                        "video_url",
                                                        "data:video/mp4;base64,AA==")))));

        Validation.validateChatRequest(request);

        Map<String, Object> unknown =
                Map.of(
                        "messages",
                        List.of(
                                Map.of(
                                        "role",
                                        "user",
                                        "content",
                                        List.of(Map.of("type", "input_file")))));
        assertThrows(IllegalArgumentException.class, () -> Validation.validateChatRequest(unknown));
    }

    @Test
    void rejectsChatOptionsWhoseSemanticsAreNotImplemented() {
        Validation.validateChatOptions(Map.of());
        Validation.validateChatOptions(Map.of("modalities", List.of("text")));
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateChatOptions(Map.of("modalities", List.of("audio"))));
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateChatOptions(Map.of("audio", Map.of())));
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateChatOptions(Map.of("prediction", Map.of())));
    }

    @Test
    void rejectsResponsesOptionsWhoseSemanticsAreNotImplemented() {
        Validation.validateResponseOptions(
                Map.of(
                        "background",
                        false,
                        "store",
                        false,
                        "truncation",
                        "disabled",
                        "include",
                        List.of()));

        for (Map<String, Object> request :
                List.<Map<String, Object>>of(
                        Map.of("previous_response_id", "resp_1"),
                        Map.of("reasoning", Map.of("effort", "low")),
                        Map.of("background", true),
                        Map.of("store", true),
                        Map.of("truncation", "auto"),
                        Map.of("include", List.of("message.input_image.image_url")))) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Validation.validateResponseOptions(request),
                    request.toString());
        }
    }

    @Test
    void reasoningEffortTakesOnlyTheKnownLevels() {
        ServerConfig config = ServerConfig.local(0);
        Map<String, Object> request = new HashMap<>(user("hi"));
        request.put("reasoning_effort", "none");
        Validation.validateGenerationParams(request, "model", config);
        request.put("reasoning_effort", "bogus");
        assertThrows(
                IllegalArgumentException.class,
                () -> Validation.validateGenerationParams(request, "model", config));
    }
}
