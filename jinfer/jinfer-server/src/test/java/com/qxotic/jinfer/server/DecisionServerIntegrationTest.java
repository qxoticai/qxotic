package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.util.Base64;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code POST /v1/systemone} over Liquid d1-2B and its vision tower. The numbers are llama.cpp's
 * answers to the same requests on the same files (2026-11-10), the tolerance two engines' Q8_0
 * kernels.
 */
@Tag("integration")
class DecisionServerIntegrationTest {

    private static final String D1 = "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf";
    private static final String MMPROJ = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
    private static final double TOLERANCE = 2e-2;

    /** As the adapters' alone-or-together tests pin them, for the same ticket and questions. */
    private static final int EACH_TOKENS = 203, PACKED_TOKENS = 139;

    private final HttpClient client = HttpClient.newHttpClient();

    @Test
    @SuppressWarnings("unchecked")
    void answersMatchLlamaCpp() throws Exception {
        Arena arena = Arenas.newCrossThread();
        try {
            DecisionModel d1 =
                    Models.loadDecisionModel(
                            TestModels.require(D1),
                            arena,
                            Map.of("media", TestModels.require(MMPROJ)));
            try (var server = DecisionServer.start(d1, 2048, "d1", ServerConfig.local(0))) {
                String url = "http://127.0.0.1:" + server.address().getPort() + "/v1/systemone";

                // an object state is read as JSON, a member a line
                var answers =
                        (Map<String, Object>)
                                post(
                                                url,
                                                """
                                                {"state": {"customer": "Ana", "order": 4411, "total": 42.5,
                                                           "vip": true, "note": null, "tags": ["billing", "repeat"],
                                                           "message": "I was charged twice for my order and I want my money back today."},
                                                 "questions": {
                                                   "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                                                   "team": {"type": "choice", "instructions": "Which team should handle this?",
                                                            "criteria": {"billing": "Charges and refunds", "shipping": "Delivery problems",
                                                                         "technical": "Product does not work", "other": null}},
                                                   "urgency": {"type": "score", "instructions": "How urgent is this?",
                                                               "criteria": ["Can wait", "This week", "Today", "Right now"]}}}
                                                """)
                                        .body()
                                        .get("answers");
                assertEquals(0.98477, number(answers, "refund", "noul"), TOLERANCE);
                var team = (Map<String, Object>) answers.get("team");
                assertEquals("billing", team.get("choice"));
                assertEquals(
                        0.95346,
                        ((Number) ((Map<String, Object>) team.get("probabilities")).get("billing"))
                                .doubleValue(),
                        TOLERANCE);
                // the expected level of llama.cpp's probabilities: .024 + 2 * .766 + 3 * .192
                assertEquals(2.13, number(answers, "urgency", "score"), 0.05);

                // the usage: what the ticket costs each question alone (the default) and the three
                // asked jointly, the same counts the LangChain4j and Spring AI adapters report
                String ticket =
                        """
                        {"state": "Customer: I was charged twice for my order last week and I want my money back today. This is the third time I am writing.",
                         %s"questions": {
                           "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
                           "team": {"type": "choice", "instructions": "Which team should handle this?",
                                    "criteria": {"billing": "Charges and refunds", "shipping": "Delivery problems",
                                                 "technical": "Product does not work", "other": null}},
                           "urgency": {"type": "score", "instructions": "How urgent is this?",
                                       "criteria": ["Can wait", "This week", "Today", "Right now"]}}}
                        """;
                assertEquals(
                        Map.of("input_tokens", (long) EACH_TOKENS, "output_tokens", 0L),
                        post(url, ticket.formatted("")).body().get("usage"));
                assertEquals(
                        Map.of("input_tokens", (long) PACKED_TOKENS, "output_tokens", 0L),
                        post(url, ticket.formatted("\"joint\": true, ")).body().get("usage"));

                // a picture and no state: the question follows it directly
                String cat =
                        "data:image/png;base64,"
                                + Base64.getEncoder()
                                        .encodeToString(
                                                Files.readAllBytes(
                                                        TestModels.fixture("lfm3/images/cat.png")));
                answers =
                        (Map<String, Object>)
                                post(
                                                url,
                                                "{\"state\": null, \"files\": [\""
                                                        + cat
                                                        + "\"], \"questions\": {\"cat\": {\"type\":"
                                                        + " \"noul\", \"instructions\": \"Is there"
                                                        + " a cat in the picture?\"}}}")
                                        .body()
                                        .get("answers");
                assertEquals(0.99076, number(answers, "cat", "noul"), TOLERANCE);

                // what does not fit is the client's to shorten, said in the client's words
                var tooLong =
                        post(
                                url,
                                "{\"state\": \""
                                        + "word ".repeat(3000)
                                        + "\", \"questions\": {\"q\": {\"type\": \"noul\","
                                        + " \"instructions\": \"Is this long?\"}}}");
                assertEquals(400, tooLong.status());
                String message =
                        (String) ((Map<String, Object>) tooLong.body().get("error")).get("message");
                assertTrue(message.contains("shorten the state"), message);
            }
        } finally {
            Arenas.close(arena);
        }
    }

    /** d1-Omni hears a clip sent as a data URI, with and without a state. */
    @Test
    @SuppressWarnings("unchecked")
    void omniHearsAClip() throws Exception {
        Arena arena = Arenas.newCrossThread();
        try {
            DecisionModel omni =
                    Models.loadDecisionModel(
                            TestModels.require(
                                    "hf.co/LiquidAI/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf"),
                            arena,
                            Map.of(
                                    "media",
                                    TestModels.require(
                                            "hf.co/LiquidAI/d1-Omni-2B-GGUF/mmproj-d1-Omni-2B-F16.gguf")));
            try (var server = DecisionServer.start(omni, 2048, "omni", ServerConfig.local(0))) {
                String url = "http://127.0.0.1:" + server.address().getPort() + "/v1/systemone";
                String wav =
                        "data:audio/wav;base64,"
                                + Base64.getEncoder()
                                        .encodeToString(
                                                Files.readAllBytes(
                                                        TestModels.fixture("parakeet/ls0.wav")));
                String questions =
                        "\"questions\": {\"speech\": {\"type\": \"noul\", \"instructions\":"
                                + " \"Is someone speaking in the recording?\"}}";
                var answers =
                        (Map<String, Object>)
                                post(
                                                url,
                                                "{\"state\": \"A recording sent by a customer.\","
                                                        + " \"files\": [\""
                                                        + wav
                                                        + "\"], "
                                                        + questions
                                                        + "}")
                                        .body()
                                        .get("answers");
                assertEquals(0.53227, number(answers, "speech", "noul"), TOLERANCE);
                answers =
                        (Map<String, Object>)
                                post(
                                                url,
                                                "{\"state\": null, \"files\": [\""
                                                        + wav
                                                        + "\"], "
                                                        + questions
                                                        + "}")
                                        .body()
                                        .get("answers");
                assertEquals(0.50960, number(answers, "speech", "noul"), TOLERANCE);
            }
        } finally {
            Arenas.close(arena);
        }
    }

    /** d1-Omni reads a picture and a clip in one request: both towers, one answer set. */
    @Test
    @SuppressWarnings("unchecked")
    void omniReadsAPictureAndAClipTogether() throws Exception {
        Arena arena = Arenas.newCrossThread();
        try {
            DecisionModel omni =
                    Models.loadDecisionModel(
                            TestModels.require(
                                    "hf.co/LiquidAI/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf"),
                            arena,
                            Map.of(
                                    "media",
                                    TestModels.require(
                                            "hf.co/LiquidAI/d1-Omni-2B-GGUF/"
                                                    + "mmproj-d1-Omni-2B-F16.gguf")));
            try (var server = DecisionServer.start(omni, 2048, "omni", ServerConfig.local(0))) {
                String url = "http://127.0.0.1:" + server.address().getPort() + "/v1/systemone";
                String cat =
                        "data:image/png;base64,"
                                + Base64.getEncoder()
                                        .encodeToString(
                                                Files.readAllBytes(
                                                        TestModels.fixture(
                                                                "lfm3/images/" + "cat.png")));
                String wav =
                        "data:audio/wav;base64,"
                                + Base64.getEncoder()
                                        .encodeToString(
                                                Files.readAllBytes(
                                                        TestModels.fixture("parakeet/ls0.wav")));
                var answers =
                        (Map<String, Object>)
                                post(
                                                url,
                                                "{\"state\": \"A photo and a recording.\","
                                                        + " \"files\": [\""
                                                        + cat
                                                        + "\", \""
                                                        + wav
                                                        + "\"], \"questions\": {"
                                                        + "\"cat\": {\"type\": \"noul\","
                                                        + " \"instructions\": \"Is there a cat in"
                                                        + " the picture?\"},"
                                                        + "\"shows\": {\"type\": \"choice\","
                                                        + " \"instructions\": \"What does the"
                                                        + " picture show?\", \"criteria\":"
                                                        + " {\"cat\": null, \"dog\": null,"
                                                        + " \"dice\": null, \"landscape\":"
                                                        + " null}},"
                                                        + "\"speech\": {\"type\": \"noul\","
                                                        + " \"instructions\": \"Is someone speaking"
                                                        + " in the recording?\"}}}")
                                        .body()
                                        .get("answers");
                assertEquals(0.90702, number(answers, "cat", "noul"), TOLERANCE);
                assertEquals("cat", ((Map<String, Object>) answers.get("shows")).get("choice"));
                assertEquals(0.36273, number(answers, "speech", "noul"), TOLERANCE);
            }
        } finally {
            Arenas.close(arena);
        }
    }

    @SuppressWarnings("unchecked")
    private static double number(Map<String, Object> answers, String name, String field) {
        return ((Number) ((Map<String, Object>) answers.get(name)).get(field)).doubleValue();
    }

    private record Reply(int status, Map<String, Object> body) {}

    @SuppressWarnings("unchecked")
    private Reply post(String url, String json) throws Exception {
        HttpResponse<String> response =
                client.send(
                        HttpRequest.newBuilder(URI.create(url))
                                .POST(HttpRequest.BodyPublishers.ofString(json))
                                .header("Content-Type", "application/json")
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        return new Reply(
                response.statusCode(), (Map<String, Object>) JsonCodec.parse(response.body()));
    }
}
