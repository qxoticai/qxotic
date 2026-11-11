package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

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
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code POST /v1/systemone} over vLLM Semantic Router's d3 (Qwen3.5), the second decision family:
 * its own prompt and readout, images but no video, and the same System One answers. The numbers are
 * llama.cpp's on the same file (2026-11-10).
 */
@Tag("integration")
@SuppressWarnings("unchecked")
final class DecisionServerD3IntegrationTest {

    private static final String D3 = "hf.co/ggml-org/vllm-d3-edge-GGUF/vllm-d3-edge-Q8_0.gguf";
    private static final String MMPROJ =
            "hf.co/ggml-org/vllm-d3-edge-GGUF/mmproj-vllm-d3-edge-BF16.gguf";
    private static final double TOLERANCE = 2e-2;

    private static final String TICKET =
            "Customer: I was charged twice for my order last week and I want my money back today."
                    + " This is the third time I am writing.";

    private final HttpClient client = HttpClient.newHttpClient();

    @Test
    void aSecondFamilyAnswersOverHttp() throws Exception {
        Arena arena = Arenas.newCrossThread();
        try {
            DecisionModel d3 =
                    Models.loadDecisionModel(
                            TestModels.require(D3),
                            arena,
                            Map.of("media", TestModels.require(MMPROJ)));
            try (var server = DecisionServer.start(d3, 4096, "d3", ServerConfig.local(0))) {
                String url = "http://127.0.0.1:" + server.address().getPort();
                Reply reply =
                        post(
                                url + "/v1/systemone",
                                "{\"state\": \""
                                        + TICKET
                                        + "\", \"questions\": {"
                                        + "\"refund\": {\"type\": \"noul\", \"instructions\":"
                                        + " \"Is the customer asking for a refund?\"},"
                                        + "\"team\": {\"type\": \"choice\", \"instructions\":"
                                        + " \"Which team should handle this?\", \"criteria\":"
                                        + " {\"billing\": \"Charges and refunds\", \"shipping\":"
                                        + " \"Delivery problems\", \"technical\": \"Product does"
                                        + " not work\", \"other\": null}},"
                                        + "\"urgency\": {\"type\": \"score\", \"instructions\":"
                                        + " \"How urgent is this?\", \"criteria\": [\"Can wait\","
                                        + " \"This week\", \"Today\", \"Right now\"]}}}");
                assertEquals(200, reply.status(), String.valueOf(reply.body()));
                Map<String, Object> answers = (Map<String, Object>) reply.body().get("answers");
                assertEquals(0.97326, number(answers, "refund", "noul"), TOLERANCE);
                Map<String, Object> team = (Map<String, Object>) answers.get("team");
                assertEquals("billing", team.get("choice"));
                assertEquals(
                        0.97421,
                        ((Number) ((Map<String, Object>) team.get("probabilities")).get("billing"))
                                .doubleValue(),
                        TOLERANCE);
                Map<String, Object> urgency = (Map<String, Object>) answers.get("urgency");
                // the expected level, as every face reports it: d3 reads this ticket as the calmest
                assertEquals(1.127, ((Number) urgency.get("score")).doubleValue(), 0.05);
                assertEquals(
                        Map.of("0", "Can wait", "1", "This week", "2", "Today", "3", "Right now"),
                        urgency.get("legend"));
            }
        } finally {
            Arenas.close(arena);
        }
    }

    /**
     * d3 reads images but not video: the card says so, and a video is refused in the client's
     * words.
     */
    @Test
    void theCardAndTheRefusalMatchTheFamily() throws Exception {
        Arena arena = Arenas.newCrossThread();
        try {
            DecisionModel d3 =
                    Models.loadDecisionModel(
                            TestModels.require(D3),
                            arena,
                            Map.of("media", TestModels.require(MMPROJ)));
            try (var server = DecisionServer.start(d3, 4096, "d3", ServerConfig.local(0))) {
                String url = "http://127.0.0.1:" + server.address().getPort();
                Reply models = get(url + "/v1/models");
                Map<String, Object> card =
                        (Map<String, Object>) ((List<?>) models.body().get("data")).get(0);
                assertEquals(
                        Map.of("input_modalities", List.of("text", "image")),
                        card.get("architecture"));

                // a real video data URI: the model has no tower for one, and says so
                assumeTrue(VideoFixture.available(), "ffmpeg is not available");
                Path clip = VideoFixture.clip();
                try {
                    String video =
                            "data:video/mp4;base64,"
                                    + java.util.Base64.getEncoder()
                                            .encodeToString(Files.readAllBytes(clip));
                    Reply refused =
                            post(
                                    url + "/v1/systemone",
                                    "{\"state\": \"A clip.\", \"files\": [\""
                                            + video
                                            + "\"], \"questions\": {\"q\": {\"type\":"
                                            + " \"noul\", \"instructions\": \"Is there a"
                                            + " cat?\"}}}");
                    assertEquals(400, refused.status(), String.valueOf(refused.body()));
                    assertTrue(refused.error().contains("takes no video"), refused.error());
                } finally {
                    Files.deleteIfExists(clip);
                }
            }
        } finally {
            Arenas.close(arena);
        }
    }

    @SuppressWarnings("unchecked")
    private static double number(Map<String, Object> answers, String name, String field) {
        return ((Number) ((Map<String, Object>) answers.get(name)).get(field)).doubleValue();
    }

    private record Reply(int status, Map<String, Object> body) {
        String error() {
            Object error = body.get("error");
            return error instanceof Map<?, ?> map ? (String) map.get("message") : null;
        }
    }

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

    @SuppressWarnings("unchecked")
    private Reply get(String url) throws Exception {
        HttpResponse<String> response =
                client.send(
                        HttpRequest.newBuilder(URI.create(url)).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
        return new Reply(
                response.statusCode(), (Map<String, Object>) JsonCodec.parse(response.body()));
    }
}
