package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.DecisionModel.Choice;
import com.qxotic.jinfer.chat.DecisionModel.Option;
import com.qxotic.jinfer.chat.DecisionModel.Predicate;
import com.qxotic.jinfer.chat.DecisionModel.Question;
import com.qxotic.jinfer.chat.DecisionModel.Rating;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * {@code POST /v1/systemone} over a stub model: the request it parses into an input and questions,
 * the System One answers it writes back, and the 400s a client can act on. No weights.
 */
class DecisionServerTest {

    /** Records what it is asked, answers with {@link #rows}; reads images, so video too. */
    static class Stub implements DecisionModel, Multimodal {
        List<Content> input;
        List<Question> questions;
        boolean together;
        Function<Question, float[]> rows = q -> new float[q.options().size()];
        RuntimeException failure;

        @Override
        public ContextState newState(int contextCapacity) {
            return new ContextState(64, 64, MemoryAllocators.ofArena(Arena.ofAuto()), false) {
                @Override
                protected void clearHistory() {}
            };
        }

        @Override
        public float[] decide(ContextState state, List<Content> input, Question question) {
            return decideEach(state, input, List.of(question))[0];
        }

        @Override
        public float[][] decideEach(
                ContextState state, List<Content> input, List<? extends Question> questions) {
            if (failure != null) throw failure;
            this.input = input;
            this.questions = new ArrayList<>(questions);
            return questions.stream().map(rows).toArray(float[][]::new);
        }

        @Override
        public float[][] decideTogether(
                ContextState state, List<Content> input, List<? extends Question> questions) {
            together = true;
            return decideEach(state, input, questions);
        }

        @Override
        @SuppressWarnings("unchecked")
        public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
            if (modality != Media.Image.class) return Optional.empty();
            return Optional.of(
                    (MediaProjector<R>)
                            new MediaProjector<Media.Image>() {
                                @Override
                                public int positions(Media.Image media) {
                                    return 1;
                                }

                                @Override
                                public void project(
                                        Media.Image media,
                                        int maxChunkSize,
                                        Consumer<MemoryView<?>> sink) {}
                            });
        }

        /** Like d1: video is read as its frames through the image tower. */
        @Override
        public boolean supports(Class<? extends Media> modality) {
            return modality == Media.Video.class || Multimodal.super.supports(modality);
        }
    }

    private final Stub stub = new Stub();
    private final HttpClient client = HttpClient.newHttpClient();
    private DecisionServer.Running server;
    private String base;

    @BeforeEach
    void start() throws Exception {
        server = DecisionServer.start(stub, 0, "d1", ServerConfig.local(0));
        base = "http://127.0.0.1:" + server.address().getPort();
    }

    @AfterEach
    void stop() {
        server.close();
    }

    private record Reply(int status, Map<String, Object> body) {
        @SuppressWarnings("unchecked")
        Map<String, Object> map(String... path) {
            Object at = body;
            for (String key : path) at = ((Map<String, Object>) at).get(key);
            return (Map<String, Object>) at;
        }

        String error() {
            return (String) map("error").get("message");
        }
    }

    @SuppressWarnings("unchecked")
    private Reply send(String method, String path, String json) throws Exception {
        HttpRequest.Builder request = HttpRequest.newBuilder(URI.create(base + path));
        request =
                json == null
                        ? request.GET()
                        : request.POST(HttpRequest.BodyPublishers.ofString(json))
                                .header("Content-Type", "application/json");
        HttpResponse<String> response =
                client.send(request.build(), HttpResponse.BodyHandlers.ofString());
        return new Reply(
                response.statusCode(), (Map<String, Object>) JsonCodec.parse(response.body()));
    }

    private Reply post(String json) throws Exception {
        return send("POST", "/v1/systemone", json);
    }

    private static final String THREE =
            """
            "questions": {
              "refund": {"type": "noul", "instructions": "A refund?",
                         "criteria": {"true": "asks for money back", "false": null}},
              "team": {"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": "Charges", "other": null}},
              "urgency": {"type": "score", "instructions": "How urgent?",
                          "criteria": ["low", "mid", "high"]}}
            """;

    @Test
    void questionsBecomeTheModelsAndAnswersTheSystemOneShape() throws Exception {
        stub.rows =
                q ->
                        switch (q) {
                            case Predicate p -> new float[] {0.25f, 0.75f};
                            case Choice c -> new float[] {0.375f, 0.625f};
                            case Rating r -> new float[] {0.25f, 0.375f, 0.375f};
                        };
        Reply reply = post("{\"state\": \"ticket\", " + THREE + "}");
        assertEquals(200, reply.status(), String.valueOf(reply.body()));

        assertEquals(List.of(new Content.Text("ticket")), stub.input);
        assertEquals(
                new Predicate("A refund?", "asks for money back", null), stub.questions.get(0));
        assertEquals(
                new Choice(
                        "Which team?",
                        List.of(new Option("billing", "Charges"), new Option("other"))),
                stub.questions.get(1));
        assertEquals(
                new Rating("How urgent?", Option.of("low", "mid", "high")), stub.questions.get(2));
        assertFalse(stub.together);

        assertEquals("d1", reply.body().get("model"));
        assertEquals(Map.of("type", "noul", "noul", 0.75), reply.map("answers", "refund"));
        Map<String, Object> team = reply.map("answers", "team");
        assertEquals("other", team.get("choice"));
        assertEquals(Map.of("billing", 0.375, "other", 0.625), team.get("probabilities"));
        assertEquals(0.625, team.get("confidence"));
        Map<String, Object> urgency = reply.map("answers", "urgency");
        // the expected level: 0.375 * 1 + 0.375 * 2
        assertEquals(1.125, ((Number) urgency.get("score")).doubleValue(), 1e-6);
        assertEquals(Map.of("0", "low", "1", "mid", "2", "high"), urgency.get("legend"));
        assertEquals(
                List.of("refund", "team", "urgency"), List.copyOf(reply.map("answers").keySet()));
    }

    @Test
    void togetherIsAskedFor() throws Exception {
        assertEquals(200, post("{\"state\": \"x\", \"together\": true, " + THREE + "}").status());
        assertTrue(stub.together);
    }

    @Test
    void aStructuredStateIsReadAsIndentedJson() throws Exception {
        assertEquals(
                200,
                post("{\"state\": {\"customer\": \"Ana\", \"order\": 4411}, " + THREE + "}")
                        .status());
        assertEquals(
                List.of(new Content.Text("{\n  \"customer\": \"Ana\",\n  \"order\": 4411\n}")),
                stub.input);
    }

    @Test
    void filesComeAheadOfTheStateAndANullStateIsTheFilesAlone() throws Exception {
        String png = png();
        assertEquals(
                200,
                post("{\"state\": \"a photo\", \"files\": [\"" + png + "\"], " + THREE + "}")
                        .status());
        assertEquals(2, stub.input.size());
        assertInstanceOf(
                Media.Image.class,
                assertInstanceOf(Content.Media.class, stub.input.get(0)).media());
        assertEquals(new Content.Text("a photo"), stub.input.get(1));

        assertEquals(
                200,
                post("{\"state\": null, \"images\": [\"" + png + "\"], " + THREE + "}").status());
        assertEquals(1, stub.input.size());
    }

    @Test
    void whatTheClientGotWrongIsA400ThatSaysSo() throws Exception {
        String one = "\"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"?\"}}";
        assertRefused("{" + one + "}", "state must be provided");
        assertRefused("{\"state\": null, " + one + "}", "nothing to judge");
        assertRefused("{\"state\": \"x\"}", "questions must be a non-empty object");
        assertRefused(
                "{\"state\": \"x\", \"questions\": {}}", "questions must be a non-empty object");
        assertRefused(
                "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"maybe\", \"instructions\":"
                        + " \"?\"}}}",
                "questions.q.type must be noul, choice or score");
        assertRefused(
                "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\":"
                        + " \"\"}}}",
                "questions.q.instructions must be a non-empty text");
        assertRefused(
                "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\":"
                        + " \"?\", \"criteria\": {\"a\": null}}}}",
                "at least 2 options");
        assertRefused(
                "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"score\", \"instructions\":"
                        + " \"?\", \"criteria\": [\"a\"]}}}",
                "2 to 10 levels");
        assertRefused(
                "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"score\", \"instructions\":"
                        + " \"?\", \"criteria\": [\"a\", 2]}}}",
                "levels must be texts");
        assertRefused(
                "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\":"
                        + " \"?\", \"criteria\": {\"a\": 1, \"b\": null}}}}",
                "questions.q.criteria.a must be a text or null");
        assertRefused(
                "{\"state\": \"x\", \"files\": [\"https://example.com/a.png\"], " + one + "}",
                "does not fetch remote URLs");
        assertRefused(
                "{\"state\": \"x\", \"files\": [\"data:text/plain;base64,aGk=\"], " + one + "}",
                "not data:text/plain");
        assertEquals(null, stub.input, "nothing reached the model");
    }

    @Test
    void anInputTooLargeNamesTheClientsRemedyNotTheLibrarys() throws Exception {
        stub.failure =
                new IllegalArgumentException(
                        "the input and its questions take 9000 tokens, the state holds 4096: open a"
                                + " larger one with newState");
        String message = post("{\"state\": \"x\", " + THREE + "}").error();
        assertEquals(
                "the input and its questions take 9000 tokens, the state holds 4096: shorten the"
                        + " state or send smaller files",
                message);
    }

    @Test
    void theProbesDescribeADecisionServer() throws Exception {
        Reply props = send("GET", "/props", null);
        assertEquals("decision", props.body().get("task"));
        assertEquals(64, ((Number) props.body().get("context_capacity")).intValue());
        @SuppressWarnings("unchecked")
        Map<String, Object> card =
                (Map<String, Object>)
                        ((List<?>) send("GET", "/v1/models", null).body().get("data")).get(0);
        assertEquals(
                Map.of("input_modalities", List.of("text", "image", "video")),
                card.get("architecture"));
        Reply wrongModel = post("{\"model\": \"other\", \"state\": \"x\", " + THREE + "}");
        assertTrue(wrongModel.error().contains("Unknown model"), wrongModel.error());
    }

    /** A family that reads images but not video advertises exactly that, as vLLM's d3 does. */
    @Test
    void aModelThatReadsNoVideoDoesNotAdvertiseIt() throws Exception {
        DecisionModel stills =
                new Stub() {
                    @Override
                    public boolean supports(Class<? extends Media> modality) {
                        return modality != Media.Video.class && super.supports(modality);
                    }
                };
        try (DecisionServer.Running other =
                DecisionServer.start(stills, 0, "d3", ServerConfig.local(0))) {
            HttpResponse<String> response =
                    client.send(
                            HttpRequest.newBuilder(
                                            URI.create(
                                                    "http://127.0.0.1:"
                                                            + other.address().getPort()
                                                            + "/v1/models"))
                                    .GET()
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());
            @SuppressWarnings("unchecked")
            Map<String, Object> card =
                    (Map<String, Object>)
                            ((List<?>)
                                            ((Map<String, Object>) JsonCodec.parse(response.body()))
                                                    .get("data"))
                                    .get(0);
            assertEquals(
                    Map.of("input_modalities", List.of("text", "image")), card.get("architecture"));
        }
    }

    /** The transport's own rules: POST only, a body, and the names a JSON object can hold once. */
    @Test
    void theTransportAnswersTheProtocolEdges() throws Exception {
        // a GET on the work endpoint is a 405 that names what the route takes
        HttpResponse<String> get =
                client.send(
                        HttpRequest.newBuilder(URI.create(base + "/v1/systemone")).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
        assertEquals(405, get.statusCode(), get.body());
        assertEquals("POST, OPTIONS", get.headers().firstValue("Allow").orElse(""));

        // a duplicate key would answer one of the two questions in silence: refused, named
        Reply duplicate =
                post(
                        "{\"state\": \"x\", \"questions\": {"
                                + "\"q\": {\"type\": \"noul\", \"instructions\": \"first?\"},"
                                + "\"q\": {\"type\": \"noul\", \"instructions\": \"second?\"}}}");
        assertEquals(400, duplicate.status(), String.valueOf(duplicate.body()));
        assertTrue(duplicate.error().contains("Duplicate key: 'q'"), duplicate.error());

        // unknown members are ignored, not refused: a client may send its own metadata
        Reply unknown =
                post(
                        "{\"state\": \"x\", \"trace_id\": \"abc\", \"questions\": {"
                                + "\"q\": {\"type\": \"noul\", \"instructions\": \"?\","
                                + " \"extra\": 1}}}");
        assertEquals(200, unknown.status(), String.valueOf(unknown.body()));

        // a noul's criteria name each side; both are passed to the model
        assertEquals(
                200,
                post("{\"state\": \"x\", \"questions\": {"
                                + "\"q\": {\"type\": \"noul\", \"instructions\": \"A"
                                + " refund?\", \"criteria\": {\"true\": \"asks for money"
                                + " back\", \"false\": \"anything else\"}}}}")
                        .status());
        assertEquals(
                new Predicate("A refund?", "asks for money back", "anything else"),
                stub.questions.getLast());

        // an empty body is a 400 that says so, never a hang
        HttpResponse<String> empty =
                client.send(
                        HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                                .POST(HttpRequest.BodyPublishers.ofString(""))
                                .header("Content-Type", "application/json")
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        assertEquals(400, empty.statusCode(), empty.body());
    }

    private void assertRefused(String json, String says) throws Exception {
        Reply reply = post(json);
        assertEquals(400, reply.status(), json);
        assertTrue(reply.error().contains(says), reply.error());
    }

    private static String png() throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB), "png", out);
        return "data:image/png;base64," + Base64.getEncoder().encodeToString(out.toByteArray());
    }
}
