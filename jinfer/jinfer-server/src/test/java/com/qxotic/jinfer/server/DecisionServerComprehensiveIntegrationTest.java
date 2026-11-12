package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code POST /v1/systemone} beyond the parity cases: a video data URI through ffmpeg and the
 * vision tower, {@code joint}, the score's distribution, the metrics counters and a malformed body.
 * Liquid d1-2B with its vision tower, so {@code integration}.
 */
@Tag("integration")
@SuppressWarnings("unchecked")
final class DecisionServerComprehensiveIntegrationTest {

    static final String D1 = "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf";
    static final String MMPROJ = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
    static final double TOLERANCE = 2e-2;

    static final String THREE =
            """
            "questions": {
              "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
              "team": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Charges and refunds", "shipping": "Delivery problems",
                                    "technical": "Product does not work", "other": null}},
              "urgency": {"type": "score", "instructions": "How urgent is this?",
                          "criteria": ["Can wait", "This week", "Today", "Right now"]}}
            """;

    static final String TICKET =
            "Customer: I was charged twice for my order last week and I want my money back today."
                    + " This is the third time I am writing.";

    static final Arena ARENA = Arenas.newCrossThread();
    static DecisionModel d1;
    static DecisionServer.Running server;
    static String base;

    private final HttpClient client = HttpClient.newHttpClient();

    @BeforeAll
    static void start() throws Exception {
        d1 =
                Models.loadDecisionModel(
                        TestModels.require(D1), ARENA, Map.of("media", TestModels.require(MMPROJ)));
        server = DecisionServer.start(d1, 4096, "d1", ServerConfig.local(0));
        base = "http://127.0.0.1:" + server.address().getPort();
    }

    @AfterAll
    static void stop() {
        if (server != null) server.close();
        ARENA.close();
    }

    @Test
    void videoIsJudgedOverHttp() throws Exception {
        assumeTrue(VideoFixture.available(), "ffmpeg is not available");
        Path clip = VideoFixture.clip();
        String data =
                "data:video/mp4;base64,"
                        + Base64.getEncoder().encodeToString(Files.readAllBytes(clip));
        Reply reply =
                post(
                        "{\"state\": \"A clip sent by a customer.\", \"files\": [\""
                                + data
                                + "\"], \"questions\": {\"both\": {\"type\": \"noul\","
                                + " \"instructions\": \"Does the video show both a cat and"
                                + " dice?\"},\"first\": {\"type\": \"choice\", \"instructions\":"
                                + " \"What does the video show first?\", \"criteria\": {\"cat\":"
                                + " null, \"dog\": null, \"dice\": null, \"landscape\":"
                                + " null}},\"last\": {\"type\": \"choice\", \"instructions\":"
                                + " \"What does the video show at the end?\", \"criteria\":"
                                + " {\"cat\": null, \"dog\": null, \"dice\": null, \"landscape\":"
                                + " null}}}}");
        assertEquals(200, reply.status(), String.valueOf(reply.body()));
        Map<String, Object> answers = answers(reply);
        assertEquals(0.98811, number(answers, "both", "noul"), TOLERANCE);
        Map<String, Object> first = (Map<String, Object>) answers.get("first");
        assertEquals("cat", first.get("choice"));
        assertEquals(
                0.97993,
                ((Number) ((Map<String, Object>) first.get("probabilities")).get("cat"))
                        .doubleValue(),
                TOLERANCE);
        Map<String, Object> last = (Map<String, Object>) answers.get("last");
        assertEquals("dice", last.get("choice"));
        assertEquals(
                0.98170,
                ((Number) ((Map<String, Object>) last.get("probabilities")).get("dice"))
                        .doubleValue(),
                TOLERANCE);
        Files.deleteIfExists(clip);
    }

    @Test
    void togetherIsTheModelsPackedReading() throws Exception {
        String alone = "{\"state\": \"" + TICKET + "\", " + THREE + "}";
        String packed = "{\"state\": \"" + TICKET + "\", \"joint\": true, " + THREE + "}";
        Map<String, Object> each = answers(post(alone));
        Map<String, Object> all = answers(post(packed));
        assertEquals("billing", ((Map<String, Object>) each.get("team")).get("choice"));
        assertEquals("billing", ((Map<String, Object>) all.get("team")).get("choice"));
        assertTrue(number(each, "refund", "noul") > 0.9);
        assertTrue(number(all, "refund", "noul") > 0.9);
        // packed, the refund answer shifts with its neighbours: the documented trade-off
        assertTrue(
                Math.abs(number(each, "refund", "noul") - number(all, "refund", "noul")) > 1e-4,
                "each " + each.get("refund") + ", together " + all.get("refund"));
    }

    @Test
    void aScoreExposesItsDistributionAndExpectedLevel() throws Exception {
        String questions =
                "\"questions\": {\"energy\": {\"type\": \"score\", \"instructions\": \"How much"
                    + " energy does it take?\", \"criteria\": [\"None\", \"A little\", \"Some\","
                    + " \"A lot\", \"Everything\"]}}";
        Reply reply = post("{\"state\": \"" + TICKET + "\", " + questions + "}");
        assertEquals(200, reply.status(), String.valueOf(reply.body()));
        Map<String, Object> answer = (Map<String, Object>) answers(reply).get("energy");
        Map<String, Object> probabilities = (Map<String, Object>) answer.get("probabilities");
        Map<String, Object> legend = (Map<String, Object>) answer.get("legend");
        assertEquals(5, probabilities.size());
        assertEquals(List.of("0", "1", "2", "3", "4"), List.copyOf(legend.keySet()));
        double sum = 0, mean = 0;
        for (int i = 0; i < 5; i++) {
            double p = ((Number) probabilities.get(Integer.toString(i))).doubleValue();
            mean += i * p;
            sum += p;
        }
        assertEquals(1, sum, 1e-5);
        assertEquals(Math.min(mean, 4), ((Number) answer.get("score")).doubleValue(), 1e-9);
        double confidence = ((Number) answer.get("confidence")).doubleValue();
        assertTrue(confidence >= 0 && confidence <= 1, "confidence " + confidence);
    }

    @Test
    void metricsCountRequestsAndQuestions() throws Exception {
        String before = get("/metrics");
        long requests = metric(before, "jinfer_decision_requests_completed_total");
        long answered = metric(before, "jinfer_decision_questions_answered_total");
        assertEquals(
                200,
                post("{\"state\": \""
                                + TICKET
                                + "\", \"questions\": {"
                                + "\"refund\": {\"type\": \"noul\", \"instructions\": \"A"
                                + " refund?\"},"
                                + "\"urgent\": {\"type\": \"noul\", \"instructions\":"
                                + " \"Is it urgent?\"}}}")
                        .status());
        String after = get("/metrics");
        assertEquals(requests + 1, metric(after, "jinfer_decision_requests_completed_total"));
        assertEquals(answered + 2, metric(after, "jinfer_decision_questions_answered_total"));
    }

    @Test
    void aMalformedBodyIsA400() throws Exception {
        Reply reply = post("this is not json");
        assertEquals(400, reply.status());
        assertNotNull(reply.error());
        assertTrue(!reply.error().isBlank(), reply.error());
    }

    /** Admission and parsing are concurrent; compute is one fair lock over one state. */
    @Test
    void concurrentRequestsAreAllAnswered() throws Exception {
        String body =
                "{\"state\": \""
                        + TICKET
                        + "\", \"questions\": {"
                        + "\"refund\": {\"type\": \"noul\", \"instructions\": \"A refund?\"}}}";
        ExecutorService pool = Executors.newFixedThreadPool(4);
        try {
            List<Future<Reply>> futures = new ArrayList<>();
            for (int i = 0; i < 4; i++) futures.add(pool.submit(() -> post(body)));
            for (Future<Reply> future : futures) {
                Reply reply = future.get(120, TimeUnit.SECONDS);
                assertEquals(200, reply.status(), String.valueOf(reply.body()));
                assertTrue(
                        number(answers(reply), "refund", "noul") > 0.9,
                        String.valueOf(reply.body()));
            }
        } finally {
            pool.shutdownNow();
        }
    }

    /** Seventeen questions in one request: past the sixteen a packed prompt holds. */
    @Test
    void seventeenQuestionsInOneRequestArePackedInTwoPrompts() throws Exception {
        StringBuilder questions = new StringBuilder("\"questions\": {");
        for (int i = 0; i < 17; i++)
            questions
                    .append(i == 0 ? "" : ", ")
                    .append("\"q")
                    .append(i)
                    .append(
                            "\": {\"type\": \"noul\", \"instructions\": \"Is the customer"
                                    + " asking for a refund?\"}");
        questions.append('}');
        Reply reply = post("{\"state\": \"" + TICKET + "\", \"joint\": true, " + questions + "}");
        assertEquals(200, reply.status(), String.valueOf(reply.body()));
        Map<String, Object> answers = answers(reply);
        assertEquals(17, answers.size());
        for (int i = 0; i < 17; i++)
            assertEquals(
                    1.0,
                    ((Number) ((Map<String, Object>) answers.get("q" + i)).get("noul"))
                                            .doubleValue()
                                    > 0.5
                            ? 1.0
                            : 0.0,
                    0,
                    "q" + i);
    }

    /** Unicode, emoji and control characters travel as JSON text and are judged. */
    @Test
    void unicodeAndEmojiAreJudged() throws Exception {
        String state =
                "Caf\u00e9 \u2615 na\u00efve r\u00e9sum\u00e9 \ud83c\udf89 \u65e5\u672c"
                        + "\u8a9e: I was charged twice and I want my money back today.";
        Reply reply =
                post(
                        "{\"state\": \""
                                + state
                                + "\", \"questions\": {"
                                + "\"refund\": {\"type\": \"noul\", \"instructions\":"
                                + " \"Is the customer asking for a refund?\"}}}");
        assertEquals(200, reply.status(), String.valueOf(reply.body()));
        assertTrue(number(answers(reply), "refund", "noul") > 0.9);
    }

    /** A structured state of every JSON shape is put to the model as JSON, a member a line. */
    @Test
    void aStructuredStateOfEveryShapeIsRead() throws Exception {
        String state =
                "{\"text\": \""
                        + TICKET.replace("\"", "'")
                        + "\", \"n\": 3, \"ratio\": 0.25,"
                        + " \"flag\": true, \"none\": null, \"tags\": [\"a\", \"b\"],"
                        + " \"nested\": {\"k\": \"v\"}}";
        Reply reply =
                post(
                        "{\"state\": "
                                + state
                                + ", \"questions\": {"
                                + "\"refund\": {\"type\": \"noul\", \"instructions\":"
                                + " \"Is the customer asking for a refund?\"}}}");
        assertEquals(200, reply.status(), String.valueOf(reply.body()));
        assertTrue(number(answers(reply), "refund", "noul") > 0.5);
    }

    /** Ten levels is the score's ceiling; sixty options the choice's fallback codes. */
    @Test
    void tenLevelsAndSixtyOptionsAnswer() throws Exception {
        StringBuilder levels = new StringBuilder();
        for (int i = 0; i < 10; i++)
            levels.append(i == 0 ? "" : ", ").append('"').append("L").append(i).append('"');
        StringBuilder options = new StringBuilder();
        for (int i = 0; i < 60; i++)
            options.append(i == 0 ? "" : ", ").append('"').append("o").append(i).append("\": null");
        Reply reply =
                post(
                        "{\"state\": \""
                                + TICKET
                                + "\", \"questions\": {"
                                + "\"n\": {\"type\": \"score\", \"instructions\": \"How"
                                + " much?\", \"criteria\": ["
                                + levels
                                + "]},"
                                + "\"c\": {\"type\": \"choice\", \"instructions\": \"Which"
                                + " one?\", \"criteria\": {"
                                + options
                                + "}}}}");
        assertEquals(200, reply.status(), String.valueOf(reply.body()));
        Map<String, Object> answers = answers(reply);
        Map<String, Object> score = (Map<String, Object>) answers.get("n");
        assertEquals(10, ((Map<String, Object>) score.get("probabilities")).size());
        assertEquals(10, ((Map<String, Object>) score.get("legend")).size());
        Map<String, Object> choice = (Map<String, Object>) answers.get("c");
        assertEquals(60, ((Map<String, Object>) choice.get("probabilities")).size());
    }

    /** Files come before the state; a state with files but no state object is refused. */
    @Test
    void theStateAndTheFilesKeepTheirOrder() throws Exception {
        String cat =
                "data:image/png;base64,"
                        + Base64.getEncoder()
                                .encodeToString(
                                        Files.readAllBytes(
                                                Path.of(
                                                        "../../test-fixtures/lfm3/images/"
                                                                + "cat.png")));
        Reply withText =
                post(
                        "{\"state\": \"A photo sent by a customer.\", \"files\": [\""
                                + cat
                                + "\"], \"questions\": {\"cat\": {\"type\": \"noul\","
                                + " \"instructions\": \"Is there a cat?\"}}}");
        assertEquals(200, withText.status(), String.valueOf(withText.body()));
        assertTrue(number(answers(withText), "cat", "noul") > 0.9);

        // media alone: state null, the question follows the picture directly
        Reply alone =
                post(
                        "{\"state\": null, \"files\": [\""
                                + cat
                                + "\"], \"questions\": {\"cat\": {\"type\": \"noul\","
                                + " \"instructions\": \"Is there a cat?\"}}}");
        assertEquals(200, alone.status(), String.valueOf(alone.body()));
        assertTrue(number(answers(alone), "cat", "noul") > 0.9);

        // null with no files judges nothing
        Reply nothing =
                post(
                        "{\"state\": null, \"questions\": {\"q\": {\"type\": \"noul\","
                                + " \"instructions\": \"?\"}}}");
        assertEquals(400, nothing.status(), String.valueOf(nothing.body()));
        assertTrue(nothing.error().contains("nothing to judge"), nothing.error());

        // an empty state object is the JSON "{}", a text
        Reply empty =
                post(
                        "{\"state\": {}, \"questions\": {\"q\": {\"type\": \"noul\","
                                + " \"instructions\": \"?\"}}}");
        assertEquals(200, empty.status(), String.valueOf(empty.body()));
    }

    /** together on d3: the family packs nothing, and answers each question as it would alone. */
    @Test
    void togetherOnAFamilyWithoutPackingIsEach() throws Exception {
        String body =
                "\"state\": \"I was charged twice and I want my money back today.\","
                        + " \"questions\": {\"refund\": {\"type\": \"noul\","
                        + " \"instructions\": \"Is the customer asking for a refund?\"}}";
        DecisionServer.Running d3Server = null;
        try {
            d3Server =
                    DecisionServer.start(
                            Models.loadDecisionModel(
                                    TestModels.require(
                                            "hf.co/ggml-org/vllm-d3-edge-GGUF/"
                                                    + "vllm-d3-edge-Q8_0.gguf"),
                                    ARENA),
                            2048,
                            "d3",
                            ServerConfig.local(0));
            String url = "http://127.0.0.1:" + d3Server.address().getPort() + "/v1/systemone";
            Reply each = postTo(url, "{" + body + "}");
            Reply packed = postTo(url, "{" + body + ", \"joint\": true}");
            assertEquals(200, each.status(), String.valueOf(each.body()));
            assertEquals(200, packed.status(), String.valueOf(packed.body()));
            assertEquals(
                    number(answers(each), "refund", "noul"),
                    number(answers(packed), "refund", "noul"),
                    1e-5);
        } finally {
            if (d3Server != null) d3Server.close();
        }
    }

    /** Every JSON shape a state can be is read: array, nested object, deep array, scalar string. */
    @Test
    void everyStructuredStateShapeIsRead() throws Exception {
        for (String state :
                List.of(
                        "[\"a\", \"b\"]",
                        "{\"outer\": {\"inner\": [1, 2, {\"deep\": null}]}}",
                        "[[1, 2], [3, 4]]",
                        "{\"message\": \"" + TICKET.replace("\"", "'") + "\"}")) {
            Reply reply =
                    post(
                            "{\"state\": "
                                    + state
                                    + ", \"questions\": {\"refund\": {\"type\": \"noul\","
                                    + " \"instructions\": \"Is the customer asking for a"
                                    + " refund?\"}}}");
            assertEquals(200, reply.status(), state + ": " + reply.body());
            assertTrue(number(answers(reply), "refund", "noul") >= 0);
        }
        // the ticket in a field still reads as a refund
        Reply ticket =
                post(
                        "{\"state\": {\"message\": \""
                                + TICKET.replace("\"", "'")
                                + "\"}, \"questions\": {\"refund\": {\"type\": \"noul\","
                                + " \"instructions\": \"Is the customer asking for a"
                                + " refund?\"}}}");
        assertTrue(number(answers(ticket), "refund", "noul") > 0.5);
    }

    /** files and images are the same list under two names; both accept a picture, in order. */
    @Test
    void filesAndImagesAreTheSameList() throws Exception {
        String cat =
                "data:image/png;base64,"
                        + Base64.getEncoder()
                                .encodeToString(
                                        Files.readAllBytes(
                                                Path.of(
                                                        "../../test-fixtures/lfm3/images/"
                                                                + "cat.png")));
        String shows =
                "\"questions\": {\"shows\": {\"type\": \"choice\", \"instructions\":"
                        + " \"What does the picture show?\", \"criteria\": {\"cat\": null,"
                        + " \"dog\": null, \"dice\": null, \"landscape\": null}}}";
        for (String field : List.of("files", "images")) {
            Reply reply =
                    post("{\"state\": null, \"" + field + "\": [\"" + cat + "\"], " + shows + "}");
            assertEquals(200, reply.status(), field + ": " + reply.body());
            assertEquals(
                    "cat",
                    ((Map<String, Object>) answers(reply).get("shows")).get("choice"),
                    field);
        }
    }

    @Test
    void theProbesDescribeTheServedModel() throws Exception {
        Map<String, Object> props = parse(get("/props"));
        assertEquals("decision", props.get("task"));
        assertEquals(4096, ((Number) props.get("context_capacity")).intValue());
        Map<String, Object> card =
                (Map<String, Object>) ((List<?>) parse(get("/v1/models")).get("data")).get(0);
        assertEquals(
                Map.of("input_modalities", List.of("text", "image", "video")),
                card.get("architecture"));
    }

    private static long metric(String exposition, String name) {
        for (String line : exposition.split("\n"))
            if (line.startsWith(name + " "))
                return Long.parseLong(line.substring(name.length() + 1).trim());
        throw new AssertionError("no " + name + " in:\n" + exposition);
    }

    private record Reply(int status, Map<String, Object> body) {
        String error() {
            Object error = body.get("error");
            return error instanceof Map<?, ?> map ? (String) map.get("message") : null;
        }
    }

    private Reply postTo(String url, String json) throws Exception {
        HttpResponse<String> response =
                client.send(
                        HttpRequest.newBuilder(URI.create(url))
                                .POST(HttpRequest.BodyPublishers.ofString(json))
                                .header("Content-Type", "application/json")
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        return new Reply(response.statusCode(), parse(response.body()));
    }

    private Reply post(String json) throws Exception {
        HttpResponse<String> response =
                client.send(
                        HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                                .POST(HttpRequest.BodyPublishers.ofString(json))
                                .header("Content-Type", "application/json")
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        return new Reply(response.statusCode(), parse(response.body()));
    }

    private String get(String path) throws Exception {
        HttpResponse<String> response =
                client.send(
                        HttpRequest.newBuilder(URI.create(base + path)).GET().build(),
                        HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), response.body());
        return response.body();
    }

    private static Map<String, Object> parse(String json) {
        return (Map<String, Object>) JsonCodec.parse(json);
    }

    private static Map<String, Object> answers(Reply reply) {
        assertEquals(200, reply.status(), String.valueOf(reply.body()));
        return (Map<String, Object>) reply.body().get("answers");
    }

    private static double number(Map<String, Object> answers, String name, String field) {
        return ((Number) ((Map<String, Object>) answers.get(name)).get(field)).doubleValue();
    }
}
