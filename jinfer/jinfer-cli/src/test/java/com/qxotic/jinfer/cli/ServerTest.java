package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.server.ServerConfig;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class ServerTest {
    @Test
    void sharedSettingsAndTransportSettingsMapToExistingConfiguration() {
        Options o =
                Options.parse(
                        "-m",
                        "unused",
                        "--threads",
                        "8",
                        "server",
                        "--port",
                        "0",
                        "--concurrency",
                        "7",
                        "--queue-depth",
                        "2",
                        "--max-body-mb",
                        "5",
                        "--request-timeout",
                        "0",
                        "--write-timeout",
                        "9",
                        "--cors-origin",
                        "https://a.example",
                        "--cors-origin",
                        "https://b.example",
                        "-n",
                        "32");
        var c = Server.config(o, null);
        assertEquals(8, o.threads);
        assertEquals(0, c.bind().getPort());
        assertEquals(7, c.limits().threads());
        assertEquals(2, c.limits().queueCapacity());
        assertEquals(5L << 20, c.limits().maxBodyBytes());
        assertEquals(Duration.ZERO, c.limits().requestTimeout());
        assertEquals(Duration.ofSeconds(9), c.limits().writeTimeout());
        assertEquals(32, c.defaults().maxOutputTokens());
        assertEquals(2, Options.parse("serve", "-m", "m", "--queue-depth", "2").server.queueDepth);
        assertThrows(Options.Usage.class, () -> Server.validateTranscription(o));
        Server.validateTranscription(Options.parse("server", "-m", "m", "--threads", "2"));
    }

    @Test
    void authenticationAndInvalidLimitsFailClearly() {
        Options insecure = Options.parse("server", "-m", "unused", "--host", "0.0.0.0");
        assertThrows(Options.Usage.class, () -> Server.config(insecure, null));
        assertDoesNotThrow(
                () ->
                        Server.config(
                                Options.parse(
                                        "server",
                                        "-m",
                                        "m",
                                        "--host",
                                        "0.0.0.0",
                                        "--api-key",
                                        "test"),
                                null));
        for (String[] tail :
                new String[][] {
                    {"--port", "65536"},
                    {"--concurrency", "0"},
                    {"--queue-depth", "-1"},
                    {"--max-body-mb", "0"},
                    {"--write-timeout", "0"}
                }) {
            String[] args = {"server", "-m", "unused", tail[0], tail[1]};
            assertThrows(Options.Usage.class, () -> Options.parse(args));
        }
    }

    @Test
    void theHttpApplicationRunsAgainstTheWeightlessModelAndReleasesItsPort() throws Exception {
        Options o =
                Options.parse(
                        "server",
                        "-m",
                        "unused",
                        "--port",
                        "0",
                        "--temp",
                        "0",
                        "--api-key",
                        "test");
        var capture = new CliFixtures.Capture("");
        var config =
                Server.config(
                        o, o.sampling(com.qxotic.jinfer.chat.LoadedModel.SamplingDefaults.NONE));
        var limits = config.limits();
        config =
                config.withLimits(
                        new com.qxotic.jinfer.server.ServerConfig.Limits(
                                limits.threads(),
                                limits.queueCapacity(),
                                limits.maxBodyBytes(),
                                limits.grammar(),
                                limits.writeTimeout(),
                                limits.requestTimeout(),
                                Duration.ZERO));
        int port;
        try (var engine = CliFixtures.engine(new CliFixtures.Template());
                var running = Server.startLanguage(engine, config, capture.io);
                var client =
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build()) {
            port = running.address().getPort();
            var request =
                    HttpRequest.newBuilder(
                                    URI.create("http://127.0.0.1:" + port + "/v1/chat/completions"))
                            .timeout(Duration.ofSeconds(10))
                            .header("Content-Type", "application/json")
                            .POST(
                                    HttpRequest.BodyPublishers.ofString(
                                            "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}"));
            assertEquals(
                    401,
                    client.send(request.build(), HttpResponse.BodyHandlers.ofString())
                            .statusCode());
            var response =
                    client.send(
                            request.header("Authorization", "Bearer test").build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode(), response.body());
            assertTrue(response.body().contains("xxxx"), response.body());
            assertEquals("", capture.out());
            assertTrue(capture.err().contains(":" + port));
        }
        try (var socket = new java.net.ServerSocket()) {
            socket.setReuseAddress(true);
            socket.bind(new java.net.InetSocketAddress("127.0.0.1", port));
        }
    }

    @Test
    void waitingClosesOnNormalReturnFailureAndInterruption() {
        AtomicInteger closed = new AtomicInteger();
        assertEquals(0, Server.await(() -> {}, closed::incrementAndGet));
        assertThrows(
                IllegalStateException.class,
                () ->
                        Server.await(
                                () -> {
                                    throw new IllegalStateException("test");
                                },
                                closed::incrementAndGet));
        try {
            assertEquals(
                    130,
                    Server.await(
                            () -> {
                                throw new InterruptedException();
                            },
                            closed::incrementAndGet));
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
        }
        assertEquals(3, closed.get());
    }

    @Test
    void clientsCanOverrideDefaultsAndUseStreamingWhileCorsIsPreserved() throws Exception {
        Options o =
                Options.parse(
                        "server",
                        "-m",
                        "unused",
                        "--port",
                        "0",
                        "--temp",
                        "0",
                        "-n",
                        "4",
                        "--cors-origin",
                        "https://client.example");
        var capture = new CliFixtures.Capture("");
        try (var engine = CliFixtures.engine(new CliFixtures.Template());
                var running = Server.startLanguage(engine, fastConfig(o), capture.io);
                var client = HttpClient.newHttpClient()) {
            URI uri =
                    URI.create(
                            "http://127.0.0.1:"
                                    + running.address().getPort()
                                    + "/v1/chat/completions");
            String messages = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]";
            var normal =
                    client.send(
                            request(uri, "{" + messages + ",\"max_tokens\":2}"),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, normal.statusCode(), normal.body());
            var choices = (List<?>) Json.parseMap(normal.body()).get("choices");
            var message = (Map<?, ?>) ((Map<?, ?>) choices.getFirst()).get("message");
            assertEquals("xx", message.get("content"));
            assertEquals(
                    "https://client.example",
                    normal.headers().firstValue("Access-Control-Allow-Origin").orElseThrow());
            var stream =
                    client.send(
                            request(uri, "{" + messages + ",\"stream\":true}"),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, stream.statusCode(), stream.body());
            assertTrue(
                    stream.headers()
                            .firstValue("Content-Type")
                            .orElseThrow()
                            .contains("text/event-stream"));
            assertTrue(stream.body().contains("data: [DONE]"));
            assertEquals("", capture.out());
        }
    }

    @Test
    void malformedHttpRequestsDoNotPoisonTheNextRequest() throws Exception {
        Options o =
                Options.parse("server", "-m", "unused", "--port", "0", "--temp", "0", "-n", "2");
        var capture = new CliFixtures.Capture("");
        try (var engine = CliFixtures.engine(new CliFixtures.Template());
                var running = Server.startLanguage(engine, fastConfig(o), capture.io);
                var client = HttpClient.newHttpClient()) {
            URI uri =
                    URI.create(
                            "http://127.0.0.1:"
                                    + running.address().getPort()
                                    + "/v1/chat/completions");
            assertEquals(
                    400,
                    client.send(request(uri, "not json"), HttpResponse.BodyHandlers.ofString())
                            .statusCode());
            var good =
                    client.send(
                            request(uri, "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, good.statusCode(), good.body());
        }
    }

    @Test
    void occupiedPortReportsTheRemedyAndLeavesTheEngineOwnedByTheCaller() throws Exception {
        try (var socket =
                        new java.net.ServerSocket(
                                0, 1, java.net.InetAddress.getByName("127.0.0.1"));
                var engine = CliFixtures.engine(new CliFixtures.Template())) {
            Options o =
                    Options.parse(
                            "server",
                            "-m",
                            "unused",
                            "--port",
                            Integer.toString(socket.getLocalPort()));
            var error =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Server.startLanguage(
                                            engine, fastConfig(o), new CliFixtures.Capture("").io));
            assertTrue(error.getMessage().contains("already in use"));
            assertTrue(error.getMessage().contains("--port"));
            assertEquals(4096, engine.contextCapacity());
        }
    }

    @Test
    void transcriptionRejectsEveryUnsupportedSettingEvenWhenItEqualsADefault() {
        for (String[] setting :
                new String[][] {
                    {"--queue-depth", "4"},
                    {"--cache", "c.jkv"},
                    {"--no-grammar"},
                    {"--raw-prompt"},
                    {"--temp", "0"},
                    {"--think", "on"},
                    {"--context-capacity", "4096"},
                    {"--batch-capacity", "512"}
                }) {
            var args = new java.util.ArrayList<>(List.of("server", "-m", "unused"));
            args.addAll(List.of(setting));
            Options o = Options.parse(args.toArray(String[]::new));
            assertThrows(
                    Options.Usage.class,
                    () -> Server.validateTranscription(o),
                    String.join(" ", setting));
        }
    }

    private static HttpRequest request(URI uri, String json) {
        return HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(5))
                .header("Content-Type", "application/json")
                .header("Origin", "https://client.example")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
    }

    private static ServerConfig fastConfig(Options o) {
        var config =
                Server.config(
                        o, o.sampling(com.qxotic.jinfer.chat.LoadedModel.SamplingDefaults.NONE));
        var l = config.limits();
        return config.withLimits(
                new ServerConfig.Limits(
                        l.threads(),
                        l.queueCapacity(),
                        l.maxBodyBytes(),
                        l.grammar(),
                        l.writeTimeout(),
                        l.requestTimeout(),
                        Duration.ZERO));
    }
}
