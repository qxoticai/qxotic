package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.server.ServerConfig;
import java.io.IOException;
import java.net.InetSocketAddress;
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
    void listeningUrlsUseTheRequestedHostAndTheActualPort() {
        for (String[] hosts :
                new String[][] {
                    {"0.0.0.0", "127.0.0.1", "; bound to 0.0.0.0"},
                    {"::", "[::1]", "; bound to ::"},
                    {"127.0.0.1", "127.0.0.1", ""},
                    {"localhost", "localhost", ""}
                }) {
            var capture = new CliFixtures.Capture("");
            Server.listening(capture.io.err(), new InetSocketAddress(hosts[0], 0), 54920, "API");
            assertEquals(
                    "listening   http://" + hosts[1] + ":54920 (API" + hosts[2] + ")\n",
                    capture.err().replace("\r\n", "\n"));
        }
    }

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
        assertEquals(5L << 20, c.limits().maxBodyBytes());
        assertEquals(Duration.ZERO, c.limits().requestTimeout());
        assertEquals(Duration.ofSeconds(9), c.limits().writeTimeout());
        assertEquals(32, c.defaults().maxOutputTokens());
        assertThrows(
                Options.UsageException.class,
                () -> Server.validateTask(o, "a transcription server"));
        Server.validateTask(
                Options.parse("server", "-m", "m", "--threads", "2"), "a transcription server");
    }

    @Test
    void authenticationAndInvalidLimitsFailClearly() {
        Options insecure = Options.parse("server", "-m", "unused", "--host", "0.0.0.0");
        assertThrows(Options.UsageException.class, () -> Server.config(insecure, null));
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
                    {"--max-body-mb", "0"},
                    {"--write-timeout", "0"}
                }) {
            String[] args = {"server", "-m", "unused", tail[0], tail[1]};
            assertThrows(Options.UsageException.class, () -> Options.parse(args));
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
    void theBindAddressKeepsTheHostAsTyped() {
        // resolved alone, :: is named 0:0:0:0:0:0:0:0 - the port-in-use message printed that
        for (String host : List.of("::", "::1", "0.0.0.0", "localhost", "127.0.0.1")) {
            var o = Options.parse("server", "-m", "x", "--host", host, "--api-key", "k");
            assertEquals(host, Server.config(o, null).bind().getHostString());
        }
        // a scoped link-local keeps its scope, which a Linux bind requires
        var o = Options.parse("server", "-m", "x", "--host", "fe80::1%1", "--api-key", "k");
        var bind = Server.config(o, null).bind();
        assertEquals("fe80::1%1", bind.getHostString());
        assertEquals(1, ((java.net.Inet6Address) bind.getAddress()).getScopeId());
    }

    @Test
    void occupiedPortNamesTheHostAsTyped() throws Exception {
        try (var socket = new java.net.ServerSocket(0, 1, java.net.InetAddress.getByName("::"));
                var engine = CliFixtures.engine(new CliFixtures.Template())) {
            Options o =
                    Options.parse(
                            "server",
                            "-m",
                            "unused",
                            "--host",
                            "::",
                            "--api-key",
                            "k",
                            "--port",
                            Integer.toString(socket.getLocalPort()));
            var error =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Server.startLanguage(
                                            engine, fastConfig(o), new CliFixtures.Capture("").io));
            assertTrue(error.getMessage().contains(" on :: is already in use"), error.getMessage());
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
                    Options.UsageException.class,
                    () -> Server.validateTask(o, "a transcription server"),
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
                        l.maxBodyBytes(),
                        l.grammar(),
                        l.writeTimeout(),
                        l.requestTimeout(),
                        Duration.ZERO));
    }

    /** Probes answer while every admission permit is held; only generation waits or is refused. */
    @Test
    void probesAnswerWhileTheGateIsSaturated() throws Exception {
        Options o =
                Options.parse(
                        "server",
                        "-m",
                        "unused",
                        "--port",
                        "0",
                        "--concurrency",
                        "1",
                        "--write-timeout",
                        "5");
        var capture = new CliFixtures.Capture("");
        var config =
                Server.config(
                        o, o.sampling(com.qxotic.jinfer.chat.LoadedModel.SamplingDefaults.NONE));
        var limits = config.limits(); // no shutdown grace: the held requests never complete
        config =
                config.withLimits(
                        new ServerConfig.Limits(
                                limits.threads(),
                                limits.maxBodyBytes(),
                                limits.grammar(),
                                limits.writeTimeout(),
                                limits.requestTimeout(),
                                Duration.ZERO));
        try (var engine = CliFixtures.engine(new CliFixtures.Template());
                var running = Server.startLanguage(engine, config, capture.io);
                var client =
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build()) {
            String base = "http://127.0.0.1:" + running.address().getPort();
            // a request that announces a body and never sends it holds its permit inside the
            // gate, reading until the write timeout - the only permit, at concurrency 1
            var held = new java.util.ArrayList<java.net.Socket>();
            try {
                for (int i = 0; i < 1; i++) {
                    var socket = new java.net.Socket("127.0.0.1", running.address().getPort());
                    socket.getOutputStream()
                            .write(
                                    ("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                                    + "Content-Type: application/json\r\n"
                                                    + "Content-Length: 64\r\n\r\n")
                                            .getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                    socket.getOutputStream().flush();
                    held.add(socket);
                }
                Thread.sleep(500);
                var refused =
                        client.send(
                                HttpRequest.newBuilder(URI.create(base + "/v1/completions"))
                                        .timeout(Duration.ofSeconds(5))
                                        .header("Content-Type", "application/json")
                                        .POST(
                                                HttpRequest.BodyPublishers.ofString(
                                                        "{\"prompt\":\"Hi\",\"max_tokens\":1}"))
                                        .build(),
                                HttpResponse.BodyHandlers.ofString());
                assertEquals(503, refused.statusCode(), "the gate is saturated: " + refused.body());
                for (String probe : List.of("/health", "/props", "/v1/models", "/metrics")) {
                    var answer =
                            client.send(
                                    HttpRequest.newBuilder(URI.create(base + probe))
                                            .timeout(Duration.ofSeconds(5))
                                            .build(),
                                    HttpResponse.BodyHandlers.ofString());
                    assertEquals(200, answer.statusCode(), probe + ": " + answer.body());
                }
            } finally {
                for (var socket : held) socket.close();
            }
        }
    }

    /** The transcription server offers the language server's probes, under the same rules. */
    @Test
    void transcriptionServerProbesMatchTheLanguageServers() throws Exception {
        Options o =
                Options.parse(
                        "server",
                        "-m",
                        "unused",
                        "--port",
                        "0",
                        "--api-key",
                        "k",
                        "--concurrency",
                        "1",
                        "--write-timeout",
                        "5");
        var config = Server.config(o, null);
        var limits = config.limits();
        config =
                config.withLimits(
                        new ServerConfig.Limits(
                                limits.threads(),
                                limits.maxBodyBytes(),
                                limits.grammar(),
                                limits.writeTimeout(),
                                limits.requestTimeout(),
                                Duration.ZERO));
        var model = new TranscribeTest.Transcriber();
        try (var running =
                        com.qxotic.jinfer.server.TranscriptionServer.start(
                                model, "asr.gguf", config);
                var client =
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build()) {
            String base = "http://127.0.0.1:" + running.address().getPort();
            java.util.function.Function<String, HttpRequest.Builder> get =
                    path ->
                            HttpRequest.newBuilder(URI.create(base + path))
                                    .timeout(Duration.ofSeconds(5));
            // the liveness probe carries no key; the model card and the scrape do
            var health =
                    client.send(get.apply("/health").build(), HttpResponse.BodyHandlers.ofString());
            assertEquals(200, health.statusCode(), health.body());
            assertTrue(health.body().contains("\"busy\":false"), health.body());
            assertEquals(
                    401,
                    client.send(
                                    get.apply("/v1/models").build(),
                                    HttpResponse.BodyHandlers.ofString())
                            .statusCode());
            var card =
                    client.send(
                            get.apply("/v1/models").header("Authorization", "Bearer k").build(),
                            HttpResponse.BodyHandlers.ofString());
            assertTrue(card.body().contains("\"input_modalities\":[\"audio\"]"), card.body());
            var props =
                    client.send(
                            get.apply("/props").header("Authorization", "Bearer k").build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, props.statusCode(), props.body());
            assertEquals("asr.gguf", Json.parseMap(props.body()).get("model"));
            assertEquals(
                    16000, ((Number) Json.parseMap(props.body()).get("sample_rate")).intValue());
            for (String path :
                    List.of("/health", "/props", "/metrics", "/v1/models", "/v1/models/asr.gguf")) {
                if (!path.equals("/health"))
                    assertEquals(
                            401,
                            client.send(
                                            get.apply(path).build(),
                                            HttpResponse.BodyHandlers.ofString())
                                    .statusCode(),
                            path);
                for (String method : List.of("POST", "PUT", "DELETE", "PATCH")) {
                    var rejected =
                            client.send(
                                    get.apply(path)
                                            .header("Authorization", "Bearer k")
                                            .method(method, HttpRequest.BodyPublishers.noBody())
                                            .build(),
                                    HttpResponse.BodyHandlers.ofString());
                    assertEquals(405, rejected.statusCode(), method + " " + path);
                    assertEquals(
                            "GET, OPTIONS", rejected.headers().firstValue("Allow").orElseThrow());
                }
                assertEquals(
                        204,
                        client.send(
                                        get.apply(path)
                                                .method(
                                                        "OPTIONS",
                                                        HttpRequest.BodyPublishers.noBody())
                                                .build(),
                                        HttpResponse.BodyHandlers.ofString())
                                .statusCode(),
                        path);
                assertEquals(
                        404,
                        client.send(
                                        get.apply(path + "XYZ")
                                                .header("Authorization", "Bearer k")
                                                .build(),
                                        HttpResponse.BodyHandlers.ofString())
                                .statusCode(),
                        path);
            }
            // one transcription, then the scrape counts it
            var body = new java.io.ByteArrayOutputStream();
            body.write(
                    "--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
                            .getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            body.write(
                    com.qxotic.jinfer.codecs.AudioCodec.wav(
                            new com.qxotic.jinfer.media.Media.Audio(new float[16000], 16000, 1)));
            body.write("\r\n--b--\r\n".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            var heard =
                    client.send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/audio/transcriptions"))
                                    .timeout(Duration.ofSeconds(10))
                                    .header("Authorization", "Bearer k")
                                    .header("Content-Type", "multipart/form-data; boundary=b")
                                    .POST(
                                            HttpRequest.BodyPublishers.ofByteArray(
                                                    body.toByteArray()))
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, heard.statusCode(), heard.body());
            // a file no decoder reads: what the client can act on, never the decoder's stderr
            var corrupt =
                    client.send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/audio/transcriptions"))
                                    .timeout(Duration.ofSeconds(30))
                                    .header("Authorization", "Bearer k")
                                    .header("Content-Type", "multipart/form-data; boundary=b")
                                    .POST(
                                            HttpRequest.BodyPublishers.ofString(
                                                    "--b\r\n"
                                                            + "Content-Disposition: form-data;"
                                                            + " name=\"file\";"
                                                            + " filename=\"x.wav\"\r\n\r\n"
                                                            + "not audio at all\r\n"
                                                            + "--b--\r\n"))
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(400, corrupt.statusCode(), corrupt.body());
            assertEquals(
                    "cannot decode audio: unsupported or corrupt file",
                    ((Map<?, ?>) Json.parseMap(corrupt.body()).get("error")).get("message"));
            // any other path: the same JSON 404 envelope as every other refusal
            var unknown =
                    client.send(
                            get.apply("/nope")
                                    .header("Authorization", "Bearer k")
                                    .POST(HttpRequest.BodyPublishers.noBody())
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(404, unknown.statusCode(), unknown.body());
            assertTrue(unknown.body().startsWith("{\"error\":{"), unknown.body());
            var metrics =
                    client.send(
                            get.apply("/metrics").header("Authorization", "Bearer k").build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, metrics.statusCode(), metrics.body());
            assertTrue(
                    metrics.body().contains("jinfer_transcriptions_completed_total 1\n"),
                    metrics.body());
            assertTrue(
                    metrics.body().contains("jinfer_transcribed_audio_seconds_total 1.0\n"),
                    metrics.body());
            // one request holding the single permit: the next is refused, the probe still answers
            try (var held = new java.net.Socket("127.0.0.1", running.address().getPort())) {
                held.getOutputStream()
                        .write(
                                ("POST /v1/audio/transcriptions HTTP/1.1\r\n"
                                     + "Host: localhost\r\n"
                                     + "Authorization: Bearer k\r\n"
                                     + "Content-Type: multipart/form-data; boundary=b\r\n"
                                     + "Content-Length: 64\r\n\r\n")
                                        .getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                held.getOutputStream().flush();
                Thread.sleep(500);
                var refused =
                        client.send(
                                HttpRequest.newBuilder(
                                                URI.create(base + "/v1/audio/transcriptions"))
                                        .timeout(Duration.ofSeconds(5))
                                        .header("Authorization", "Bearer k")
                                        .header("Content-Type", "multipart/form-data; boundary=b")
                                        .POST(
                                                HttpRequest.BodyPublishers.ofByteArray(
                                                        body.toByteArray()))
                                        .build(),
                                HttpResponse.BodyHandlers.ofString());
                assertEquals(503, refused.statusCode(), refused.body());
                assertNotNull(refused.headers().firstValue("Retry-After").orElse(null));
                var busy =
                        client.send(
                                get.apply("/health").build(), HttpResponse.BodyHandlers.ofString());
                assertEquals(200, busy.statusCode(), busy.body());
                assertTrue(busy.body().contains("\"busy\":true"), busy.body());
                assertEquals(
                        200,
                        client.send(
                                        get.apply("/props")
                                                .header("Authorization", "Bearer k")
                                                .build(),
                                        HttpResponse.BodyHandlers.ofString())
                                .statusCode());
            }
        }
    }

    /** One limit: --concurrency requests are held, the next is refused naming that number. */
    @Test
    void concurrencyIsTheOnlyLimitAndTheRefusalNamesIt() throws Exception {
        Options o =
                Options.parse(
                        "server",
                        "-m",
                        "unused",
                        "--port",
                        "0",
                        "--concurrency",
                        "3",
                        "--write-timeout",
                        "5");
        var capture = new CliFixtures.Capture("");
        var config =
                Server.config(
                        o, o.sampling(com.qxotic.jinfer.chat.LoadedModel.SamplingDefaults.NONE));
        var limits = config.limits();
        config =
                config.withLimits(
                        new ServerConfig.Limits(
                                limits.threads(),
                                limits.maxBodyBytes(),
                                limits.grammar(),
                                limits.writeTimeout(),
                                limits.requestTimeout(),
                                Duration.ZERO));
        try (var engine = CliFixtures.engine(new CliFixtures.Template());
                var running = Server.startLanguage(engine, config, capture.io);
                var client =
                        HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build()) {
            String base = "http://127.0.0.1:" + running.address().getPort();
            var held = new java.util.ArrayList<java.net.Socket>();
            try {
                for (int i = 0; i < 3; i++) { // every one of the three is admitted and waits
                    var socket = new java.net.Socket("127.0.0.1", running.address().getPort());
                    socket.getOutputStream()
                            .write(
                                    ("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                                    + "Content-Type: application/json\r\n"
                                                    + "Content-Length: 64\r\n\r\n")
                                            .getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                    socket.getOutputStream().flush();
                    held.add(socket);
                }
                Thread.sleep(500);
                var refused =
                        client.send(
                                HttpRequest.newBuilder(URI.create(base + "/v1/completions"))
                                        .timeout(Duration.ofSeconds(5))
                                        .header("Content-Type", "application/json")
                                        .POST(
                                                HttpRequest.BodyPublishers.ofString(
                                                        "{\"prompt\":\"Hi\",\"max_tokens\":1}"))
                                        .build(),
                                HttpResponse.BodyHandlers.ofString());
                assertEquals(503, refused.statusCode(), refused.body());
                assertTrue(
                        refused.body()
                                .contains("Server busy: 3 requests in flight; retry after 6 s"),
                        refused.body());
                assertEquals("6", refused.headers().firstValue("Retry-After").orElse(null));
                for (var socket : held)
                    assertFalse(socket.isClosed(), "an admitted request is still being read");
            } finally {
                for (var socket : held) socket.close();
            }
        }
    }
}
