package ai.qxotic.jota.examples.mnist;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.runtime.DefaultRuntimeRegistry;
import ai.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import ai.qxotic.jota.runtime.spi.RuntimeProbe;
import ai.qxotic.jota.runtime.c.CRuntimeProvider;
import ai.qxotic.jota.runtime.hip.HipRuntimeProvider;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class MnistDemoServer {
    private static final int PORT = 8080;

    private final Environment baseEnvironment;
    private final Device defaultDevice;
    private final Map<Device, Environment> backendEnvironments = new ConcurrentHashMap<>();
    private final Map<Device, MnistMlp> models = new ConcurrentHashMap<>();

    private MnistDemoServer(Environment baseEnvironment, Device defaultDevice) {
        this.baseEnvironment = baseEnvironment;
        this.defaultDevice = defaultDevice;
    }

    public static void main(String[] args) throws Exception {
        Device defaultDevice = configureBackend();
        new MnistDemoServer(Environment.global(), defaultDevice).start();
    }

    private static Device configureBackend() {
        String backend = System.getProperty("jota.backend", "panama");
        Environment current = Environment.global();
        Device device =
                backend == null || backend.isBlank()
                        ? current.defaultDevice()
                        : switch (backend.toLowerCase(java.util.Locale.ROOT)) {
                            case "hip" -> Device.HIP;
                            case "c" -> Device.C;
                            case "panama", "cpu" -> Device.PANAMA;
                            default -> throw new IllegalArgumentException("Unknown backend: " + backend);
                        };
        if (!current.runtimes().hasRuntime(device)) {
            tryRegisterFallbackProvider(current, device);
        }
        if (!current.runtimes().hasRuntime(device)) {
            System.err.println(
                    "Requested backend "
                            + device
                            + " is not available. Available backends: "
                            + current.runtimes().devices());
            if (device == Device.HIP) {
                System.err.println(
                        "HIP backend requires the HIP JNI library to be built and on"
                                + " java.library.path/LD_LIBRARY_PATH.");
            }
            throw new IllegalStateException("Backend not available: " + device);
        }
        ExecutionMode mode = resolveExecutionMode(device, current.executionMode());
        Environment.configureGlobal(
                new Environment(
                        device,
                        current.defaultFloat(),
                        current.runtimes(),
                        mode));
        return device;
    }

    private static ExecutionMode resolveExecutionMode(Device device, ExecutionMode fallback) {
        String configured = System.getProperty("jota.executionMode");
        if (configured != null && !configured.isBlank()) {
            return switch (configured.toLowerCase(java.util.Locale.ROOT)) {
                case "lazy" -> ExecutionMode.LAZY;
                case "eager" -> ExecutionMode.EAGER;
                default -> throw new IllegalArgumentException("Unknown execution mode: " + configured);
            };
        }
        return Device.PANAMA.equals(device) ? ExecutionMode.EAGER : fallback;
    }

    private static void tryRegisterFallbackProvider(Environment current, Device device) {
        if (!(current.runtimes() instanceof DefaultRuntimeRegistry registry)) {
            return;
        }
        DeviceRuntimeProvider provider;
        if (Device.HIP.equals(device)) {
            provider = new HipRuntimeProvider();
        } else if (Device.C.equals(device)) {
            provider = new CRuntimeProvider();
        } else {
            provider = null;
        }
        if (provider == null) {
            return;
        }
        RuntimeProbe probe = provider.probe();
        if (!probe.isAvailable()) {
            return;
        }
        registry.register(provider.create());
    }

    private void start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/", this::handleStatic);
        server.createContext("/app.js", this::handleStatic);
        server.createContext("/styles.css", this::handleStatic);
        server.createContext("/runtime-info", this::handleRuntimeInfo);
        server.createContext("/infer", this::handleInfer);
        server.createContext("/benchmark", this::handleBenchmark);
        server.createContext("/debug", this::handleDebug);
        server.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("MNIST demo running on http://localhost:" + PORT);
    }

    private void handleStatic(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        if (path.equals("/")) {
            serveResource(exchange, "/web/index.html", "text/html; charset=utf-8");
            return;
        }
        if (path.equals("/app.js")) {
            serveResource(exchange, "/web/app.js", "application/javascript; charset=utf-8");
            return;
        }
        if (path.equals("/styles.css")) {
            serveResource(exchange, "/web/styles.css", "text/css; charset=utf-8");
            return;
        }
        send(exchange, 404, "Not found");
    }

    private void handleInfer(HttpExchange exchange) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) {
            send(exchange, 405, "Method not allowed");
            return;
        }
        try {
            Device device = backendForRequest(exchange);
            float[] input = readFloatPayload(exchange.getRequestBody());
            int batch = input.length / MnistMlp.INPUT_SIZE;
            MnistMlp.InferenceResult result = infer(device, input, batch);
            String json =
                    buildInferJson(
                            batch, result.preds(), result.confidences(), result.probs(), 0, 0);
            sendJson(exchange, json);
        } catch (Exception e) {
            e.printStackTrace();
            sendJson(exchange, buildErrorJson(e));
        }
    }

    private void handleBenchmark(HttpExchange exchange) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) {
            send(exchange, 405, "Method not allowed");
            return;
        }
        try {
            Device device = backendForRequest(exchange);
            float[] input = readFloatPayload(exchange.getRequestBody());
            int batch = input.length / MnistMlp.INPUT_SIZE;
            MnistMlp.InferenceResult result = infer(device, input, batch);
            MnistMlp.InferenceTimings timings = benchmark(device, input, batch);
            String json =
                    buildInferJson(
                            batch,
                            result.preds(),
                            result.confidences(),
                            result.probs(),
                            timings.batchMs(),
                            timings.seqMs());
            sendJson(exchange, json);
        } catch (Exception e) {
            e.printStackTrace();
            sendJson(exchange, buildErrorJson(e));
        }
    }

    private void handleRuntimeInfo(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) {
            send(exchange, 405, "Method not allowed");
            return;
        }
        StringBuilder builder = new StringBuilder(256);
        builder.append('{');
        builder.append("\"defaultBackend\":\"").append(defaultDevice.leafName()).append("\",");
        builder.append("\"availableBackends\":[");
        boolean first = true;
        for (Device device : baseEnvironment.runtimes().devices()) {
            if (Device.PANAMA.equals(device)) {
                if (!first) {
                    builder.append(',');
                }
                builder.append("\"panama\"");
                first = false;
            } else if (Device.C.equals(device) || Device.HIP.equals(device)) {
                if (!first) {
                    builder.append(',');
                }
                builder.append('"').append(device.leafName()).append('"');
                first = false;
            }
        }
        builder.append("]}");
        sendJson(exchange, builder.toString());
    }

    private void handleDebug(HttpExchange exchange) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) {
            send(exchange, 405, "Method not allowed");
            return;
        }
        float[] input = readFloatPayload(exchange.getRequestBody());
        int batch = input.length / MnistMlp.INPUT_SIZE;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0;
        int nonzero = 0;
        for (float v : input) {
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
            if (v != 0.0f) {
                nonzero++;
            }
        }
        double mean = sum / input.length;
        StringBuilder builder = new StringBuilder(256);
        builder.append('{');
        builder.append("\"batch\":").append(batch).append(',');
        builder.append("\"min\":").append(formatFloat(min)).append(',');
        builder.append("\"max\":").append(formatFloat(max)).append(',');
        builder.append("\"mean\":")
                .append(String.format(java.util.Locale.ROOT, "%.6f", mean))
                .append(',');
        builder.append("\"nonzero\":").append(nonzero).append(',');
        builder.append("\"first\":[");
        int limit = Math.min(32, input.length);
        for (int i = 0; i < limit; i++) {
            if (i > 0) {
                builder.append(',');
            }
            builder.append(formatFloat(input[i]));
        }
        builder.append("]}");
        sendJson(exchange, builder.toString());
    }

    private float[] readFloatPayload(InputStream input) throws IOException {
        byte[] bytes = input.readAllBytes();
        if (bytes.length % 4 != 0) {
            throw new IllegalArgumentException("Payload length must be multiple of 4");
        }
        int count = bytes.length / 4;
        float[] out = new float[count];
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            out[i] = buffer.getFloat();
        }
        if (count % MnistMlp.INPUT_SIZE != 0) {
            throw new IllegalArgumentException("Payload does not align to 784 floats");
        }
        return out;
    }

    private MnistMlp.InferenceResult infer(Device device, float[] input, int batch) {
        Environment env = environmentFor(device);
        MnistMlp model = modelFor(device, env);
        return Environment.with(env, () -> model.infer(input, batch));
    }

    private MnistMlp.InferenceTimings benchmark(Device device, float[] input, int batch) {
        Environment env = environmentFor(device);
        MnistMlp model = modelFor(device, env);
        return Environment.with(env, () -> model.benchmark(input, batch));
    }

    private MnistMlp modelFor(Device device, Environment env) {
        return models.computeIfAbsent(device, __ -> Environment.with(env, MnistMlp::new));
    }

    private Environment environmentFor(Device device) {
        return backendEnvironments.computeIfAbsent(
                device,
                d ->
                        new Environment(
                                d,
                                baseEnvironment.defaultFloat(),
                                baseEnvironment.runtimes(),
                                resolveExecutionMode(d, baseEnvironment.executionMode())));
    }

    private Device backendForRequest(HttpExchange exchange) {
        String selected = queryParam(exchange, "backend");
        if (selected == null || selected.isBlank()) {
            return defaultDevice;
        }
        Device device =
                switch (selected.toLowerCase(java.util.Locale.ROOT)) {
                    case "hip" -> Device.HIP;
                    case "c" -> Device.C;
                    case "panama", "cpu" -> Device.PANAMA;
                    default -> throw new IllegalArgumentException("Unknown backend: " + selected);
                };
        if (!baseEnvironment.runtimes().hasRuntime(device)) {
            throw new IllegalArgumentException("Backend not available: " + selected);
        }
        return device;
    }

    private static String queryParam(HttpExchange exchange, String key) {
        String raw = exchange.getRequestURI().getRawQuery();
        if (raw == null || raw.isBlank()) {
            return null;
        }
        for (String part : raw.split("&")) {
            int eq = part.indexOf('=');
            String k = eq >= 0 ? part.substring(0, eq) : part;
            if (!k.equals(key)) {
                continue;
            }
            String value = eq >= 0 ? part.substring(eq + 1) : "";
            return URLDecoder.decode(value, StandardCharsets.UTF_8);
        }
        return null;
    }

    private String buildInferJson(
            int batch, int[] preds, float[] confidences, float[] probs, long batchMs, long seqMs) {
        StringBuilder builder = new StringBuilder(1024);
        builder.append('{');
        builder.append("\"status\":\"ok\",");
        builder.append("\"batch\":").append(batch).append(',');
        builder.append("\"predCount\":").append(preds.length).append(',');
        builder.append("\"preds\":[");
        for (int i = 0; i < preds.length; i++) {
            if (i > 0) {
                builder.append(',');
            }
            builder.append(preds[i]);
        }
        builder.append("],");
        builder.append("\"confidences\":[");
        for (int i = 0; i < confidences.length; i++) {
            if (i > 0) {
                builder.append(',');
            }
            builder.append(formatFloat(confidences[i]));
        }
        builder.append("],");
        builder.append("\"probs\":[");
        for (int i = 0; i < probs.length; i++) {
            if (i > 0) {
                builder.append(',');
            }
            builder.append(formatFloat(probs[i]));
        }
        builder.append(']');
        if (batchMs > 0 || seqMs > 0) {
            builder.append(',');
            builder.append("\"batchMs\":").append(batchMs).append(',');
            builder.append("\"seqMs\":").append(seqMs);
        }
        builder.append('}');
        return builder.toString();
    }

    private String buildErrorJson(Exception e) {
        StringBuilder builder = new StringBuilder(256);
        builder.append('{');
        builder.append("\"status\":\"error\",");
        builder.append("\"message\":\"").append(escapeJson(e.getMessage())).append("\"");
        builder.append('}');
        return builder.toString();
    }

    private String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private String formatFloat(float value) {
        return String.format(java.util.Locale.ROOT, "%.6f", value);
    }

    private void serveResource(HttpExchange exchange, String resource, String contentType)
            throws IOException {
        try (InputStream stream = MnistDemoServer.class.getResourceAsStream(resource)) {
            if (stream == null) {
                send(exchange, 404, "Not found");
                return;
            }
            byte[] bytes = stream.readAllBytes();
            Headers headers = exchange.getResponseHeaders();
            headers.set("Content-Type", contentType);
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        }
    }

    private void sendJson(HttpExchange exchange, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private void send(HttpExchange exchange, int code, String message) throws IOException {
        byte[] bytes = message.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }
}
