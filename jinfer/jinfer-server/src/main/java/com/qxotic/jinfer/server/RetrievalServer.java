package com.qxotic.jinfer.server;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import java.util.function.IntSupplier;

/**
 * The retrieval transport over one embedding or one reranking model: {@code POST /v1/embeddings}
 * (OpenAI's shape) or {@code POST /v1/rerank} (llama.cpp's), plus the probes every jinfer server
 * answers. One request computes at a time over one state sized at start: the compute pool is
 * process-wide, so two concurrent batches would only fight over it.
 */
public final class RetrievalServer {

    /** What the transport needs from an embedding model; a test fakes it. */
    interface Embedder {
        int dimension();

        /** Equal to {@link #dimension()} for a fixed-width model; below it, Matryoshka. */
        int minimumDimension();

        int contextCapacity();

        /** Embeds every text in order; returns the input-token count. */
        int embed(List<String> texts, int dimension, Consumer<float[]> sink);

        void close();
    }

    /** What the transport needs from a reranker; a test fakes it. */
    interface Ranker {
        int contextCapacity();

        /** Scores every document against the query, in order; returns the token count. */
        int score(String query, List<String> documents, DoubleConsumer sink);

        void close();
    }

    private final String servedModel;
    private final ServerConfig config;
    private final TaskTransport transport;
    private final ReentrantLock compute = new ReentrantLock(true);
    private final AtomicLong requests = new AtomicLong(), promptTokens = new AtomicLong();

    private RetrievalServer(String servedModel, ServerConfig config) throws IOException {
        if (servedModel == null || servedModel.isBlank())
            throw new IllegalArgumentException("servedModel is required");
        this.servedModel = servedModel;
        this.config = config;
        this.transport = new TaskTransport(config);
    }

    /** A running transport. The caller keeps the model; the server's own state closes with it. */
    public static final class Running implements AutoCloseable {
        private final TaskTransport.Running running;

        private Running(TaskTransport.Running running) {
            this.running = running;
        }

        public InetSocketAddress address() {
            return running.address();
        }

        /** Blocks until {@link #close()} is called. */
        public void await() throws InterruptedException {
            running.await();
        }

        @Override
        public void close() {
            running.close();
        }
    }

    /** Serves {@code embedder} over a state of {@code contextCapacity} tokens. Does not block. */
    public static Running start(
            LoadedEmbedder<?> embedder,
            int contextCapacity,
            String servedModel,
            ServerConfig config)
            throws IOException {
        checkCapacity(contextCapacity, embedder.model().configuration().maxContextLength());
        return start(adapt(embedder, contextCapacity), servedModel, config);
    }

    /** Serves {@code reranker} over a state of {@code contextCapacity} tokens. Does not block. */
    public static Running start(
            LoadedReranker<?> reranker,
            int contextCapacity,
            String servedModel,
            ServerConfig config)
            throws IOException {
        checkCapacity(contextCapacity, reranker.model().configuration().maxContextLength());
        return start(adapt(reranker, contextCapacity), servedModel, config);
    }

    static Running start(Embedder embedder, String servedModel, ServerConfig config)
            throws IOException {
        RetrievalServer server = new RetrievalServer(servedModel, config);
        Map<String, Object> props =
                JsonCodec.object(
                        "model",
                        servedModel,
                        "task",
                        "embedding",
                        "n_ctx",
                        embedder.contextCapacity(),
                        "dimension",
                        embedder.dimension(),
                        "min_dimension",
                        embedder.minimumDimension());
        return server.serve(
                "embedding",
                props,
                "/v1/embeddings",
                exchange -> server.embeddings(exchange, embedder),
                embedder::close);
    }

    static Running start(Ranker ranker, String servedModel, ServerConfig config)
            throws IOException {
        RetrievalServer server = new RetrievalServer(servedModel, config);
        Map<String, Object> props =
                JsonCodec.object(
                        "model", servedModel, "task", "rerank", "n_ctx", ranker.contextCapacity());
        return server.serve(
                "rerank",
                props,
                "/v1/rerank",
                exchange -> server.rerank(exchange, ranker),
                ranker::close);
    }

    private static void checkCapacity(int contextCapacity, int maximum) {
        if (contextCapacity < 1 || contextCapacity > maximum)
            throw new IllegalArgumentException(
                    "contextCapacity " + contextCapacity + " is outside 1.." + maximum);
    }

    private Running serve(
            String task,
            Map<String, Object> props,
            String path,
            TaskTransport.Work work,
            Runnable close) {
        transport.probes(
                servedModel,
                "text",
                () -> props,
                () -> {
                    StringBuilder sb = new StringBuilder();
                    Metrics.metric(sb, "jinfer_uptime_seconds", "gauge", transport.uptimeSeconds());
                    Metrics.metric(
                            sb,
                            "jinfer_" + task + "_requests_completed_total",
                            "counter",
                            requests.get());
                    Metrics.metric(sb, "jinfer_prompt_tokens_total", "counter", promptTokens.get());
                    Metrics.metric(
                            sb,
                            "jinfer_" + task + "_requests_in_flight",
                            "gauge",
                            transport.inFlight());
                    return sb.toString();
                });
        transport.work(path, work);
        // the state closes after the request computing on it: the fair lock waits it out
        Runnable closing =
                () -> {
                    compute.lock();
                    try {
                        close.run();
                    } finally {
                        compute.unlock();
                    }
                };
        return new Running(transport.start("POST " + path, "jinfer-" + task + "-handler", closing));
    }

    /** Runs model work under the lock and counts it; a too-long text is the client's 400. */
    private int compute(String param, IntSupplier work) {
        int tokens;
        compute.lock();
        try {
            tokens = work.getAsInt();
        } catch (IllegalArgumentException tooLong) {
            // the library names the text and its length, then a remedy for Java callers
            String message = Http.errorMessage(tooLong);
            int remedy = message.indexOf(" - raise");
            throw new Values.InvalidParam(
                    param,
                    "Invalid argument: "
                            + (remedy < 0 ? message : message.substring(0, remedy) + "; split it"));
        } finally {
            compute.unlock();
        }
        requests.incrementAndGet();
        promptTokens.addAndGet(tokens);
        return tokens;
    }

    private static Map<String, Object> usage(int tokens) {
        return JsonCodec.object("prompt_tokens", tokens, "total_tokens", tokens);
    }

    @SuppressWarnings("unchecked")
    private void embeddings(HttpExchange exchange, Embedder embedder) throws IOException {
        Map<String, Object> request = Http.readJsonObject(exchange, config.limits());
        if (request == null) return;
        Validation.validateModel(request, servedModel);
        Object input = request.get("input");
        Validation.requireParam(
                input instanceof String
                        || input instanceof List<?> items
                                && !items.isEmpty()
                                && items.stream().allMatch(String.class::isInstance),
                "input",
                "Invalid argument: input must be a string or a non-empty array of strings; token"
                        + " ids are not supported");
        List<String> texts = input instanceof String text ? List.of(text) : (List<String>) input;
        String format = Values.stringValue(request.get("encoding_format"), "float");
        Validation.requireParam(
                format.equals("float") || format.equals("base64"),
                "encoding_format",
                "Invalid argument: encoding_format must be float or base64; got '" + format + "'");
        int dimension =
                Values.intValue(request.get("dimensions"), "dimensions", embedder.dimension());
        Validation.requireParam(
                embedder.minimumDimension() <= dimension && dimension <= embedder.dimension(),
                "dimensions",
                "Invalid argument: dimensions must be within ["
                        + embedder.minimumDimension()
                        + ", "
                        + embedder.dimension()
                        + "]; got "
                        + dimension);

        List<float[]> vectors = new ArrayList<>(texts.size());
        int tokens = compute("input", () -> embedder.embed(texts, dimension, vectors::add));
        List<Object> data = new ArrayList<>(vectors.size());
        for (float[] vector : vectors) {
            data.add(
                    JsonCodec.object(
                            "object",
                            "embedding",
                            "index",
                            data.size(),
                            "embedding",
                            format.equals("base64") ? base64(vector) : floats(vector)));
        }
        Http.sendJson(
                exchange,
                200,
                JsonCodec.object(
                        "object",
                        "list",
                        "data",
                        data,
                        "model",
                        Requests.modelId(request, servedModel),
                        "usage",
                        usage(tokens)));
    }

    /** OpenAI's base64 encoding: the vector's float32 values, little-endian. */
    private static String base64(float[] vector) {
        ByteBuffer bytes = ByteBuffer.allocate(vector.length * Float.BYTES);
        bytes.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(vector);
        return Base64.getEncoder().encodeToString(bytes.array());
    }

    private static List<Float> floats(float[] vector) {
        List<Float> boxed = new ArrayList<>(vector.length);
        for (float value : vector) boxed.add(value);
        return boxed;
    }

    @SuppressWarnings("unchecked")
    private void rerank(HttpExchange exchange, Ranker ranker) throws IOException {
        Map<String, Object> request = Http.readJsonObject(exchange, config.limits());
        if (request == null) return;
        Validation.validateModel(request, servedModel);
        Validation.requireParam(
                request.get("query") instanceof String query && !query.isBlank(),
                "query",
                "Invalid argument: query must be a non-empty string");
        Validation.requireParam(
                request.get("documents") instanceof List<?> items
                        && items.stream().allMatch(String.class::isInstance),
                "documents",
                "Invalid argument: documents must be an array of strings");
        String query = (String) request.get("query");
        List<String> documents = (List<String>) request.get("documents");
        int topN = Values.intValue(request.get("top_n"), "top_n", documents.size());
        Validation.requireParam(
                request.get("top_n") == null || topN >= 1,
                "top_n",
                "Invalid argument: top_n must be at least 1");
        boolean returnDocuments =
                Values.booleanValue(request.get("return_documents"), "return_documents", false);

        double[] scores = new double[documents.size()];
        int[] at = {0};
        int tokens =
                compute(
                        "documents",
                        () -> ranker.score(query, documents, s -> scores[at[0]++] = s));
        // most relevant first; the sort is stable, so ties keep input order
        List<Integer> order = new ArrayList<>(scores.length);
        for (int i = 0; i < scores.length; i++) order.add(i);
        order.sort(Comparator.comparingDouble((Integer i) -> scores[i]).reversed());
        List<Object> results = new ArrayList<>();
        for (int index : order.subList(0, Math.min(topN, scores.length))) {
            Map<String, Object> result =
                    JsonCodec.object("index", index, "relevance_score", scores[index]);
            if (returnDocuments)
                result.put("document", JsonCodec.object("text", documents.get(index)));
            results.add(result);
        }
        Http.sendJson(
                exchange,
                200,
                JsonCodec.object(
                        "model",
                        Requests.modelId(request, servedModel),
                        "results",
                        results,
                        "usage",
                        usage(tokens)));
    }

    private static <S extends ContextState> Embedder adapt(
            LoadedEmbedder<S> loaded, int contextCapacity) {
        S state = loaded.model().newState(contextCapacity);
        return new Embedder() {
            public int dimension() {
                return loaded.dimension();
            }

            public int minimumDimension() {
                return loaded.minimumDimension();
            }

            public int contextCapacity() {
                return contextCapacity;
            }

            public int embed(List<String> texts, int dimension, Consumer<float[]> sink) {
                return loaded.embedAll(state, texts, dimension, sink);
            }

            public void close() {
                state.close();
            }
        };
    }

    private static <S extends ContextState> Ranker adapt(
            LoadedReranker<S> loaded, int contextCapacity) {
        S state = loaded.model().newState(contextCapacity);
        String instruction = loaded.reranker().defaultInstruction();
        return new Ranker() {
            public int contextCapacity() {
                return contextCapacity;
            }

            public int score(String query, List<String> documents, DoubleConsumer sink) {
                return loaded.scoreAll(state, instruction, query, documents, sink);
            }

            public void close() {
                state.close();
            }
        };
    }
}
