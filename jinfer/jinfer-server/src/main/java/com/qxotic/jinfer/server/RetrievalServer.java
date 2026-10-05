package com.qxotic.jinfer.server;

import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;

/**
 * The retrieval transport, over one embedding or one reranking model: {@code POST /v1/embeddings}
 * (OpenAI's shape) for an embedder, {@code POST /v1/rerank} (llama.cpp's, Jina's and Cohere's
 * shape; also {@code /rerank}, {@code /v1/reranking} and {@code /reranking}, llama.cpp's aliases)
 * for a reranker, and the {@code /v1/models}, {@code /health}, {@code /props} and {@code /metrics}
 * probes every jinfer server answers. Started by the CLI when {@code server} is given a retrieval
 * checkpoint.
 *
 * <p>One request computes at a time, over one state sized at start - the compute pool is the
 * process-wide one, so two concurrent batches would only fight over it - while admission, parsing
 * and validation stay concurrent. An embeddings request packs all its inputs into as few forward
 * passes as the context allows.
 */
public final class RetrievalServer {

    /** OpenAI's ceiling on one embeddings request's input array. */
    static final int MAX_INPUTS = 2048;

    /**
     * What the transport needs from an embedding model; {@link #start(LoadedEmbedder, int, String,
     * ServerConfig)} adapts a loaded one, and a test fakes it.
     */
    interface Embedder {
        int dimension();

        /** Equal to {@link #dimension()} for a fixed-width model; below it, Matryoshka. */
        int minimumDimension();

        String queryPrefix();

        String documentPrefix();

        int contextCapacity();

        /** The framed length of one input, as {@link #embed} will count it. */
        int tokens(String text);

        /** Embeds every text in order; returns the exact framed input-token count. */
        int embed(List<String> texts, int dimension, Consumer<float[]> sink);

        void close();
    }

    /** What the transport needs from a reranker; a test fakes it. */
    interface Ranker {
        int contextCapacity();

        /** Scores every document against the query, in order; returns the exact token count. */
        int score(String query, List<String> documents, DoubleConsumer sink);

        void close();
    }

    private final String servedModel;
    private final ServerConfig config;
    private final TaskTransport transport;
    private final ReentrantLock compute = new ReentrantLock(true);
    private final AtomicLong requests = new AtomicLong(),
            items = new AtomicLong(),
            promptTokens = new AtomicLong();

    private RetrievalServer(String servedModel, ServerConfig config) throws IOException {
        if (config == null) throw new IllegalArgumentException("config is required");
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

        /** Stops serving, waits out the request computing, then frees the server's state. */
        @Override
        public void close() {
            running.close();
        }
    }

    /**
     * Serves {@code embedder} on {@code POST /v1/embeddings}, packing each request into a state of
     * {@code contextCapacity} tokens - also the longest input it embeds. Does not block, prints
     * nothing, owns no shutdown hook.
     */
    public static Running start(
            LoadedEmbedder<?> embedder,
            int contextCapacity,
            String servedModel,
            ServerConfig config)
            throws IOException {
        if (embedder == null) throw new IllegalArgumentException("embedder is required");
        RetrievalServer server = new RetrievalServer(servedModel, config);
        return server.serveEmbeddings(
                adapt(embedder, capacity(embedder.model().configuration(), contextCapacity)));
    }

    /**
     * Serves {@code reranker} on {@code POST /v1/rerank}, scoring in a state of {@code
     * contextCapacity} tokens. Does not block, prints nothing, owns no shutdown hook.
     */
    public static Running start(
            LoadedReranker<?> reranker,
            int contextCapacity,
            String servedModel,
            ServerConfig config)
            throws IOException {
        if (reranker == null) throw new IllegalArgumentException("reranker is required");
        RetrievalServer server = new RetrievalServer(servedModel, config);
        return server.serveReranking(
                adapt(reranker, capacity(reranker.model().configuration(), contextCapacity)));
    }

    static Running start(Embedder embedder, String servedModel, ServerConfig config)
            throws IOException {
        return new RetrievalServer(servedModel, config).serveEmbeddings(embedder);
    }

    static Running start(Ranker ranker, String servedModel, ServerConfig config)
            throws IOException {
        return new RetrievalServer(servedModel, config).serveReranking(ranker);
    }

    private static int capacity(ContextConfiguration configuration, int contextCapacity) {
        int maximum = configuration.maxContextLength();
        if (contextCapacity < 1 || contextCapacity > maximum)
            throw new IllegalArgumentException(
                    "contextCapacity " + contextCapacity + " is outside 1.." + maximum);
        return contextCapacity;
    }

    private Map<String, Object> modelCard() {
        return JsonCodec.object(
                "id",
                servedModel,
                "object",
                "model",
                "created",
                0,
                "owned_by",
                "jinfer",
                "architecture",
                JsonCodec.object("input_modalities", List.of("text")));
    }

    private String exposition(String task) {
        StringBuilder sb = new StringBuilder();
        Metrics.metric(sb, "jinfer_uptime_seconds", "gauge", transport.uptimeSeconds());
        Metrics.metric(
                sb, "jinfer_" + task + "_requests_completed_total", "counter", requests.get());
        Metrics.metric(
                sb,
                "jinfer_"
                        + task
                        + (task.equals("embedding") ? "_inputs_total" : "_documents_total"),
                "counter",
                items.get());
        Metrics.metric(sb, "jinfer_prompt_tokens_total", "counter", promptTokens.get());
        Metrics.metric(sb, "jinfer_" + task + "_requests_in_flight", "gauge", transport.inFlight());
        return sb.toString();
    }

    private Running serveEmbeddings(Embedder embedder) {
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
        transport.probes(servedModel, modelCard(), () -> props, () -> exposition("embedding"));
        transport.work("/v1/embeddings", exchange -> embeddings(exchange, embedder));
        return new Running(
                transport.start(
                        "POST /v1/embeddings",
                        "jinfer-embedding-handler",
                        closing(embedder::close)));
    }

    private Running serveReranking(Ranker ranker) {
        Map<String, Object> props =
                JsonCodec.object(
                        "model", servedModel, "task", "rerank", "n_ctx", ranker.contextCapacity());
        transport.probes(servedModel, modelCard(), () -> props, () -> exposition("rerank"));
        for (String path : List.of("/v1/rerank", "/rerank", "/v1/reranking", "/reranking"))
            transport.work(path, exchange -> rerank(exchange, ranker));
        return new Running(
                transport.start(
                        "POST /v1/rerank", "jinfer-rerank-handler", closing(ranker::close)));
    }

    /** The state closes after the request computing on it: the fair lock waits it out. */
    private Runnable closing(Runnable close) {
        return () -> {
            compute.lock();
            try {
                close.run();
            } finally {
                compute.unlock();
            }
        };
    }

    // ---- /v1/embeddings ----------------------------------------------------------------------

    /**
     * A validated embeddings request: the texts as the model will see them, and the reply shape.
     */
    record EmbeddingRequest(List<String> texts, int dimension, boolean base64, String model) {}

    static EmbeddingRequest embeddingRequest(
            Map<String, Object> request, Embedder embedder, String servedModel) {
        Validation.validateModel(request, servedModel);
        List<String> inputs = inputs(request.get("input"));
        String prefix =
                switch (Values.stringValue(request.get("input_type"), "")) {
                    case "" -> "";
                    case "query" -> embedder.queryPrefix();
                    case "document" -> embedder.documentPrefix();
                    default ->
                            throw new Values.InvalidParam(
                                    "input_type",
                                    "Invalid argument: input_type must be query or document; got "
                                            + JsonCodec.stringify(request.get("input_type")));
                };
        String format = Values.stringValue(request.get("encoding_format"), "float");
        Validation.requireParam(
                format.equals("float") || format.equals("base64"),
                "encoding_format",
                "Invalid argument: encoding_format must be float or base64; got '" + format + "'");
        int dimension = embedder.dimension();
        if (request.get("dimensions") != null) {
            int requested = Values.intValue(request.get("dimensions"), "dimensions", dimension);
            if (embedder.minimumDimension() == embedder.dimension()) {
                Validation.requireParam(
                        requested == dimension,
                        "dimensions",
                        "Invalid argument: this model's embeddings have a fixed "
                                + dimension
                                + " dimensions and cannot be shortened; omit dimensions");
            } else {
                Validation.requireParam(
                        embedder.minimumDimension() <= requested && requested <= dimension,
                        "dimensions",
                        "Invalid argument: dimensions must be within ["
                                + embedder.minimumDimension()
                                + ", "
                                + dimension
                                + "]; got "
                                + requested);
            }
            dimension = requested;
        }
        List<String> texts = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            String text = prefix + inputs.get(i);
            int tokens = embedder.tokens(text);
            if (tokens > embedder.contextCapacity()) {
                throw new Values.InvalidParam(
                        "input",
                        "Invalid argument: "
                                + (request.get("input") instanceof String
                                        ? "input"
                                        : "input[" + i + "]")
                                + " is "
                                + tokens
                                + " tokens, over this server's "
                                + embedder.contextCapacity()
                                + "-token context; split it");
            }
            texts.add(text);
        }
        return new EmbeddingRequest(
                texts, dimension, format.equals("base64"), Requests.modelId(request, servedModel));
    }

    /** OpenAI's {@code input}: one string, or an array of them; token arrays are not taken. */
    private static List<String> inputs(Object input) {
        Validation.requireParam(input != null, "input", "Invalid argument: input is required");
        if (input instanceof String text) {
            Validation.requireParam(
                    !text.isEmpty(), "input", "Invalid argument: input must not be empty");
            return List.of(text);
        }
        Validation.requireParam(
                input instanceof List<?>,
                "input",
                "Invalid argument: input must be a string or an array of strings");
        List<?> items = (List<?>) input;
        Validation.requireParam(
                !items.isEmpty(), "input", "Invalid argument: input must not be an empty array");
        Validation.requireParam(
                items.size() <= MAX_INPUTS,
                "input",
                "Invalid argument: input holds "
                        + items.size()
                        + " items; at most "
                        + MAX_INPUTS
                        + " per request");
        List<String> texts = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Object item = items.get(i);
            if (item instanceof Number || item instanceof List<?>) {
                throw new Values.InvalidParam(
                        "input",
                        "Invalid argument: token-id input is not supported; send the text");
            }
            Validation.requireParam(
                    item instanceof String,
                    "input",
                    "Invalid argument: input[" + i + "] must be a string");
            Validation.requireParam(
                    !((String) item).isEmpty(),
                    "input",
                    "Invalid argument: input[" + i + "] must not be empty");
            texts.add((String) item);
        }
        return texts;
    }

    private void embeddings(HttpExchange exchange, Embedder embedder) throws IOException {
        Map<String, Object> body = Http.readJsonObject(exchange, config.limits());
        if (body == null) return;
        EmbeddingRequest request = embeddingRequest(body, embedder, servedModel);
        List<float[]> vectors = new ArrayList<>(request.texts().size());
        int tokens;
        compute.lock();
        try {
            tokens = embedder.embed(request.texts(), request.dimension(), vectors::add);
        } finally {
            compute.unlock();
        }
        requests.incrementAndGet();
        items.addAndGet(vectors.size());
        promptTokens.addAndGet(tokens);
        Http.sendJson(exchange, 200, embeddingResponse(request, vectors, tokens));
    }

    static Map<String, Object> embeddingResponse(
            EmbeddingRequest request, List<float[]> vectors, int tokens) {
        List<Object> data = new ArrayList<>(vectors.size());
        for (int i = 0; i < vectors.size(); i++) {
            float[] vector = vectors.get(i);
            data.add(
                    JsonCodec.object(
                            "object",
                            "embedding",
                            "index",
                            i,
                            "embedding",
                            request.base64() ? base64(vector) : floats(vector)));
        }
        return JsonCodec.object(
                "object",
                "list",
                "data",
                data,
                "model",
                request.model(),
                "usage",
                JsonCodec.object("prompt_tokens", tokens, "total_tokens", tokens));
    }

    /** OpenAI's base64 encoding: the vector's float32 values, little-endian. */
    static String base64(float[] vector) {
        ByteBuffer bytes = ByteBuffer.allocate(vector.length * Float.BYTES);
        bytes.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(vector);
        return Base64.getEncoder().encodeToString(bytes.array());
    }

    private static List<Float> floats(float[] vector) {
        Float[] boxed = new Float[vector.length];
        for (int i = 0; i < vector.length; i++) boxed[i] = vector[i];
        return Arrays.asList(boxed);
    }

    // ---- /v1/rerank --------------------------------------------------------------------------

    /** A validated rerank request. */
    record RerankRequest(
            String query,
            List<String> documents,
            int topN,
            boolean returnDocuments,
            String model) {}

    static RerankRequest rerankRequest(Map<String, Object> request, String servedModel) {
        Validation.validateModel(request, servedModel);
        Object query = request.get("query");
        Validation.requireParam(
                query instanceof String text && !text.isBlank(),
                "query",
                "Invalid argument: query must be a non-empty string");
        Validation.requireParam(
                request.get("documents") != null,
                "documents",
                "Invalid argument: documents is required");
        Validation.requireParam(
                request.get("documents") instanceof List<?>,
                "documents",
                "Invalid argument: documents must be an array of strings");
        List<?> items = (List<?>) request.get("documents");
        List<String> documents = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Object item = items.get(i);
            // Jina and Cohere also take {"text": ...}, the shape they return
            if (item instanceof Map<?, ?> object) item = object.get("text");
            Validation.requireParam(
                    item instanceof String,
                    "documents",
                    "Invalid argument: documents["
                            + i
                            + "] must be a string or {\"text\": string}");
            documents.add((String) item);
        }
        int topN = documents.size();
        if (request.get("top_n") != null) {
            int requested = Values.intValue(request.get("top_n"), "top_n", topN);
            Validation.requireParam(
                    requested >= 1, "top_n", "Invalid argument: top_n must be at least 1");
            topN = Math.min(requested, documents.size());
        }
        boolean returnDocuments =
                Values.booleanValue(request.get("return_documents"), "return_documents", false);
        return new RerankRequest(
                (String) query,
                documents,
                topN,
                returnDocuments,
                Requests.modelId(request, servedModel));
    }

    private void rerank(HttpExchange exchange, Ranker ranker) throws IOException {
        Map<String, Object> body = Http.readJsonObject(exchange, config.limits());
        if (body == null) return;
        RerankRequest request = rerankRequest(body, servedModel);
        double[] scores = new double[request.documents().size()];
        int[] at = {0};
        int tokens;
        compute.lock();
        try {
            tokens = ranker.score(request.query(), request.documents(), s -> scores[at[0]++] = s);
        } catch (IllegalArgumentException tooLong) {
            // the recipe names the document and its length; the remedy it suggests is a Java one
            String message = Http.errorMessage(tooLong);
            int remedy = message.indexOf(" - raise");
            throw new Values.InvalidParam(
                    "documents",
                    "Invalid argument: "
                            + (remedy < 0 ? message : message.substring(0, remedy) + "; split it"));
        } finally {
            compute.unlock();
        }
        requests.incrementAndGet();
        items.addAndGet(scores.length);
        promptTokens.addAndGet(tokens);
        Http.sendJson(exchange, 200, rerankResponse(request, scores, tokens));
    }

    /** Most relevant first (ties keep input order), cut to {@code top_n}. */
    static Map<String, Object> rerankResponse(RerankRequest request, double[] scores, int tokens) {
        List<Integer> order = new ArrayList<>(scores.length);
        for (int i = 0; i < scores.length; i++) order.add(i);
        order.sort(Comparator.comparingDouble((Integer i) -> scores[i]).reversed());
        List<Object> results = new ArrayList<>(request.topN());
        for (int index : order.subList(0, request.topN())) {
            Map<String, Object> result =
                    JsonCodec.object("index", index, "relevance_score", scores[index]);
            if (request.returnDocuments())
                result.put("document", JsonCodec.object("text", request.documents().get(index)));
            results.add(result);
        }
        return JsonCodec.object(
                "id",
                "rerank-" + Long.toUnsignedString(System.nanoTime(), 36),
                "model",
                request.model(),
                "results",
                results,
                "usage",
                JsonCodec.object("prompt_tokens", tokens, "total_tokens", tokens));
    }

    // ---- adapters over the loaded models -----------------------------------------------------

    private static <S extends ContextState> Embedder adapt(
            LoadedEmbedder<S> loaded, int contextCapacity) {
        S state = loaded.model().newState(contextCapacity);
        int framing = loaded.prefixTokens().length() + loaded.suffixTokens().length();
        return new Embedder() {
            public int dimension() {
                return loaded.dimension();
            }

            public int minimumDimension() {
                return loaded.minimumDimension();
            }

            public String queryPrefix() {
                return loaded.queryPrefix();
            }

            public String documentPrefix() {
                return loaded.documentPrefix();
            }

            public int contextCapacity() {
                return contextCapacity;
            }

            public int tokens(String text) {
                return framing + loaded.tokenizer().encode(text).length();
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
