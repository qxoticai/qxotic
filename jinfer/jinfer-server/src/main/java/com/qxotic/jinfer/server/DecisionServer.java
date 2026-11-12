package com.qxotic.jinfer.server;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.DecisionModel.Choice;
import com.qxotic.jinfer.chat.DecisionModel.Option;
import com.qxotic.jinfer.chat.DecisionModel.Predicate;
import com.qxotic.jinfer.chat.DecisionModel.Question;
import com.qxotic.jinfer.chat.DecisionModel.Rating;
import com.qxotic.jinfer.codecs.VideoCodec;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.Multimodal;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The decision transport: {@code POST /v1/systemone}, with {@code /v1/models}, {@code /health},
 * {@code /props} and {@code /metrics}, over one {@link DecisionModel}. Started by the CLI when
 * {@code server} is given a decision checkpoint.
 *
 * <p>The request and the response are the System One shape, as llama.cpp serves it:
 *
 * <pre>{@code
 * {"state": "I was charged twice.",
 *  "files": ["data:image/png;base64,..."],
 *  "questions": {
 *    "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
 *    "team":    {"type": "choice", "instructions": "Which team?",
 *                "criteria": {"billing": "Charges and refunds", "shipping": null}},
 *    "urgency": {"type": "score",  "instructions": "How urgent?",
 *                "criteria": ["Can wait", "Today", "Right now"]}}}
 * }</pre>
 *
 * {@code state} is a text, any other JSON value - put to the model as JSON, a member a line - or
 * null for media alone; {@code files} (or {@code images}) are images, audio and video as base64
 * data URIs, read ahead of the state through the model's towers. {@code "together": true}, jinfer's
 * own, has the model read the state once for all the questions, an answer then free to shift with
 * the questions beside it; otherwise each is answered as if asked alone.
 *
 * <p>Requests run one at a time - the compute pool is the process-wide one - while admission and
 * parsing stay concurrent.
 */
public final class DecisionServer {

    /**
     * The d1 reference's sampling ({@code Lfm3VlVideoProcessor}): two frames a second of the clip,
     * 4 at least and 768 at most, never more than it has, at {@code round(linspace(0, frames - 1,
     * n))} (ponytail: the only video decision family; a per-model sampler when a second arrives).
     */
    private static final VideoSampler FRAMES =
            video -> {
                VideoCodec codec = VideoCodec.ffmpeg();
                long period = codec.framePeriod(video).toNanos();
                long frames =
                        Math.max(
                                1,
                                Math.round((double) codec.totalDuration(video).toNanos() / period));
                int n =
                        (int)
                                Math.min(
                                        frames,
                                        Math.clamp(frames * period * 2 / 1_000_000_000L, 4, 768));
                double step = n > 1 ? (double) (frames - 1) / (n - 1) : 0;
                Duration[] times = new Duration[n];
                for (int i = 0; i < n; i++)
                    times[i] = Duration.ofNanos(Math.round(i * step) * period);
                return codec.framesAt(video, times);
            };

    private final DecisionModel model;
    private final ContextState state;
    private final String servedModel;
    private final ServerConfig config;
    private final ReentrantLock compute = new ReentrantLock(true);
    private final TaskTransport transport;
    private final AtomicLong requests = new AtomicLong(), answered = new AtomicLong();

    private DecisionServer(
            DecisionModel model, int contextCapacity, String servedModel, ServerConfig config)
            throws IOException {
        this.model = model;
        this.state = model.newState(contextCapacity);
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

    /**
     * Serves {@code model} over a state of {@code contextCapacity} tokens, as {@link
     * DecisionModel#newState} reads it. Does not block, prints nothing, owns no shutdown hook.
     */
    public static Running start(
            DecisionModel model, int contextCapacity, String servedModel, ServerConfig config)
            throws IOException {
        if (model == null) throw new IllegalArgumentException("model is required");
        if (config == null) throw new IllegalArgumentException("config is required");
        return new DecisionServer(model, contextCapacity, servedModel, config).serve();
    }

    private Running serve() {
        transport.probes(
                servedModel,
                MediaParts.modalities(model, true),
                () ->
                        JsonCodec.object(
                                "model",
                                servedModel,
                                "task",
                                "decision",
                                "context_capacity",
                                state.contextCapacity()),
                () -> {
                    StringBuilder sb = new StringBuilder();
                    Metrics.metric(sb, "jinfer_uptime_seconds", "gauge", transport.uptimeSeconds());
                    Metrics.metric(
                            sb,
                            "jinfer_decision_requests_completed_total",
                            "counter",
                            requests.get());
                    Metrics.metric(
                            sb,
                            "jinfer_decision_questions_answered_total",
                            "counter",
                            answered.get());
                    Metrics.metric(
                            sb,
                            "jinfer_decision_requests_in_flight",
                            "gauge",
                            transport.inFlight());
                    return sb.toString();
                },
                compute::getQueueLength);
        transport.work("/v1/systemone", this::handle);
        // the state closes after the request computing on it: the fair lock waits it out
        Runnable closing =
                () -> {
                    compute.lock();
                    try {
                        state.close();
                    } finally {
                        compute.unlock();
                    }
                };
        return new Running(
                transport.start("POST /v1/systemone", "jinfer-decision-handler", closing));
    }

    private void handle(HttpExchange exchange) throws IOException {
        Map<String, Object> request = Http.readJsonObject(exchange, config.limits());
        if (request == null) return;
        Validation.validateModel(request, servedModel);
        List<Content> input = input(request, model);
        Map<String, Object> asked = questions(request.get("questions"));
        List<String> names = new ArrayList<>(asked.keySet());
        List<Question> questions = new ArrayList<>();
        for (String name : names) questions.add(question(name, asked.get(name)));
        boolean together = Values.booleanValue(request.get("together"), "together", false);

        float[][] probabilities;
        compute.lock();
        try {
            probabilities =
                    together
                            ? model.decideTogether(state, input, questions)
                            : model.decideEach(state, input, questions);
        } catch (IllegalArgumentException tooLong) {
            // the library's remedy is for Java callers; a client shortens what it sends
            String message = Http.errorMessage(tooLong);
            int remedy = message.indexOf(": open a larger one");
            throw new Values.InvalidParam(
                    "state",
                    remedy < 0
                            ? message
                            : message.substring(0, remedy)
                                    + ": shorten the state or send smaller files");
        } finally {
            compute.unlock();
        }
        requests.incrementAndGet();
        answered.addAndGet(names.size());

        Map<String, Object> answers = new LinkedHashMap<>();
        for (int i = 0; i < names.size(); i++)
            answers.put(names.get(i), answer(questions.get(i), probabilities[i]));
        Http.sendJson(
                exchange,
                200,
                JsonCodec.object(
                        "model", Requests.modelId(request, servedModel), "answers", answers));
    }

    /** The media of {@code files}, then the state: its text, or its JSON; nothing when null. */
    private static List<Content> input(Map<String, Object> request, DecisionModel model) {
        Validation.requireParam(
                request.containsKey("state"),
                "state",
                "state must be provided: a text, a JSON value, or null with files alone");
        List<Content> input = new ArrayList<>();
        for (String field : List.of("files", "images")) {
            Object files = request.get(field);
            if (files == null) continue;
            Validation.requireParam(
                    files instanceof List<?>, field, field + " must be an array of data: URIs");
            for (Object file : (List<?>) files) input.add(media(file, field, model));
        }
        Object state = request.get("state");
        if (state instanceof String text) input.add(new Content.Text(text));
        else if (state != null)
            input.add(new Content.Text(com.qxotic.jinfer.chat.JsonCodec.stringify(state, true)));
        Validation.requireParam(
                !input.isEmpty(),
                "state",
                "state is null and there are no files: nothing to judge");
        return input;
    }

    /**
     * One file by the type its data: URI states: the URI's own form first - a client's malformed
     * payload is the client's error whatever the model reads - then the model's capability, before
     * any decode: no ffmpeg run, and the client is told what this checkpoint takes.
     */
    private static Content.Media media(Object file, String field, DecisionModel model) {
        Validation.requireParam(
                file instanceof String url && url.startsWith("data:") && url.indexOf(',') > 0,
                field,
                field
                        + " holds base64 data: URIs (data:image/..., data:audio/... or"
                        + " data:video/...); the server does not fetch remote URLs");
        String url = (String) file;
        // a client's malformed URI is refused on its own terms, before any capability check
        Validation.requireParam(
                url.substring(0, url.indexOf(',')).endsWith(";base64"),
                field,
                field + " data: URIs must be base64-encoded (data:image/png;base64,...)");
        if (url.startsWith("data:image/")) {
            require(modality(model, Media.Image.class), "image");
            return MediaParts.image(Map.of("image_url", url));
        }
        if (url.startsWith("data:audio/")) {
            require(modality(model, Media.Audio.class), "audio");
            return MediaParts.audio(
                    Map.of("input_audio", Map.of("data", url.substring(url.indexOf(',') + 1))));
        }
        if (url.startsWith("data:video/")) {
            require(modality(model, Media.Video.class), "video");
            return MediaParts.video(Map.of("video_url", url), FRAMES);
        }
        throw new Values.InvalidParam(
                field, field + " takes image, audio and video data: URIs, not " + type(url));
    }

    private static boolean modality(DecisionModel model, Class<? extends Media> kind) {
        return model instanceof Multimodal multimodal && multimodal.supports(kind);
    }

    private static void require(boolean supported, String kind) {
        if (!supported)
            throw new UnsupportedOperationException(
                    "this model takes no " + kind + ": send text, or load its media companion");
    }

    private static String type(String url) {
        int end = url.indexOf(';');
        return url.substring(0, end < 0 ? url.indexOf(',') : end);
    }

    private static Map<String, Object> questions(Object value) {
        Validation.requireParam(
                value instanceof Map<?, ?> map && !map.isEmpty(),
                "questions",
                "questions must be a non-empty object of named questions");
        return Values.asObject(value, "questions");
    }

    private static Question question(String name, Object value) {
        String at = "questions." + name;
        Validation.requireParam(value instanceof Map<?, ?>, at, at + " must be an object");
        Map<String, Object> q = Values.asObject(value, at);
        Validation.requireParam(
                q.get("instructions") instanceof String text && !text.isEmpty(),
                at + ".instructions",
                at + ".instructions must be a non-empty text");
        String text = (String) q.get("instructions");
        Object criteria = q.get("criteria");
        String type = q.get("type") instanceof String s ? s : "";
        switch (type) {
            case "noul" -> {
                Validation.requireParam(
                        criteria == null || criteria instanceof Map<?, ?>,
                        at + ".criteria",
                        at + ".criteria must be an object with \"true\" and \"false\"");
                Map<?, ?> sides = criteria == null ? Map.of() : (Map<?, ?>) criteria;
                return new Predicate(
                        text,
                        described(sides.get("true"), at + ".criteria.true"),
                        described(sides.get("false"), at + ".criteria.false"));
            }
            case "choice" -> {
                Validation.requireParam(
                        criteria instanceof Map<?, ?> map && map.size() >= 2,
                        at + ".criteria",
                        at + ".criteria must be an object of at least 2 options");
                List<Option> options = new ArrayList<>();
                ((Map<?, ?>) criteria)
                        .forEach(
                                (key, description) ->
                                        options.add(
                                                new Option(
                                                        (String) key,
                                                        described(
                                                                description,
                                                                at + ".criteria." + key))));
                return new Choice(text, options);
            }
            case "score" -> {
                Validation.requireParam(
                        criteria instanceof List<?> list && list.size() >= 2 && list.size() <= 10,
                        at + ".criteria",
                        at + ".criteria must be an array of 2 to 10 levels");
                List<Option> levels = new ArrayList<>();
                for (Object level : (List<?>) criteria) {
                    Validation.requireParam(
                            level instanceof String,
                            at + ".criteria",
                            at + ".criteria levels must be texts");
                    levels.add(new Option((String) level));
                }
                return new Rating(text, levels);
            }
            default ->
                    throw new Values.InvalidParam(
                            at + ".type", at + ".type must be noul, choice or score");
        }
    }

    private static String described(Object description, String at) {
        Validation.requireParam(
                description == null || description instanceof String,
                at,
                at + " must be a text or null");
        return (String) description;
    }

    /**
     * One answer in the System One shape, the same rules for every model: a score is the expected
     * level (the probability-weighted mean of the levels, as LangChain4j and Spring AI define it),
     * a confidence the probability of the likeliest option.
     */
    private static Map<String, Object> answer(Question question, float[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        return switch (question) {
            case Predicate q -> JsonCodec.object("type", "noul", "noul", (double) p[1]);
            case Choice q -> {
                Map<String, Object> probabilities = new LinkedHashMap<>();
                for (int i = 0; i < p.length; i++)
                    probabilities.put(q.options().get(i).label(), (double) p[i]);
                yield JsonCodec.object(
                        "type",
                        "choice",
                        "choice",
                        q.options().get(best).label(),
                        "probabilities",
                        probabilities,
                        "confidence",
                        (double) p[best]);
            }
            case Rating q -> {
                Map<String, Object> legend = new LinkedHashMap<>(),
                        probabilities = new LinkedHashMap<>();
                double mean = 0;
                for (int i = 0; i < p.length; i++) {
                    legend.put(Integer.toString(i), q.levels().get(i).label());
                    probabilities.put(Integer.toString(i), (double) p[i]);
                    mean += i * (double) p[i];
                }
                yield JsonCodec.object(
                        "type",
                        "score",
                        "score",
                        Math.min(mean, p.length - 1), // rounding can carry it a hair past the top
                        "legend",
                        legend,
                        "probabilities",
                        probabilities,
                        "confidence",
                        (double) p[best]);
            }
        };
    }
}
