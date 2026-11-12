package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel.Choice;
import com.qxotic.jinfer.chat.DecisionModel.Option;
import com.qxotic.jinfer.chat.DecisionModel.Predicate;
import com.qxotic.jinfer.chat.DecisionModel.Rating;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.hub.ModelStore;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.decision.DecisionModel;
import dev.langchain4j.model.decision.listener.DecisionModelListener;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.ContentsInput;
import dev.langchain4j.model.decision.request.DecisionInput;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.DecisionRequestParameters;
import dev.langchain4j.model.decision.request.FieldsInput;
import dev.langchain4j.model.decision.request.Question;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.TextInput;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import dev.langchain4j.model.decision.response.ChoiceAnswer;
import dev.langchain4j.model.decision.response.DecisionAnswer;
import dev.langchain4j.model.decision.response.DecisionResponse;
import dev.langchain4j.model.decision.response.ScaleAnswer;
import dev.langchain4j.model.decision.response.YesNoAnswer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

/**
 * langchain4j {@link DecisionModel} backed by jinfer: a local decision GGUF (e.g. Liquid's d1)
 * answering yes/no, choice and scale questions with probabilities, in process, nothing generated.
 * The family's prompt, the tokens read for each option and the calibration live in its port - this
 * class maps types and owns the pipeline.
 *
 * <p>The input is a text, fields - put to the model as JSON, a member a line - or contents: text,
 * images, audio and video in their order, the media read through the model's towers, its {@code
 * media} companion. A field holding contents is its text (or its media's kinds); its media join the
 * input's, never the JSON. A conversation is not an input: {@code inputMessages} is refused.
 *
 * <p>The questions of a request are answered in its order, the model's own way unless {@link
 * Builder#independent} asks for each alone.
 *
 * <p>Concurrency contract as everywhere: an instance is ONE serial pipeline over one reusable
 * state; for parallel pipelines build several instances - weights are shared via the OS page cache.
 */
public final class JinferDecisionModel implements DecisionModel, AutoCloseable {

    private final com.qxotic.jinfer.chat.DecisionModel model;
    private final ContextState state;
    private final Arena arena;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine
    private final boolean independent;
    private final VideoSampler videoSampler;
    private final String modelName; // the GGUF's file name; null for a model loaded by the caller
    private final List<DecisionModelListener> listeners;

    private JinferDecisionModel(Builder b) {
        // ONE private arena: weights land in it only when THIS instance loads them
        this.arena = Arenas.newCrossThread();
        try {
            try {
                this.model =
                        b.loaded != null
                                ? b.loaded
                                : Models.loadDecisionModel(b.modelPath, arena, b.companionPaths);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.state = model.newState(b.contextCapacity);
            this.modelName = b.modelPath == null ? null : b.modelPath.getFileName().toString();
            this.independent = b.independent;
            this.videoSampler = b.videoSampler;
            this.listeners = b.listeners;
        } catch (RuntimeException | Error e) {
            Arenas.close(arena); // free before failing
            throw e;
        }
    }

    /**
     * Blocking, idempotent: waits out an in-flight request (the fair lock), then frees the state
     * and the arena holding the weights this instance loaded; later calls fail with
     * IllegalStateException.
     */
    @Override
    public void close() {
        lock.lock();
        try {
            JinferEmbeddingModel.closeStateThenArena(state, arena);
        } finally {
            lock.unlock();
        }
    }

    /** The listeners core's {@code decide} dispatches onRequest/onResponse/onError to. */
    @Override
    public List<DecisionModelListener> listeners() {
        return listeners;
    }

    /**
     * The model this instance answers as: a request naming it is answered, a request naming another
     * model is refused. A caller-loaded model names none and accepts any.
     */
    @Override
    public DecisionRequestParameters defaultRequestParameters() {
        return modelName == null
                ? DecisionRequestParameters.EMPTY
                : DecisionRequestParameters.builder().modelName(modelName).build();
    }

    @Override
    public DecisionResponse doDecide(DecisionRequest request) {
        rejectModelSwitch(request);
        List<Content> input = input(request.input());
        Map<String, Question> asked = request.questions();
        List<com.qxotic.jinfer.chat.DecisionModel.Question> questions =
                asked.values().stream().map(JinferDecisionModel::question).toList();

        float[][] probabilities;
        lock.lock();
        try {
            probabilities =
                    independent
                            ? model.decideIndependently(state, input, questions)
                            : model.decide(state, input, questions);
        } finally {
            lock.unlock();
        }
        Map<String, DecisionAnswer> answers = new LinkedHashMap<>();
        int i = 0;
        for (Map.Entry<String, Question> entry : asked.entrySet())
            answers.put(
                    entry.getKey(),
                    answer(entry.getValue(), row(questions.get(i), probabilities[i++])));
        return DecisionResponse.builder().answers(answers).modelName(modelName).build();
    }

    /** A port that breaks the contract (a probability per option) fails here, loudly. */
    private static float[] row(
            com.qxotic.jinfer.chat.DecisionModel.Question question, float[] row) {
        int options = question.options().size();
        if (row.length != options)
            throw new IllegalStateException(
                    "the model answered "
                            + row.length
                            + " probabilities for "
                            + options
                            + " options");
        return row;
    }

    /** One loaded GGUF a pipeline: a request naming another model is refused, never answered. */
    private void rejectModelSwitch(DecisionRequest request) {
        if (modelName != null
                && request.modelName() != null
                && !request.modelName().equals(modelName))
            throw new UnsupportedFeatureException(
                    "per-request modelName is not supported: this model IS '"
                            + modelName
                            + "' (one loaded GGUF per instance)");
    }

    private List<Content> input(DecisionInput input) {
        return switch (input) {
            case TextInput text -> List.of(new Content.Text(text.text()));
            case FieldsInput fields -> {
                // a field's contents are MEDIA, never JSON text (FieldsInput's own contract): their
                // media join the input's, ahead of the JSON their text stays in
                List<Content> parts = new ArrayList<>();
                Map<String, Object> plain = new LinkedHashMap<>();
                fields.fields()
                        .forEach(
                                (name, value) -> {
                                    if (value instanceof dev.langchain4j.data.message.Content c)
                                        plain.put(name, text(List.of(c), parts));
                                    else if (isContentList(value))
                                        plain.put(name, text((List<?>) value, parts));
                                    else plain.put(name, value);
                                });
                parts.add(new Content.Text(JsonCodec.stringify(plain, true)));
                yield parts;
            }
            case ContentsInput contents -> Mappings.parts(contents.contents(), videoSampler);
            default ->
                    throw new UnsupportedFeatureException(
                            "a jinfer decision model judges a text, fields or contents, not "
                                    + input.type().name().toLowerCase(Locale.ROOT)
                                    + ": put what is to be judged as input(...),"
                                    + " inputFields(...) or inputContents(...)");
        };
    }

    /** A field's list of contents ({@link FieldsInput} allows one), never a plain list. */
    private static boolean isContentList(Object value) {
        return value instanceof List<?> list
                && !list.isEmpty()
                && list.getFirst() instanceof dev.langchain4j.data.message.Content;
    }

    /**
     * A field's contents as its value: its text joined, or its media's kinds when it holds no text.
     * The media go to {@code media}.
     */
    private String text(List<?> contents, List<Content> media) {
        StringBuilder text = new StringBuilder(), kinds = new StringBuilder();
        boolean written = false;
        for (Object item : contents) {
            // FieldsInput keeps a content list's items langchain4j contents
            Content part = Mappings.part((dev.langchain4j.data.message.Content) item, videoSampler);
            if (part instanceof Content.Media m) {
                media.add(m);
                kinds.append(kinds.isEmpty() ? "" : ", ")
                        .append(m.media().getClass().getSimpleName().toLowerCase(Locale.ROOT));
            } else {
                text.append(((Content.Text) part).text());
                written = true;
            }
        }
        return written ? text.toString() : kinds.toString();
    }

    private static com.qxotic.jinfer.chat.DecisionModel.Question question(Question question) {
        return switch (question) {
            case YesNoQuestion q -> new Predicate(q.text(), q.yesWhen(), q.noWhen());
            case ChoiceQuestion q -> {
                List<Option> options = new ArrayList<>();
                // an option added without a description carries its name as one
                q.options()
                        .forEach(
                                (name, description) ->
                                        options.add(
                                                new Option(
                                                        name,
                                                        name.equals(description)
                                                                ? null
                                                                : description)));
                yield new Choice(q.text(), options);
            }
            case ScaleQuestion q -> {
                List<Option> levels = new ArrayList<>();
                for (int i = 0; i < q.levels().size(); i++)
                    levels.add(new Option(q.levels().get(i), q.levelDescriptions().get(i)));
                yield new Rating(q.text(), levels);
            }
            default ->
                    throw new UnsupportedFeatureException(
                            question.getClass().getSimpleName() + " is not supported");
        };
    }

    private static DecisionAnswer answer(Question question, float[] p) {
        return switch (question) {
            case YesNoQuestion q -> YesNoAnswer.from(p[1]);
            case ChoiceQuestion q -> {
                Map<String, Double> probabilities = new LinkedHashMap<>();
                int at = 0;
                for (String name : q.options().keySet()) probabilities.put(name, (double) p[at++]);
                // the likeliest, the first of a tie
                String best =
                        Collections.max(probabilities.entrySet(), Map.Entry.comparingByValue())
                                .getKey();
                yield ChoiceAnswer.builder().value(best).probabilities(probabilities).build();
            }
            default -> {
                List<Double> probabilities = new ArrayList<>();
                double mean = 0;
                for (int i = 0; i < p.length; i++) {
                    probabilities.add((double) p[i]);
                    mean += i * (double) p[i];
                }
                // rounding can carry the mean a hair past the top level
                yield ScaleAnswer.builder()
                        .mean(Math.min(mean, p.length - 1))
                        .probabilities(probabilities)
                        .build();
            }
        };
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Object source; // Path | model-ref String | DecisionModel: the last setter wins
        private Path modelPath; // derived from source at build()
        private com.qxotic.jinfer.chat.DecisionModel loaded; // derived from source at build()
        private Map<String, Path> companionPaths; // resolved at build()
        private final Map<String, Path> localCompanions = new LinkedHashMap<>();
        private final Map<String, String> companionRefs = new LinkedHashMap<>();
        private int contextCapacity; // 0: the default
        private boolean independent;
        private VideoSampler videoSampler = VideoSampler.UNIFORM;
        private List<DecisionModelListener> listeners = List.of();

        /** The decision GGUF to load. Required. */
        public Builder modelPath(Path modelPath) {
            this.source = Objects.requireNonNull(modelPath, "modelPath");
            return this;
        }

        /**
         * The model as a model ref, resolved - downloading to the local cache on first use - by
         * {@link #build()}.
         *
         * <pre>{@code
         * model("LiquidAI/d1-2B-GGUF:Q8_0");
         * }</pre>
         *
         * <p>The full grammar is documented once in {@link com.qxotic.jinfer.hub.ModelRef}. For a
         * file already on disk use {@link #modelPath(Path)}.
         */
        public Builder model(String modelRef) {
            ModelStore.requireRef(modelRef);
            this.source = modelRef;
            return this;
        }

        /**
         * A model you loaded yourself, its companions with it. Its weights arena stays yours; close
         * the arena only after every model built on it.
         */
        public Builder model(com.qxotic.jinfer.chat.DecisionModel loaded) {
            this.source = Objects.requireNonNull(loaded, "loaded");
            return this;
        }

        /** Attaches a local companion file. This method never touches the network. */
        public Builder companionPath(String capability, Path companionPath) {
            Objects.requireNonNull(capability, "capability");
            Objects.requireNonNull(companionPath, "companionPath");
            companionRefs.remove(capability);
            localCompanions.put(capability, companionPath);
            return this;
        }

        /**
         * Attaches a companion from a supported model repository, resolved at {@link #build()}:
         * {@code "media"} is the model's mmproj, the towers it reads images, audio and video
         * through.
         */
        public Builder companion(String capability, String companionRef) {
            Objects.requireNonNull(capability, "capability");
            if (!ModelStore.isRef(companionRef)) {
                throw new IllegalArgumentException(
                        "'"
                                + companionRef
                                + "' is not a companion model ref. Use companionPath(...) for a"
                                + " local file; download plain URLs first.");
            }
            localCompanions.remove(capability);
            companionRefs.put(capability, companionRef);
            return this;
        }

        /**
         * Upper bound on an input with its questions, in tokens, a media row a token. {@code 0},
         * the default, is {@link PromptCache.Options#DEFAULT_CONTEXT_CAPACITY} (4096) or the
         * model's {@code maxContextLength} when that is smaller; {@link
         * PromptCache.Options#MAXIMUM_CONTEXT} (-1) is that length; any other value is exact, and
         * one past that length is refused at build.
         *
         * @throws IllegalArgumentException if {@code contextCapacity < -1}
         */
        public Builder contextCapacity(int contextCapacity) {
            PromptCache.Options.contextCapacity(contextCapacity, Integer.MAX_VALUE, 0);
            this.contextCapacity = contextCapacity;
            return this;
        }

        /**
         * Whether each question is answered as if asked alone ({@code decideIndependently}). Off by
         * default: the model's own way ({@code decide}), where an answer may shift with the other
         * questions asked.
         */
        public Builder independent(boolean independent) {
            this.independent = independent;
            return this;
        }

        /**
         * The listeners notified of every decision: request, response and error. Called on the
         * thread of the call; an exception they throw is logged and ignored by langchain4j.
         */
        public Builder listeners(List<DecisionModelListener> listeners) {
            this.listeners = List.copyOf(Objects.requireNonNull(listeners, "listeners"));
            return this;
        }

        /** How a video content's frames are sampled; uniform across the clip by default. */
        public Builder videoSampler(VideoSampler videoSampler) {
            this.videoSampler = Objects.requireNonNull(videoSampler);
            return this;
        }

        public JinferDecisionModel build() {
            modelPath = null;
            loaded = null;
            if (source == null)
                throw new IllegalArgumentException(
                        "a model is required: model(\"owner/repo:Q8_0\"), modelPath(...) or"
                                + " model(DecisionModel)");
            if (source instanceof com.qxotic.jinfer.chat.DecisionModel l) {
                if (!companionRefs.isEmpty() || !localCompanions.isEmpty())
                    throw new IllegalArgumentException(
                            "companions are load-time settings; apply them when you load the"
                                    + " DecisionModel passed to model(...)");
                loaded = l;
                companionPaths = Map.of();
                return new JinferDecisionModel(this);
            }
            // the model (when it is a ref) and the companions resolve in ONE batch, so a cold
            // start pays the slowest download, not the sum
            List<String> wanted = new ArrayList<>();
            if (source instanceof String ref) wanted.add(ref);
            wanted.addAll(companionRefs.values());
            List<Path> resolved = ModelStore.standard().resolveAll(wanted);
            int at = 0;
            modelPath = source instanceof Path path ? path : resolved.get(at++);
            Map<String, Path> companions = new LinkedHashMap<>(localCompanions);
            for (String capability : companionRefs.keySet())
                companions.put(capability, resolved.get(at++));
            companionPaths = Collections.unmodifiableMap(companions);
            return new JinferDecisionModel(this);
        }
    }
}
