package com.qxotic.jinfer.models.lfm3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Resample;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.MemorySegment;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;

/**
 * Liquid d1 and d1-Omni on LFM3 ({@code lfm3.decision.type} of {@code lfm3-d1} and {@code
 * lfm3-d1-omni}): an input and typed questions in, a probability per option out, nothing generated.
 * The checkpoints were trained on one prompt layout and one readout, both reproduced here from
 * their reference ({@code prompt.py}, {@code media.py} and {@code attention.py} of LiquidAI/d1-2B,
 * the GGUF's {@code tokenizer.chat_template.systemone}):
 *
 * <pre>
 *   &lt;|startoftext|&gt;&lt;|im_start|&gt;user
 *   {the input's media}{its text}
 *
 *
 *   QUESTION:
 *   {the question, then the options and the reply rule of its kind}&lt;|im_end|&gt;
 *   &lt;|im_start|&gt;assistant
 *   &lt;think&gt;&lt;/think&gt;
 * </pre>
 *
 * The media head the turn in their order, whatever their place among the input's text parts, each
 * the rows of its tower between {@code <|vision_start|>} and {@code <|vision_end|>}: a picture, no
 * larger than {@link #MAX_PIXELS}; a clip of sound, on d1-Omni; a video, as the frames it is given,
 * two a temporal patch behind the time they were taken at. The towers come with the model's mmproj
 * ({@link Lfm3#withMedia}). The text parts follow, joined; an input without any is its media alone,
 * and the question follows them directly, without the {@code QUESTION:} line.
 *
 * <p>A prompt is read whole, both ways by the full-attention layers ({@link
 * Lfm3.Configuration#causalAttention} false). At its last row each option scores the best logit
 * among its label tokens: a predicate's {@code no|No|NO} and {@code yes|Yes}, a choice's code
 * ({@code A}, {@code B}, .., with and without a leading space), a rating's digit. The scores,
 * divided by the checkpoint's temperature for the kind, softmax into the answer.
 *
 * <p>Questions asked together are packed after one copy of the input, each behind its own {@code
 * QUESTION:} line and read at its own last row, {@link #GROUP} a prompt: the input's rows see every
 * question of the prompt and a question's rows the input and themselves, which is how the
 * checkpoint was trained to share one reading of an input. Several prompts run as one batch.
 */
final class Lfm3Decision implements DecisionModel, Multimodal {

    /**
     * The most pixels of a picture the checkpoints were trained on ({@code d1_vision_max_pixels} of
     * their configuration, which the GGUF does not carry): 1,600 rows.
     */
    private static final long MAX_PIXELS = 1_638_400;

    /** jinfer's default context capacity, on every face. */
    private static final int DEFAULT_CAPACITY = 4096;

    /**
     * The questions a packed prompt holds: the most this checkpoint was trained to read at once.
     */
    static final int GROUP = 16;

    /**
     * The tokens up to which prompts share a batch. Short prompts run faster side by side (16 of 56
     * tokens: 1.5x); past about this many tokens a batch gains nothing over a prompt at a time
     * (measured 0.9x at 523 and at 2005 tokens a prompt), so a longer prompt runs alone.
     */
    private static final int SHARED_BATCH = 1024;

    /** What opens a question after the input, and each further question of a packed prompt. */
    private static final String HEADER = "QUESTION:\n";

    /** Zero-width space: after a control token's first character, the text is no longer it. */
    private static final String BREAK = Character.toString(0x200B);

    private final Lfm3 model;
    private final Tokenizer tokenizer;
    private final int bos, imStart, imEnd, think, endThink, visionOpen, visionClose;
    private final float predicateTemperature, choiceTemperature, ratingTemperature;

    /** The tokenizer's control tokens as text, longest first: what {@link #plain} breaks. */
    private final List<String> controls;

    /** Matches the added tokens left in the caller's text, as the reference tokenizer does. */
    private final Specials specials;

    Lfm3Decision(Lfm3 model, GGUF gguf) {
        String type = gguf.getStringOrDefault("lfm3.decision.type", "");
        if (!(type.equals("lfm3-d1") || type.equals("lfm3-d1-omni"))
                || model.configuration().causalAttention())
            throw new IllegalArgumentException(
                    "LFM3: not a d1 decision checkpoint (lfm3.decision.type = '" + type + "')");
        this.model = model;
        this.tokenizer = model.tokenizer();
        bos = SpecialTokens.require(tokenizer, "<|startoftext|>");
        imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        think = SpecialTokens.require(tokenizer, "<think>");
        endThink = SpecialTokens.require(tokenizer, "</think>");
        visionOpen = SpecialTokens.require(tokenizer, "<|vision_start|>");
        visionClose = SpecialTokens.require(tokenizer, "<|vision_end|>");
        // the file's names for the three kinds: noul (yes/no), choice, score (a rating)
        predicateTemperature = temperature(gguf, "noul");
        choiceTemperature = temperature(gguf, "choice");
        ratingTemperature = temperature(gguf, "score");
        Vocabulary vocabulary = tokenizer.vocabulary();
        List<String> names = new ArrayList<>();
        for (int id = 0; id < vocabulary.size(); id++)
            if (vocabulary.isTokenOfType(id, StandardTokenType.CONTROL)
                    && vocabulary.token(id).length() > 1) names.add(vocabulary.token(id));
        names.sort(Comparator.comparingInt(String::length).reversed());
        controls = List.copyOf(names);
        specials = SpecialTokens.encoder(tokenizer);
    }

    private static float temperature(GGUF gguf, String kind) {
        return gguf.getValueOrDefault(float.class, "lfm3.decision.temperature." + kind, 1f);
    }

    /** The towers of the model's mmproj: what an input's media can be. */
    @Override
    public <R extends Media> Optional<MediaProjector<R>> projector(Class<R> modality) {
        return model.projector(modality);
    }

    /**
     * Video included, when the vision tower that reads its frames is attached: the decision reads a
     * clip's frames as pictures, which the chat template of this family does not.
     */
    @Override
    public boolean supports(Class<? extends Media> modality) {
        return modality == Media.Video.class
                ? model.vision() != null
                : Multimodal.super.supports(modality);
    }

    /** A whole prompt is one batch, so the state's batch is as large as its context. */
    @Override
    public Lfm3.State newState(int contextCapacity) {
        int most = model.configuration().maxContextLength();
        if (contextCapacity < -1 || contextCapacity > most)
            throw new IllegalArgumentException(
                    "contextCapacity "
                            + contextCapacity
                            + ": the model reads at most "
                            + most
                            + " tokens; 0 is the default, -1 the maximum");
        int capacity =
                contextCapacity == -1
                        ? most
                        : contextCapacity == 0 ? Math.min(DEFAULT_CAPACITY, most) : contextCapacity;
        return model.newState(capacity, capacity);
    }

    // --- the three ways to ask ---

    @Override
    public float[] decide(ContextState state, List<Content> input, Question question) {
        return decideEach(state, input, List.of(question))[0];
    }

    /** A prompt a question, short prompts side by side in a batch. */
    @Override
    public float[][] decideEach(
            ContextState state, List<Content> input, List<? extends Question> questions) {
        return run(own(state), judged(input), questions, 1);
    }

    /**
     * The questions packed {@link #GROUP} a prompt: the input is read once a prompt, with every
     * question of the prompt in view, so an answer can shift with the questions asked beside it and
     * with their order. A question alone in its prompt is answered as {@link #decide} answers it.
     */
    @Override
    public float[][] decideTogether(
            ContextState state, List<Content> input, List<? extends Question> questions) {
        return run(own(state), judged(input), questions, GROUP);
    }

    /**
     * A question as the checkpoint takes it: its text, each option's label tokens, its temperature.
     */
    private record Asked(String body, int[][] labels, float temperature) {}

    private Asked asked(Question question) {
        return switch (question) {
            case Predicate q -> {
                int[][] labels = {singles("no", "No", "NO"), singles("yes", "Yes", "YES")};
                if (labels[0].length == 0 || labels[1].length == 0)
                    throw new IllegalStateException(
                            "the tokenizer holds no single-token yes and no");
                yield new Asked(predicateBody(q), labels, predicateTemperature);
            }
            case Choice q -> {
                int[][] labels = new int[q.options().size()][];
                yield new Asked(
                        choiceBody(q, codes(q.options(), labels)), labels, choiceTemperature);
            }
            case Rating q -> {
                String body = ratingBody(q); // refuses more levels than digits
                int[][] labels = new int[q.levels().size()][];
                for (int i = 0; i < labels.length; i++) labels[i] = singles(Integer.toString(i));
                yield new Asked(body, labels, ratingTemperature);
            }
        };
    }

    /**
     * The questions as prompts of up to {@code group} each, short prompts sharing a batch, every
     * answer read at its question's last row. The input's media are encoded once, for every prompt.
     */
    private float[][] run(
            Lfm3.State state, Judged judged, List<? extends Question> questions, int group) {
        Asked[] asked = new Asked[questions.size()];
        for (int i = 0; i < asked.length; i++) asked[i] = asked(questions.get(i));
        float[][] answers = new float[asked.length][];
        // a question of an input without text follows its media directly, header and all
        String header = judged.text() == null ? "" : HEADER;

        MemoryArena<MemorySegment> scratch =
                judged.media().isEmpty() ? null : Arenas.newCrossThreadMemoryArena();
        try {
            Head head = head(judged, scratch, state.batchCapacity());
            List<int[]> tokens = new ArrayList<>(), parts = new ArrayList<>();
            int first = 0, held = 0; // the batch being filled: its first question, its rows
            for (int from = 0; from < asked.length; from += group) {
                int count = Math.min(group, asked.length - from);
                // the first question as its own prompt, so the shared part is what precedes its
                // text; each further one behind its own header
                int[][] pieces = new int[count][];
                pieces[0] = prompt(head, judged.text(), asked[from].body());
                int[] ends = new int[count + 1];
                ends[0] = pieces[0].length - part("", asked[from].body()).length;
                ends[1] = pieces[0].length;
                for (int k = 1; k < count; k++) {
                    pieces[k] = part(header, asked[from + k].body());
                    ends[k + 1] = ends[k] + pieces[k].length;
                }
                int length = ends[count];
                if (length > state.batchCapacity())
                    throw new IllegalArgumentException(
                            "the input and its questions take "
                                    + length
                                    + " tokens"
                                    + (head.rows() == 0
                                            ? ""
                                            : " (" + head.rows() + " of them its media)")
                                    + ", the state holds "
                                    + state.batchCapacity()
                                    + ": open a larger one with newState");
                if (held > 0 && held + length > Math.min(SHARED_BATCH, state.batchCapacity())) {
                    answer(state, head, tokens, parts, asked, first, answers);
                    tokens.clear();
                    parts.clear();
                    first = from;
                    held = 0;
                }
                int[] prompt = new int[length];
                for (int k = 0, at = 0; k < count; at += pieces[k++].length)
                    System.arraycopy(pieces[k], 0, prompt, at, pieces[k].length);
                tokens.add(prompt);
                // a lone question is a plain prompt: one part, every row sees every row
                parts.add(count == 1 ? new int[] {length} : ends);
                held += length;
            }
            if (!tokens.isEmpty()) answer(state, head, tokens, parts, asked, first, answers);
            return answers;
        } finally {
            if (scratch != null) Arenas.close(scratch);
        }
    }

    /** One batch: its prompts through the model, then each question's answer from its last row. */
    private void answer(
            Lfm3.State state,
            Head head,
            List<int[]> prompts,
            List<int[]> parts,
            Asked[] asked,
            int question,
            float[][] answers) {
        int total = 0;
        for (int[] prompt : prompts) total += prompt.length;
        int[] batch = new int[total];
        List<Lfm3.Placed> media = new ArrayList<>();
        int at = 0;
        for (int[] prompt : prompts) {
            System.arraycopy(prompt, 0, batch, at, prompt.length);
            for (Lfm3.Placed placed : head.media())
                media.add(new Lfm3.Placed(at + placed.at(), placed.rows(), placed.count()));
            at += prompt.length;
        }
        state.reset();
        model.ingestPrompts(state, batch, parts.toArray(int[][]::new), media);
        int base = 0;
        for (int[] ends : parts) {
            // a lone question's prompt is its one part; a packed one's questions end at ends[1..]
            for (int k = ends.length == 1 ? 0 : 1; k < ends.length; k++, question++)
                answers[question] = probabilities(state, base + ends[k] - 1, asked[question]);
            base += ends[ends.length - 1];
        }
    }

    private static Lfm3.State own(ContextState state) {
        if (state instanceof Lfm3.State own) return own;
        throw new IllegalArgumentException("the state is not one of this model's newState");
    }

    /**
     * At retained row {@code output} each option scores its best label logit, and the scores over
     * the temperature softmax into the answer. Only the label tokens' logits are computed: the
     * vocabulary's other 128,000 are never read.
     */
    private float[] probabilities(Lfm3.State state, int output, Asked asked) {
        int[][] labels = asked.labels();
        int count = 0;
        for (int[] forms : labels) count += forms.length;
        int[] tokens = new int[count];
        for (int i = 0, at = 0; i < labels.length; at += labels[i++].length)
            System.arraycopy(labels[i], 0, tokens, at, labels[i].length);
        float[] logits = new float[count];
        model.logits(state, output, tokens, logits);
        float[] p = new float[labels.length];
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, at = 0; i < p.length; i++) {
            float best = Float.NEGATIVE_INFINITY;
            for (int form = 0; form < labels[i].length; form++) best = Math.max(best, logits[at++]);
            p[i] = best / asked.temperature();
            max = Math.max(max, p[i]);
        }
        float sum = 0;
        for (int i = 0; i < p.length; i++) sum += p[i] = (float) Math.exp(p[i] - max);
        for (int i = 0; i < p.length; i++) p[i] /= sum;
        return p;
    }

    // --- the prompt: each kind's question as the checkpoint read it in training ---

    /**
     * What is judged, as the checkpoint reads it: its media in their order, then its text - the
     * text parts joined in theirs, null when it has none, which is not an empty text.
     */
    private record Judged(List<Media> media, String text) {}

    private static Judged judged(List<Content> input) {
        if (input.isEmpty()) throw new IllegalArgumentException("an empty input: nothing to judge");
        List<Media> media = new ArrayList<>();
        StringBuilder text = null;
        for (Content part : input)
            switch (part) {
                case Content.Text t ->
                        (text == null ? text = new StringBuilder() : text).append(t.text());
                case Content.Media m -> media.add(m.media());
                default ->
                        throw new IllegalArgumentException(
                                "a decision's input is text and media, not "
                                        + part.getClass().getSimpleName());
            }
        return new Judged(media, text == null ? null : text.toString());
    }

    /**
     * A prompt's opening when its input has media: the tokens through the last of them, a
     * placeholder a media row, and the rows to put there. Without media, {@link #NONE}.
     */
    private record Head(int[] tokens, List<Lfm3.Placed> media) {
        int rows() {
            return media.stream().mapToInt(Lfm3.Placed::count).sum();
        }
    }

    private static final Head NONE = new Head(null, List.of());

    /**
     * The media at the head of the user turn, as the checkpoint's template and its processor write
     * them: a picture or a clip is its tower's rows between {@code <|vision_start|>} and {@code
     * <|vision_end|>}; a video, between the same two, is each pair of frames so, behind the time
     * the pair was taken at, {@code <1.4 seconds>}.
     */
    private Head head(Judged judged, MemoryArena<MemorySegment> arena, int capacity) {
        if (judged.media().isEmpty()) return NONE;
        List<Integer> tokens = new ArrayList<>();
        tokens.add(bos);
        tokens.add(imStart);
        for (int id : tokenizer.encodeToArray("user\n")) tokens.add(id);
        List<Lfm3.Placed> placed = new ArrayList<>();
        for (Media media : judged.media()) {
            if (media instanceof Media.Video video) {
                tokens.add(visionOpen);
                video(video, arena, tokens, placed, capacity);
                tokens.add(visionClose);
            } else {
                place(encode(taken(media), arena, capacity - tokens.size()), tokens, placed);
            }
        }
        return new Head(tokens.stream().mapToInt(Integer::intValue).toArray(), placed);
    }

    /** {@code rows} between the two markers, a placeholder token a row. */
    private void place(
            MemoryView<MemorySegment> rows, List<Integer> tokens, List<Lfm3.Placed> placed) {
        int count = Math.toIntExact(rows.shape().flatAt(0));
        tokens.add(visionOpen);
        placed.add(new Lfm3.Placed(tokens.size(), rows, count));
        for (int i = 0; i < count; i++) tokens.add(visionOpen); // never read
        tokens.add(visionClose);
    }

    /**
     * A video's frames two by two, the last one twice when they are odd: each pair one temporal
     * patch, stamped with the mean of its two frames' times to a tenth of a second.
     */
    private void video(
            Media.Video video,
            MemoryArena<MemorySegment> arena,
            List<Integer> tokens,
            List<Lfm3.Placed> placed,
            int capacity) {
        Lfm3Vision vision = model.vision();
        if (vision == null) throw new UnsupportedOperationException(refusal(video));
        List<Media.Video.Frame> frames = video.frames();
        if (frames.isEmpty()) throw new IllegalArgumentException("a video without frames");
        Media.Image shape = frames.get(0).image();
        for (Media.Video.Frame frame : frames)
            if (frame.image().width() != shape.width() || frame.image().height() != shape.height())
                throw new IllegalArgumentException("a video's frames are of one size");
        int[] size = vision.videoSize(frames.size(), shape);
        int count = vision.positions(size), dim = model.width();
        fits((frames.size() + 1) / 2 * count, capacity - tokens.size());
        for (int i = 0; i < frames.size(); i += 2) {
            Media.Video.Frame first = frames.get(i);
            Media.Video.Frame second = frames.get(Math.min(i + 1, frames.size() - 1));
            double seconds =
                    (first.timestamp().toNanos() + second.timestamp().toNanos()) / 2.0 / 1e9;
            // the reference's "%.1f": the double's exact value, ties to even
            String stamp = new BigDecimal(seconds).setScale(1, RoundingMode.HALF_EVEN).toString();
            for (int id : tokenizer.encodeToArray("<" + stamp + " seconds>")) tokens.add(id);
            MemoryView<MemorySegment> rows = Views.allocateF32(arena, count, dim);
            MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
            try {
                Convert.copyF32(
                        vision.encode(first.image(), second.image(), size, scratch),
                        0,
                        rows,
                        0,
                        (long) count * dim);
            } finally {
                Arenas.close(scratch);
            }
            place(rows, tokens, placed);
        }
    }

    /** Refuses media of {@code rows} rows where {@code room} are left, before a tower runs. */
    private static void fits(int rows, int room) {
        if (rows > room)
            throw new IllegalArgumentException(
                    "the input's media take "
                            + rows
                            + " tokens where the state has "
                            + Math.max(0, room)
                            + " left: open a larger one with newState");
    }

    /** One picture's or clip's rows from its tower, {@code room} rows being left for them. */
    private <R extends Media> MemoryView<MemorySegment> encode(
            R media, MemoryArena<MemorySegment> arena, int room) {
        @SuppressWarnings("unchecked")
        MediaProjector<R> projector =
                model.projector((Class<R>) media.getClass())
                        .orElseThrow(() -> new UnsupportedOperationException(refusal(media)));
        int count = projector.positions(media), dim = model.width();
        fits(count, room);
        MemoryView<MemorySegment> rows = Views.allocateF32(arena, count, dim);
        int[] done = {0};
        projector.project(
                media,
                count,
                chunk -> {
                    int n = Math.toIntExact(chunk.shape().flatAt(0));
                    Convert.copyF32(
                            Views.castToSegmentBacked(chunk, "media rows"),
                            0,
                            rows,
                            (long) done[0] * dim,
                            (long) n * dim);
                    done[0] += n;
                });
        if (done[0] != count)
            throw new IllegalStateException("a tower gave " + done[0] + " rows of " + count);
        return rows;
    }

    /**
     * A picture as the checkpoint was given it in training: one over {@link #MAX_PIXELS} is scaled
     * down to them first, keeping its shape, before the tower sizes it.
     */
    @SuppressWarnings("unchecked")
    private static <R extends Media> R taken(R media) {
        if (!(media instanceof Media.Image image)) return media;
        long pixels = (long) image.width() * image.height();
        if (pixels <= MAX_PIXELS) return media;
        double scale = Math.sqrt((double) MAX_PIXELS / pixels);
        int width = Math.max(1, (int) (image.width() * scale));
        int height = Math.max(1, (int) (image.height() * scale));
        int[] rgb = Resample.bicubic(image, width, height, true);
        float[] values = new float[rgb.length];
        for (int i = 0; i < rgb.length; i++) values[i] = rgb[i] / 255f;
        return (R) new Media.Image(values, height, width, 3);
    }

    private String refusal(Media media) {
        String kind = media.getClass().getSimpleName().toLowerCase(Locale.ROOT);
        return model.vision() == null && model.projector(Media.Audio.class).isEmpty()
                ? "this model reads "
                        + kind
                        + " through its mmproj: load it as the \"media\" companion"
                : "this model takes no " + kind;
    }

    /** The tokens of {@code question} asked alone about the text {@code input}. */
    int[] prompt(String input, Question question) {
        return prompt(NONE, input, asked(question).body());
    }

    private static String predicateBody(Predicate q) {
        StringBuilder text = new StringBuilder(q.text());
        if (q.yesMeans() != null || q.noMeans() != null)
            text.append("\nYes: ") // a missing side reads None, as the reference writes it
                    .append(q.yesMeans() == null ? "None" : q.yesMeans())
                    .append("\nNo: ")
                    .append(q.noMeans() == null ? "None" : q.noMeans());
        return text.append("\n\nReply with yes or no only.").toString();
    }

    private static String choiceBody(Choice q, String[] codes) {
        StringBuilder text = new StringBuilder(q.text()).append("\n\nOptions:\n");
        for (int i = 0; i < codes.length; i++) {
            if (i > 0) text.append('\n');
            Option option = q.options().get(i);
            text.append(codes[i])
                    .append(' ')
                    .append(shown(option, option.label().replace('_', ' ')));
        }
        return text.append("\n\nReply with the option code only.").toString();
    }

    /** Levels under the digits 0..n-1, which are also the tokens read: at most ten of them. */
    private static String ratingBody(Rating q) {
        List<Option> levels = q.levels();
        if (levels.size() > 10)
            throw new IllegalArgumentException(
                    "a rating has at most 10 levels on this model, a digit each, not "
                            + levels.size());
        StringBuilder text = new StringBuilder(q.text()).append("\n\n");
        for (int i = 0; i < levels.size(); i++)
            text.append(i)
                    .append(' ')
                    .append(shown(levels.get(i), levels.get(i).label()))
                    .append('\n');
        return text.append("\nReply with a single digit 0-")
                .append(levels.size() - 1)
                .append(" only.")
                .toString();
    }

    /** The one text the checkpoint read for an option: its description, else {@code label}. */
    private static String shown(Option option, String label) {
        String description = option.description();
        return description == null || description.isEmpty() ? label : description;
    }

    /**
     * A whole prompt's tokens, ending where the answer would start: the media, the text when there
     * is one, the question.
     */
    private int[] prompt(Head head, String text, String body) {
        String asks = (text == null ? "" : plain(text + "\n\n") + "\n" + HEADER) + plain(body);
        if (head.tokens() == null) return framed(true, "user\n" + asks);
        int[] rest = framed(false, asks);
        int[] ids = Arrays.copyOf(head.tokens(), head.tokens().length + rest.length);
        System.arraycopy(rest, 0, ids, head.tokens().length, rest.length);
        return ids;
    }

    /**
     * A question's own run of a packed prompt: {@code header}, its text, through its answer slot.
     */
    private int[] part(String header, String body) {
        return framed(false, header + plain(body));
    }

    /**
     * {@code text}, after the opening tokens when {@code opens}, then the turn's end up to the
     * answer slot.
     */
    private int[] framed(boolean opens, String text) {
        int[] body = specials.encode(tokenizer, text).toArray();
        int[] newline = tokenizer.encodeToArray("\n");
        int[] assistant = tokenizer.encodeToArray("assistant\n");
        int[] ids = new int[(opens ? 2 : 0) + body.length + newline.length + assistant.length + 4];
        int at = 0;
        if (opens) {
            ids[at++] = bos;
            ids[at++] = imStart;
        }
        System.arraycopy(body, 0, ids, at, body.length);
        at += body.length;
        ids[at++] = imEnd;
        System.arraycopy(newline, 0, ids, at, newline.length);
        at += newline.length;
        ids[at++] = imStart;
        System.arraycopy(assistant, 0, ids, at, assistant.length);
        at += assistant.length;
        ids[at++] = think;
        ids[at] = endThink;
        return ids;
    }

    /**
     * {@code text} as the caller's own: a control token written in it is broken by a zero-width
     * space after its first character, so an input holding {@code <|im_end|>} cannot close the
     * turn. Repeated for a match that overlapped the one broken before it. Other added tokens
     * ({@code <think>}) stay tokens, as the checkpoint saw them in training.
     */
    private String plain(String text) {
        for (boolean changed = true; changed; ) {
            changed = false;
            StringBuilder out = new StringBuilder(text.length());
            for (int i = 0; i < text.length(); ) {
                String hit = null;
                for (String control : controls)
                    if (text.startsWith(control, i)) {
                        hit = control;
                        break;
                    }
                if (hit == null) {
                    out.append(text.charAt(i++));
                    continue;
                }
                out.append(hit, 0, 1).append(BREAK).append(hit, 1, hit.length());
                i += hit.length();
                changed = true;
            }
            text = out.toString();
        }
        return text;
    }

    // --- the labels ---

    /**
     * A choice's codes as the options list shows them: the labels themselves when every one is a
     * letter, else {@code A..Z}, then {@code AA, AB, ..}; a code the tokenizer does not hold as one
     * token of its own is replaced from the pool. {@code labels} gets each option's token ids: its
     * code, and the code after a space where that is another single token.
     */
    private String[] codes(List<Option> options, int[][] labels) {
        int n = options.size();
        boolean letters = true;
        for (Option option : options) {
            String name = option.label().strip();
            letters &= name.length() == 1 && Character.isLetter(name.charAt(0));
        }
        List<String> pool = null;
        Set<Integer> used = new HashSet<>();
        String[] codes = new String[n];
        for (int i = 0; i < n; i++) {
            String code =
                    letters
                            ? options.get(i).label().strip()
                            : i < 26
                                    ? Character.toString('A' + i)
                                    : ""
                                            + (char) ('A' + (i - 26) / 26)
                                            + (char) ('A' + (i - 26) % 26);
            int id = single(code);
            if (id < 0 || !used.add(id)) {
                if (pool == null) pool = pool();
                id = -1;
                for (String candidate : pool) {
                    int c = single(candidate);
                    if (c >= 0 && used.add(c)) {
                        code = candidate;
                        id = c;
                        break;
                    }
                }
                if (id < 0)
                    throw new IllegalArgumentException(
                            "no single-token label left for " + n + " options");
            }
            codes[i] = code;
            int spaced = single(" " + code);
            labels[i] = spaced >= 0 && spaced != id ? new int[] {id, spaced} : new int[] {id};
        }
        return codes;
    }

    /** The reference's fallback codes, in its order. */
    private static List<String> pool() {
        List<String> pool = new ArrayList<>();
        for (char c = 'A'; c <= 'Z'; c++) pool.add(Character.toString(c));
        for (int i = 0; i < 100; i++) pool.add(String.format("%02d", i));
        for (char c = 'a'; c <= 'z'; c++) pool.add(Character.toString(c));
        for (int i = 0; i < 200; i++) pool.add("#" + i);
        for (char a = 'A'; a <= 'Z'; a++) for (char b = 'A'; b <= 'Z'; b++) pool.add("" + a + b);
        return pool;
    }

    /** The distinct ids of the forms that are one token each. */
    private int[] singles(String... forms) {
        int[] ids = new int[forms.length];
        int n = 0;
        for (String form : forms) {
            int id = single(form);
            boolean seen = id < 0;
            for (int i = 0; i < n && !seen; i++) seen = ids[i] == id;
            if (!seen) ids[n++] = id;
        }
        return Arrays.copyOf(ids, n);
    }

    /** {@code text}'s token id when it is exactly one token, else -1. */
    private int single(String text) {
        int[] ids = tokenizer.encodeToArray(text);
        return ids.length == 1 ? ids[0] : -1;
    }
}
