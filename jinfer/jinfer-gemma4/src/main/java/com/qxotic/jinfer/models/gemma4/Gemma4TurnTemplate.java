package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Embedder;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.MultiModal;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Hand-written Gemma 4 chat framing, matching the GGUF chat_template's plain-conversation shape and
 * the hand-built prompt precedent (Gemma4VisionRun/Gemma4AudioRun).
 *
 * <p>Layout: {@code <bos>} once, then per turn {@code <|turn>{role}\n{content}<turn|>\n},
 * generation prompt {@code <|turn>model\n}. Gemma's assistant role name is {@code model} ({@link
 * Role#ASSISTANT} maps to it). The role header and the content are tokenized as ONE contiguous
 * plain-encoded run — that is how a rendered template tokenizes (specials force the only splits),
 * and BPE merges across the header/content boundary.
 *
 * <p>Media parts are structural (never parsed out of text): an image lowers to {@code <|image>}
 * [bidirectional embeddings] {@code <image|>} and audio to {@code <|audio>} [causal embeddings]
 * {@code <audio|>}, in part order, encoders resolved through the model's {@link MultiModal} seam at
 * encode time. A text-only load still frames text turns; a media part then throws naming the
 * missing encoder.
 *
 * <p>Two domains: {@code <bos>}/{@code <|turn>}/{@code <turn|>}/media wrappers are emitted as
 * trusted ids; everything else goes through plain {@link Tokenizer#encode} so conversation text can
 * never mint control tokens.
 */
public final class Gemma4TurnTemplate implements TurnTemplate {

    // Gemma spells reasoning as a named channel, not <think> - spelled once, read by both the
    // non-thinking scaffold and the reply parser.
    static final String CHANNEL_OPEN = "<|channel>";
    static final String CHANNEL_CLOSE = "<channel|>";

    private final Tokenizer tokenizer;
    private final MultiModal media; // encoder source; null or empty modalities on text-only loads
    private final int modelDim;
    private final int bos; // <bos>
    private final int turnOpen; // <|turn>
    private final int turnClose; // <turn|>
    private final List<Integer> newline; // encode("\n"), constant
    private final List<Batch> generationPrompt; // <|turn>model\n, constant
    // <|turn>model\n<|channel>thought\n<channel|> - the NON-thinking prefix. The template's rule
    // is inverted from the obvious reading: thinking OFF is what needs scaffold, an EMPTY and
    // already-closed thought channel, so the model goes straight to the answer. Omit it and the
    // model helpfully writes the header itself, whose channel NAME is plain text and lands in the
    // reply as a literal "thought" in front of every answer.
    //
    // Not every Gemma 4 checkpoint has this rule: E2B's chat_template ends its generation prompt
    // at <|turn>model\n unconditionally, so `thinking` is a no-op there and this aliases
    // generationPrompt. Its vocabulary still carries the channel specials (the template splits on
    // them to strip thought from history), so token presence cannot tell the two apart - given the
    // scaffold off-contract, E2B answers in reasoning prose instead of skipping the thought.
    private final List<Batch> generationPromptNoThink;
    // the checkpoint declares channel-aware generation scaffolding (12B/26B); without it the
    // tool-response thought tail would be off-template (E2B)
    private final boolean scaffoldsNonThinking;
    // <|channel>thought\n after a trailing tool response (thinking on) - constant, co-produced
    // with its reply seed; empty when the checkpoint has no channel scaffolding
    private final List<Batch> thoughtTail;
    private final int[] thoughtSeed;
    private final List<Batch> closeTurn; // <turn|>\n, constant
    private final TokenRuns proto; // compiled spelling table, forked per turn

    /**
     * Text-only, and no non-thinking scaffold - see {@link #Gemma4TurnTemplate(Tokenizer,
     * MultiModal, int, boolean)}.
     */
    public Gemma4TurnTemplate(Tokenizer tokenizer) {
        this(tokenizer, null, 0, false);
    }

    /**
     * {@code scaffoldsNonThinking} is the checkpoint's answer to one question: does its
     * chat_template close an empty thought channel when thinking is off? {@link
     * Gemma4#turnTemplate()} reads it from the template source; false makes {@code thinking} a
     * no-op, the safe reading, since the scaffold harms checkpoints that do not declare it.
     */
    public Gemma4TurnTemplate(
            Tokenizer tokenizer, MultiModal media, int modelDim, boolean scaffoldsNonThinking) {
        this.tokenizer = tokenizer;
        this.media = media;
        this.modelDim = modelDim;
        this.bos = SpecialTokens.require(tokenizer, "<bos>");
        this.turnOpen = SpecialTokens.require(tokenizer, "<|turn>");
        this.turnClose = SpecialTokens.require(tokenizer, "<turn|>");
        this.newline = List.copyOf(tokenizer.encode("\n").toList());
        List<Integer> gen = new ArrayList<>();
        gen.add(turnOpen);
        gen.addAll(tokenizer.encode("model\n").toList());
        this.generationPrompt = List.of(Batch.prefill(gen));
        if (scaffoldsNonThinking) {
            List<Integer> noThink = new ArrayList<>(gen);
            noThink.add(SpecialTokens.require(tokenizer, CHANNEL_OPEN));
            noThink.addAll(tokenizer.encode("thought\n").toList());
            noThink.add(SpecialTokens.require(tokenizer, CHANNEL_CLOSE));
            this.generationPromptNoThink = List.of(Batch.prefill(noThink));
        } else {
            this.generationPromptNoThink = generationPrompt;
        }
        List<Integer> close = new ArrayList<>();
        close.add(turnClose);
        close.addAll(newline);
        this.closeTurn = List.of(Batch.prefill(close));
        this.scaffoldsNonThinking = scaffoldsNonThinking;
        if (scaffoldsNonThinking) {
            List<Integer> tail = new ArrayList<>();
            tail.add(SpecialTokens.require(tokenizer, CHANNEL_OPEN));
            tail.addAll(tokenizer.encode("thought\n").toList());
            this.thoughtTail = List.of(Batch.prefill(tail));
            this.thoughtSeed = tail.stream().mapToInt(Integer::intValue).toArray();
        } else {
            this.thoughtTail = List.of();
            this.thoughtSeed = new int[0];
        }
        this.proto = new TokenRuns(tokenizer);
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[] {bos}));
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        // <|turn> {role}\n{parts...} <turn|> \n - text accumulates into contiguous plain runs,
        // media cuts the stream and splices its wrapped embeddings block in part order.
        TokenRuns runs = proto.fresh();
        runs.id(turnOpen).text(roleName(message.role())).text("\n");
        boolean hasMedia = message.content().stream().anyMatch(p -> p instanceof Part.Blob);
        if (!hasMedia) {
            // Gemma's template trims each message's text (| trim for user/system, strip_thinking
            // for model); a text-only turn's content is stripped to stay token-exact with the
            // render.
            runs.text(message.textOnly().strip());
        } else {
            for (Part p : message.content()) {
                if (p instanceof Part.Text t) {
                    runs.text(t.text());
                } else if (p instanceof Part.Blob blob) {
                    encodeMedia(blob, runs);
                }
            }
        }
        runs.id(turnClose).text("\n");
        return runs.batches();
    }

    // === media-encode cache: PER MODEL, in memory ===
    // Replay of encoder output, keyed by the media's SOURCE digest. The digest alone is a complete
    // key BECAUSE THIS CACHE BELONGS TO ONE MODEL: everything else the towers depend on - the
    // projector, the image decoder, the token budgets, the resize mode - is fixed for this
    // template's lifetime, so it cannot vary between two entries here. (The BLOCK key handed to
    // Batch.embeddings is a different matter: it outlives the process, so it has to carry what
    // this one may omit.)
    //
    // Values hold rows plus block structure, never a finished batch: replay re-emits timestamp TEXT
    // through the token runs so tokenization fuses with the surrounding turn exactly like a cold
    // encode, and a finished batch would freeze token boundaries that depend on neighboring text
    // (the byte-identity law). Keyless blobs bypass the cache.

    /**
     * LRU budget in rows bytes, PER MODEL - each loaded model keeps its own cache, so N models cost
     * N times this. Sized for one model's working set rather than a process-wide pool, because a
     * model's cached rows are worthless to any other model: they were produced by its projector.
     */
    private static final long MEDIA_CACHE_BYTES_PER_MODEL =
            Long.getLong("jinfer.mediaCacheMB", 192) << 20;

    /**
     * One wrapped media block: optional leading text, then {@code <open>} [rows] {@code <close>}.
     */
    private record CachedBlock(
            String text,
            int openId,
            int closeId,
            FloatTensor rows,
            boolean bidirectional,
            byte[] blockKey) {}

    /** Access-ordered, so {@code get} refreshes LRU position. Keys print as {@code sha256:…}. */
    private final LinkedHashMap<String, List<CachedBlock>> mediaCache =
            new LinkedHashMap<>(16, 0.75f, true);

    private long mediaCacheBytes;

    /** A digest as a map key: a String has value equality, which a {@code byte[]} does not. */
    private static String cacheKey(byte[] digest) {
        return "sha256:" + HexFormat.of().formatHex(digest);
    }

    private static long rowsBytes(List<CachedBlock> blocks) {
        long bytes = 0;
        for (CachedBlock b : blocks) {
            bytes += b.rows().size() * Float.BYTES;
        }
        return bytes;
    }

    @Override
    public boolean mediaEncodingCached(byte[] contentKey) {
        return contentKey != null && cacheGet(contentKey) != null;
    }

    private List<CachedBlock> cacheGet(byte[] key) {
        synchronized (mediaCache) {
            return mediaCache.get(cacheKey(key));
        }
    }

    private void cachePut(byte[] key, List<CachedBlock> blocks) {
        long add = rowsBytes(blocks);
        synchronized (mediaCache) {
            if (mediaCache.putIfAbsent(cacheKey(key), blocks) != null) return;
            mediaCacheBytes += add;
            var eldest = mediaCache.entrySet().iterator();
            // never evict the entry just inserted (it is the one the caller is about to use), so a
            // single oversized entry is allowed to exceed the budget rather than evict itself
            while (mediaCacheBytes > MEDIA_CACHE_BYTES_PER_MODEL && mediaCache.size() > 1) {
                mediaCacheBytes -= rowsBytes(eldest.next().getValue());
                eldest.remove();
            }
        }
    }

    /**
     * {@code <open>} [embeddings] {@code <close>}: wrapper ids around the encoded block —
     * bidirectional for images (one attention group), causal for audio (gemma4ua). A keyed blob
     * replays from the media cache when its digest is known - the media payload is not touched on a
     * hit, which is what lets a caller pass a frameless keyed video (see {@link
     * com.qxotic.jinfer.chat.ChatTemplate#mediaEncodingCached}).
     */
    private void encodeMedia(Part.Blob blob, TokenRuns runs) {
        byte[] key = blob.contentKey();
        if (key != null) {
            List<CachedBlock> hit = cacheGet(key);
            if (hit != null) {
                for (CachedBlock b : hit) emit(null, runs, b);
                return;
            }
            if (blob.media() instanceof Media.Video v && v.frames().isEmpty()) {
                throw new IllegalStateException(
                        "frameless keyed video: its media-cache entry was evicted between the"
                                + " caller's mediaEncodingCached check and encode - retry with the"
                                + " decoded video");
            }
        }
        List<CachedBlock> record = key == null ? null : new ArrayList<>();
        Media m = blob.media();
        switch (m) {
            case Media.Image img ->
                    emit(
                            record,
                            runs,
                            block(
                                    null,
                                    "<|image>",
                                    "<image|>",
                                    encode(Media.Image.class, img),
                                    true,
                                    key));
            case Media.Audio aud ->
                    emit(
                            record,
                            runs,
                            block(
                                    null,
                                    "<|audio>",
                                    "<audio|>",
                                    encode(Media.Audio.class, aud),
                                    false,
                                    key));
            case Media.Video vid -> {
                // Video decomposes into frames, rendered token-exact with the reference
                // processor's replace_video_token: segments "MM:SS <|image>[soft]<image|>" joined
                // by SINGLE SPACES ("00:00 <|image>...<image|> 00:01 <|image>..." - no newlines);
                // minutes/seconds are the floor of the frame's TRUE timestamp (any sampling).
                // Frames encode at the VIDEO budget (default 70, the reference video processor's
                // own; -Djinfer.gemma4.videoTokenBudget) - stills keep the independent image
                // budget, so many frames fit the context. Frame keys DERIVE from the video key
                // (digest ‖ timestamp) - never shared (same key + same in-batch positions would
                // collide across frames).
                boolean first = true;
                for (Media.Video.Frame frame : vid.frames()) {
                    Duration t = frame.timestamp();
                    String ts =
                            String.format(
                                    first ? "%02d:%02d " : " %02d:%02d ",
                                    t.toMinutes(),
                                    t.toSecondsPart());
                    emit(
                            record,
                            runs,
                            block(
                                    ts,
                                    "<|image>",
                                    "<image|>",
                                    encodeFrame(frame.image()),
                                    true,
                                    frameKey(key, t)));
                    first = false;
                }
            }
            default ->
                    throw new IllegalArgumentException(
                            "Gemma 4: unsupported media " + m.getClass().getSimpleName());
        }
        if (record != null) cachePut(key, record);
    }

    private CachedBlock block(
            String text, String open, String close, FloatTensor rows, boolean bidi, byte[] key) {
        return new CachedBlock(
                text,
                SpecialTokens.require(tokenizer, open),
                SpecialTokens.require(tokenizer, close),
                rows,
                bidi,
                key);
    }

    /** Emit one block into the runs, recording it for the cache when {@code record} is present. */
    private void emit(List<CachedBlock> record, TokenRuns runs, CachedBlock b) {
        if (b.text() != null) runs.text(b.text());
        runs.id(b.openId())
                .block(
                        Batch.embeddings(
                                b.rows(),
                                (int) (b.rows().size() / modelDim),
                                b.bidirectional(),
                                b.blockKey()))
                .id(b.closeId());
        if (record != null) record.add(b);
    }

    /**
     * One video frame through the vision tower at the VIDEO soft-token budget ({@link
     * VisionPreprocess#VIDEO_TOKEN_BUDGET}, default 70 - the reference video processor's own
     * default; stills keep the independent image budget). A tower without the {@link VisionBudget}
     * seam falls back to its image budget.
     */
    private FloatTensor encodeFrame(Media.Image img) {
        if (media != null
                && media.embedder(Media.Image.class).orElse(null) instanceof VisionBudget tower) {
            return tower.encode(img, VisionPreprocess.VIDEO_TOKEN_BUDGET);
        }
        return encode(Media.Image.class, img);
    }

    /**
     * A sampled frame's cache key: SHA-256(video digest ‖ timestamp nanos), or null for a keyless
     * video. The timestamp - not a frame index - is the coordinate: the same instant keys the same
     * pixels under any sampling policy, while an index means different content whenever the policy
     * or frame count changes. No frame shares the raw video key (same key + same in-batch positions
     * would collide across frames).
     */
    private static byte[] frameKey(byte[] videoKey, Duration t) {
        if (videoKey == null) return null;
        try {
            var md = MessageDigest.getInstance("SHA-256");
            md.update(videoKey);
            md.update(ByteBuffer.allocate(Long.BYTES).putLong(t.toNanos()).array());
            return md.digest();
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * Runs the modality's embedder and materializes the model-dim rows (chunks are ephemeral
     * views).
     */
    /** Best-effort media positions via the modality's embedder plan (no encoding). */
    @Override
    public int mediaPositions(Media m) {
        return switch (m) {
            case Media.Image img -> plan(Media.Image.class, img);
            case Media.Audio aud -> plan(Media.Audio.class, aud);
            default ->
                    throw new UnsupportedOperationException(
                            m.getClass().getSimpleName() + " is not supported by this model");
        };
    }

    private <R extends Media> int plan(Class<R> type, R m) {
        if (media == null) {
            throw new UnsupportedOperationException(unsupported(type));
        }
        return media.embedder(type)
                .orElseThrow(() -> new UnsupportedOperationException(unsupported(type)))
                .positions(m);
    }

    /** Lower-case modality on purpose: framework kits grep for the word as the user typed it. */
    private static String unsupported(Class<? extends Media> type) {
        return type.getSimpleName().toLowerCase(java.util.Locale.ROOT)
                + " input is not supported by this model (load the mmproj sidecar)";
    }

    private <R extends Media> FloatTensor encode(Class<R> type, R m) {
        // capability gap, not an internal invariant: UnsupportedOperationException like plan(),
        // so the framework adapters surface their own unsupported-feature type instead of a 500
        if (media == null) {
            throw new UnsupportedOperationException(unsupported(type));
        }
        Embedder<R> embedder =
                media.embedder(type)
                        .orElseThrow(() -> new UnsupportedOperationException(unsupported(type)));
        // ONE copy out of the sink's ephemeral view, straight into the caller-owned (GC-managed)
        // return - the unbounded maxChunkSize means one chunk in practice, so the concatenation
        // leg below is a contract-completeness fallback, not a path that runs
        List<F32FloatTensor> parts = new ArrayList<>(1);
        embedder.embed(
                m,
                Integer.MAX_VALUE,
                t -> {
                    F32FloatTensor copy = F32FloatTensor.allocate(Arena.ofAuto(), (int) t.size());
                    t.copyTo(0, copy, 0, (int) t.size());
                    parts.add(copy);
                });
        if (parts.size() == 1) return parts.get(0);
        int total = 0;
        for (F32FloatTensor p : parts) total += (int) p.size();
        F32FloatTensor rows = F32FloatTensor.allocate(Arena.ofAuto(), total);
        int at = 0;
        for (F32FloatTensor p : parts) {
            p.copyTo(0, rows, at, (int) p.size());
            at += (int) p.size();
        }
        return rows;
    }

    /**
     * The codec face, media and tools included: text and {@link Part.Blob} parts fold through the
     * media-capable {@link #encodeTurn}; tools render the template's exact flow - declarations in
     * the system turn ({@code <|tool>declaration:...<tool|>}), and the whole call round-trip as ONE
     * open model turn: {@code <|tool_call>call:...<tool_call|>} then folded {@code
     * <|tool_response>response:...<tool_response|>} blocks then the answer text, with the
     * generation prompt suppressed while the model turn is open. Reasoning follows the upstream
     * template fix (35b4173): a {@link Part.Reasoning} renders as {@code <|channel>thought\n...} in
     * assistant turns AFTER the last user message (the in-flight tool loop) and is stripped
     * everywhere before it; {@code preserve_thinking} is not plumbed (punts to the whole render).
     * Punts: media on a text-only load.
     *
     * <p>The generation tail follows the fixed reference plus llama.cpp's patch on its one quirk:
     * after a TRAILING tool response with thinking on, the prompt re-opens the thought channel
     * ({@code <|channel>thought\n}) and the reply begins inside it - so this method co-produces
     * that tail with its {@link Prompt#replySeed} (one computation, parser state cannot disagree
     * with the prompt). When the final model turn CLOSED with {@code prev} still reading 'response'
     * (an assistant message carrying call + answer content), the raw template emits a bare thought
     * channel outside any turn; llama.cpp reopens the model turn instead, and this port follows the
     * patch.
     */
    @Override
    public Prompt encodePrompt(Conversation conversation) {
        requireSupported(conversation);
        List<Message> msgs = conversation.messages();
        List<Batch> out = new ArrayList<>(conversationStart());
        boolean systemFirst = !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM);
        int start = 0;
        if (systemFirst || !conversation.tools().isEmpty()) {
            systemBlock(systemFirst ? msgs.get(0) : null, conversation.tools(), out);
            if (systemFirst) start = 1;
        }
        // upstream template fix 35b4173: reasoning is PRESERVED in turns after the last user
        // message (the in-flight tool loop) and stripped everywhere before it
        int lastUserIdx = -1;
        for (int i = 0; i < msgs.size(); i++) {
            if (msgs.get(i).role().equals(Role.USER)) lastUserIdx = i;
        }
        // the template's per-message state: 'call'/'response' leave the model turn OPEN
        String prev = null;
        Role prevNonToolRole = null;
        boolean openTail = false; // did the FINAL emitted turn stay open?
        TokenRuns carry = null; // merged model turns share ONE runs so their text BPE-fuses
        for (int i = start; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) continue; // folded into its call turn below
            prev = null;
            openTail = false;
            boolean assistant = m.role().equals(Role.ASSISTANT);
            boolean continuation = assistant && Role.ASSISTANT.equals(prevNonToolRole);
            prevNonToolRole = m.role();
            List<Part.ToolCall> calls =
                    m.content().stream()
                            .filter(p -> p instanceof Part.ToolCall)
                            .map(p -> (Part.ToolCall) p)
                            .toList();
            // thinking_gate (preserve_thinking defaults false and is not plumbed - a request
            // setting it punts to the whole render): render reasoning only after the last user
            Part.Reasoning reasoning = assistant ? m.reasoning() : null;
            String thought =
                    reasoning != null && i > lastUserIdx && !reasoning.text().isEmpty()
                            ? reasoning.text()
                            : null;
            // turn-tag balance (upstream fix): an assistant turn does not close when the next
            // non-tool message is also assistant - the turns merge into one model turn
            Role nextRole = null;
            for (int j = i + 1; j < msgs.size(); j++) {
                if (!msgs.get(j).role().equals(Role.TOOL)) {
                    nextRole = msgs.get(j).role();
                    break;
                }
            }
            boolean continuesIntoNext = assistant && Role.ASSISTANT.equals(nextRole);
            if (calls.isEmpty() && !continuation && reasoning == null && !continuesIntoNext) {
                out.addAll(encodeTurn(m));
                continue;
            }
            if (m.content().stream().anyMatch(p -> p instanceof Part.Blob)) {
                throw new UnsupportedConversation("media in a tool-call/continuation model turn");
            }
            TokenRuns runs = carry != null ? carry : proto.fresh();
            carry = null;
            if (!continuation) {
                runs.id(turnOpen).text(roleName(m.role())).text("\n");
            }
            if (thought != null) {
                // <|channel>thought\n{reasoning}\n<channel|> - before calls and content
                runs.id(require(CHANNEL_OPEN))
                        .text("thought\n")
                        .text(thought)
                        .text("\n")
                        .id(require(CHANNEL_CLOSE));
            }
            for (Part.ToolCall call : calls) {
                runs.id(require("<|tool_call>"));
                sinkInto(runs, s -> Gemma4ToolSyntax.call(call.name(), call.arguments(), s));
                runs.id(require("<tool_call|>"));
            }
            // forward-fold the consecutive tool-role results, names resolved from the calls.
            // A tool turn's result arrives as Part.ToolResult (typed API) OR Part.Text (the
            // server's lowering shape - same law as NemotronH); dropping the Text form silently
            // starved the model of every served tool result.
            boolean responses = false;
            int nthResult = 0; // results fold in call order: the id-less Text shape resolves
            for (int j = i + 1; j < msgs.size() && msgs.get(j).role().equals(Role.TOOL); j++) {
                for (Part part : msgs.get(j).content()) {
                    String callId;
                    String resultText;
                    if (part instanceof Part.ToolResult r) {
                        callId = r.callId();
                        resultText = r.text();
                    } else if (part instanceof Part.Text t && !t.text().isEmpty()) {
                        callId = "";
                        resultText = t.text();
                    } else {
                        continue;
                    }
                    runs.id(require("<|tool_response>"));
                    String name =
                            callId.isEmpty() && nthResult < calls.size()
                                    ? calls.get(nthResult).name()
                                    : resolveName(calls, callId);
                    nthResult++;
                    sinkInto(runs, s -> Gemma4ToolSyntax.response(name, resultText, s));
                    runs.id(require("<tool_response|>"));
                    responses = true;
                    prev = "response";
                }
            }
            if (!calls.isEmpty() && !responses) prev = "call";
            String content = m.text().strip(); // the lenient view: call parts render above
            runs.text(content);
            if ("call".equals(prev)) {
                runs.id(require("<|tool_response>")); // awaiting results: the turn stays open
            } else if (continuesIntoNext) {
                // turn-tag balance (upstream fix): no close - the next assistant message
                // continues this model turn; hand it the SAME runs so juxtaposed text
                // BPE-merges exactly like the whole render
                carry = runs;
                continue;
            } else if (!(responses && content.isEmpty() && nextRole == null)) {
                // close unless the conversation ENDS on folded responses with no answer yet
                // (then the open turn is the generation surface); a following non-assistant
                // turn always closes this one (upstream turn-tag balance: next_nt.found)
                runs.id(turnClose).text("\n");
            } else {
                openTail = true; // trailing folded responses: the model turn stays open
            }
            out.addAll(runs.batches());
        }
        int[] seed = new int[0];
        if (openTail) {
            // trailing folded responses left the turn open. Thinking on: the model RESUMES
            // thinking - the prompt opens the channel and the reply starts inside it, so the
            // seed IS this tail. Thinking off (or no channel scaffolding): the model answers
            // directly in the open turn - the reference emits nothing.
            if (conversation.thinking() && scaffoldsNonThinking) {
                out.addAll(thoughtTail);
                seed = thoughtSeed;
            }
        } else if (!"call".equals(prev)) {
            // normal end - and the llama.cpp patch: a CLOSED final turn always reopens
            // <|turn>model regardless of what prev reads ("call" stays open at the await marker)
            out.addAll(generationPrompt(conversation.thinking()));
        }
        return new Prompt(out, seed);
    }

    @Override
    public List<Batch> encode(Conversation conversation) {
        return encodePrompt(conversation).batches();
    }

    /**
     * {@code <|turn>system\n} + trimmed system text + one {@code <|tool>declaration<tool|>} block
     * per tool + {@code <turn|>\n} - the template's tool-definitions block.
     */
    private void systemBlock(Message system, List<Tool> tools, List<Batch> out) {
        TokenRuns runs = proto.fresh();
        runs.id(turnOpen).text("system\n");
        if (system != null) runs.text(system.textOnly().strip());
        for (Tool tool : tools) {
            runs.id(require("<|tool>"));
            Object parsed = JsonCodec.parse(tool.rawJson());
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) parsed;
            sinkInto(runs, s -> Gemma4ToolSyntax.declaration(map, s));
            runs.id(require("<tool|>"));
        }
        runs.id(turnClose).text("\n");
        out.addAll(runs.batches());
    }

    /** Runs a tool-syntax renderer: text runs accumulate, quotes emit the trusted id. */
    private void sinkInto(TokenRuns runs, Consumer<Gemma4ToolSyntax.Sink> render) {
        render.accept(
                new Gemma4ToolSyntax.Sink() {
                    @Override
                    public void text(String s) {
                        runs.text(s);
                    }

                    @Override
                    public void quote() {
                        runs.id(require("<|\"|>"));
                    }
                });
    }

    private static String resolveName(List<Part.ToolCall> calls, String callId) {
        for (Part.ToolCall call : calls) {
            if (call.id().equals(callId)) return call.name();
        }
        return calls.size() == 1 ? calls.get(0).name() : "unknown";
    }

    private int require(String name) {
        return SpecialTokens.require(tokenizer, name);
    }

    /** The part shapes this port frames byte-exactly; anything else punts to the whole render. */
    private void requireSupported(Conversation conversation) {
        for (Message m : conversation.messages()) {
            boolean toolTurn = m.role().equals(Role.TOOL);
            boolean assistant = m.role().equals(Role.ASSISTANT);
            for (Part part : m.content()) {
                boolean ok =
                        switch (part) {
                            case Part.Text t -> true;
                            case Part.Blob b -> !toolTurn;
                            case Part.ToolCall c -> assistant;
                            case Part.ToolResult r -> toolTurn;
                            // upstream fix 35b4173: reasoning renders in turns after the last
                            // user message (thought channel), stripped before it
                            case Part.Reasoning r -> assistant;
                        };
                if (!ok)
                    throw new UnsupportedConversation(
                            m.role().name() + " turn: " + part.getClass().getSimpleName());
                if (part instanceof Part.Blob && media == null)
                    throw new UnsupportedConversation("media on a text-only load");
            }
        }
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return thinking ? generationPrompt : generationPromptNoThink;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    /** Gemma's template names the assistant turn {@code model}. */
    private static String roleName(Role role) {
        return role.equals(Role.ASSISTANT) ? "model" : role.name();
    }

    private ReplyLanguage.Spans spans; // the family's derived faces, markers written once

    private ReplyLanguage.Spans spans() {
        if (spans == null) {
            List<ReplyLanguage.Node> ends = new ArrayList<>();
            ends.add(ReplyLanguage.mark("<turn|>"));
            ends.add(ReplyLanguage.mark("<end_of_turn>"));
            ends.add(ReplyLanguage.mark("<|endoftext|>"));
            if (tokenizer.vocabulary().contains("<eos>")) {
                ends.add(ReplyLanguage.markId("<eos>", tokenizer.vocabulary().id("<eos>")));
            }
            spans =
                    new ReplyLanguage.Spans(
                            CHANNEL_OPEN,
                            CHANNEL_CLOSE,
                            "<|tool_call>",
                            "<tool_call|>",
                            Gemma4ToolSyntax::parseBlock,
                            new ReplyLanguage.Node.Alt(ends),
                            tokenizer);
        }
        return spans;
    }

    /**
     * The reply-language walk: the thought CHANNEL is the think span (the channel name streams
     * inside it, as the span parser saw it); a call span claims interior control tokens AS THEIR
     * SPELLINGS, so the {@code <|"|>} quote token reaches {@link Gemma4ToolSyntax#parseBlock}
     * exactly as the old span parser fed it; the mistyped-NORMAL {@code <eos>} (pinned by id) is
     * control everywhere - the old {@code dropping} wrapper and the handoff stop both derive from
     * the control rule.
     */
    @Override
    public ReplyParser parser() {
        return spans().parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedAuto(String contentGbnf) {
        return Optional.of(spans().constrainedAuto(contentGbnf));
    }

    /** Forced calls seed {@code <|tool_call>} and pin {@code call:name}. */
    @Override
    public int[] callSeed() {
        return new int[] {require("<|tool_call>")};
    }

    @Override
    public Optional<String> callPrefix() {
        return Optional.of("call:");
    }
}
