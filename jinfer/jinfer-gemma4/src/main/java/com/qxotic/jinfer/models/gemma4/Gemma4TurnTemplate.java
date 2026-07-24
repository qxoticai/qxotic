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
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
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

    private final Tokenizer tokenizer;
    private final MultiModal media; // encoder source; null or empty modalities on text-only loads
    private final int modelDim;
    private final int bos; // <bos>
    private final int turnOpen; // <|turn>
    private final int turnClose; // <turn|>
    private final List<Integer> newline; // encode("\n"), constant
    private final List<Batch> generationPrompt; // <|turn>model\n, constant
    private final List<Batch> closeTurn; // <turn|>\n, constant

    public Gemma4TurnTemplate(Tokenizer tokenizer) {
        this(tokenizer, null, 0);
    }

    public Gemma4TurnTemplate(Tokenizer tokenizer, MultiModal media, int modelDim) {
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
        List<Integer> close = new ArrayList<>();
        close.add(turnClose);
        close.addAll(newline);
        this.closeTurn = List.of(Batch.prefill(close));
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[] {bos}));
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        // <|turn> {role}\n{parts...} <turn|> \n — text accumulates into contiguous plain runs,
        // media flushes the run and splices its wrapped embeddings block in part order.
        List<Batch> out = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        ids.add(turnOpen);
        StringBuilder text = new StringBuilder(roleName(message.role())).append('\n');
        boolean hasMedia = message.content().stream().anyMatch(p -> p instanceof Part.Blob);
        if (!hasMedia) {
            // Gemma's template trims each message's text (| trim for user/system, strip_thinking
            // for
            // model); a text-only turn's content is stripped to stay token-exact with the render.
            text.append(message.textOnly().strip());
            flushText(text, ids);
        } else {
            for (Part p : message.content()) {
                if (p instanceof Part.Text t) {
                    text.append(t.text());
                } else if (p instanceof Part.Blob blob) {
                    flushText(text, ids);
                    encodeMedia(blob.media(), ids, out);
                }
            }
            flushText(text, ids);
        }
        ids.add(turnClose);
        ids.addAll(newline);
        out.add(Batch.prefill(ids));
        return out;
    }

    /**
     * {@code <open>} [embeddings] {@code <close>}: wrapper ids around the encoded block —
     * bidirectional for images (one attention group), causal for audio (gemma4ua).
     */
    private void encodeMedia(Media m, List<Integer> ids, List<Batch> out) {
        switch (m) {
            case Media.Image img -> {
                ids.add(SpecialTokens.require(tokenizer, "<|image>"));
                out.add(Batch.prefill(ids));
                ids.clear();
                FloatTensor rows = encode(Media.Image.class, img);
                out.add(Batch.embeddings(rows, (int) (rows.size() / modelDim)));
                ids.add(SpecialTokens.require(tokenizer, "<image|>"));
            }
            case Media.Audio aud -> {
                ids.add(SpecialTokens.require(tokenizer, "<|audio>"));
                out.add(Batch.prefill(ids));
                ids.clear();
                FloatTensor rows = encode(Media.Audio.class, aud);
                out.add(Batch.embeddings(rows, (int) (rows.size() / modelDim), false));
                ids.add(SpecialTokens.require(tokenizer, "<audio|>"));
            }
            case Media.Video vid -> {
                // Video decomposes into frames: each frame is a timestamped image block,
                // interleaved
                // as the docs show ("00:00 <|image>...", "00:01 ..."). Timestamps are plain text
                // (not
                // special tokens). Per-frame token cost = image budget (~256 at budget 280) - use a
                // low
                // jinfer.gemma4.imageTokenBudget for video so many frames fit the context.
                Media.Image[] frames = vid.frames();
                for (int i = 0; i < frames.length; i++) {
                    int sec = (int) (i / Math.max(vid.fps(), 1f));
                    ids.addAll(
                            tokenizer
                                    .encode(String.format("%n%02d:%02d%n", sec / 60, sec % 60))
                                    .toList());
                    encodeMedia(frames[i], ids, out);
                }
            }
            default ->
                    throw new IllegalArgumentException(
                            "Gemma 4: unsupported media " + m.getClass().getSimpleName());
        }
    }

    /**
     * Runs the modality's embedder and materializes the model-dim rows (chunks are ephemeral
     * views).
     */
    private <R extends Media> FloatTensor encode(Class<R> type, R m) {
        if (media == null) {
            throw new IllegalStateException(
                    "text-only template: construct with the model's encoders for media turns");
        }
        Embedder<R> embedder =
                media.embedder(type)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "this Gemma 4 load carries no "
                                                        + type.getSimpleName()
                                                        + " encoder (load with an mmproj)"));
        List<float[]> chunks = new ArrayList<>();
        embedder.embed(
                m,
                Integer.MAX_VALUE,
                t -> {
                    float[] c = new float[(int) t.size()];
                    for (int i = 0; i < c.length; i++) c[i] = t.getFloat(i);
                    chunks.add(c);
                });
        int total = 0;
        for (float[] c : chunks) total += c.length;
        F32FloatTensor rows = F32FloatTensor.allocate(total);
        int at = 0;
        for (float[] c : chunks) {
            for (float v : c) rows.setFloat(at++, v);
        }
        return rows;
    }

    /**
     * The codec face, media and tools included: text and {@link Part.Blob} parts fold through the
     * media-capable {@link #encodeTurn}; tools render the template's exact flow - declarations in
     * the system turn ({@code <|tool>declaration:...<tool|>}), and the whole call round-trip as ONE
     * open model turn: {@code <|tool_call>call:...<tool_call|>} then folded {@code
     * <|tool_response>response:...<tool_response|>} blocks then the answer text, with the
     * generation prompt suppressed while the model turn is open. Punts: {@link Part.Reasoning}
     * (thought channel not ported) and media on a text-only load.
     */
    @Override
    public List<Batch> encode(Conversation conversation) {
        requireSupported(conversation);
        List<Message> msgs = conversation.messages();
        List<Batch> out = new ArrayList<>(conversationStart());
        boolean systemFirst = !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM);
        int start = 0;
        if (systemFirst || !conversation.tools().isEmpty()) {
            systemBlock(systemFirst ? msgs.get(0) : null, conversation.tools(), out);
            if (systemFirst) start = 1;
        }
        // the template's per-message state: 'call'/'response' leave the model turn OPEN
        String prev = null;
        Role prevNonToolRole = null;
        for (int i = start; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) continue; // folded into its call turn below
            prev = null;
            boolean continuation =
                    m.role().equals(Role.ASSISTANT) && Role.ASSISTANT.equals(prevNonToolRole);
            prevNonToolRole = m.role();
            List<Part.ToolCall> calls =
                    m.content().stream()
                            .filter(p -> p instanceof Part.ToolCall)
                            .map(p -> (Part.ToolCall) p)
                            .toList();
            if (calls.isEmpty() && !continuation) {
                out.addAll(encodeTurn(m));
                continue;
            }
            if (m.content().stream().anyMatch(p -> p instanceof Part.Blob)) {
                throw new UnsupportedConversation("media in a tool-call/continuation model turn");
            }
            List<Integer> ids = new ArrayList<>();
            StringBuilder text = new StringBuilder();
            if (!continuation) {
                ids.add(turnOpen);
                text.append(roleName(m.role())).append('\n');
            }
            for (Part.ToolCall call : calls) {
                flushText(text, ids);
                ids.add(require("<|tool_call>"));
                sinkInto(text, ids, s -> Gemma4ToolSyntax.call(call.name(), call.arguments(), s));
                ids.add(require("<tool_call|>"));
            }
            // forward-fold the consecutive tool-role results, names resolved from the calls
            boolean responses = false;
            for (int j = i + 1; j < msgs.size() && msgs.get(j).role().equals(Role.TOOL); j++) {
                for (Part part : msgs.get(j).content()) {
                    if (!(part instanceof Part.ToolResult r)) continue;
                    flushText(text, ids);
                    ids.add(require("<|tool_response>"));
                    String name = resolveName(calls, r.callId());
                    sinkInto(text, ids, s -> Gemma4ToolSyntax.response(name, r.text(), s));
                    ids.add(require("<tool_response|>"));
                    responses = true;
                    prev = "response";
                }
            }
            if (!calls.isEmpty() && !responses) prev = "call";
            String content = m.text().strip(); // the lenient view: call parts render above
            text.append(content);
            flushText(text, ids);
            if ("call".equals(prev)) {
                ids.add(require("<|tool_response>")); // awaiting results: the turn stays open
            } else if (!(responses && content.isEmpty())) {
                ids.add(turnClose);
                ids.addAll(newline);
            }
            if (!ids.isEmpty()) out.add(Batch.prefill(ids));
        }
        if (!"call".equals(prev) && !"response".equals(prev)) {
            out.addAll(generationPrompt(conversation.thinking()));
        }
        return out;
    }

    /**
     * {@code <|turn>system\n} + trimmed system text + one {@code <|tool>declaration<tool|>} block
     * per tool + {@code <turn|>\n} - the template's tool-definitions block.
     */
    private void systemBlock(Message system, List<Tool> tools, List<Batch> out) {
        List<Integer> ids = new ArrayList<>();
        ids.add(turnOpen);
        StringBuilder text = new StringBuilder("system\n");
        if (system != null) text.append(system.textOnly().strip());
        flushText(text, ids);
        for (Tool tool : tools) {
            ids.add(require("<|tool>"));
            Object parsed = JsonCodec.parse(tool.rawJson());
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) parsed;
            sinkInto(text, ids, s -> Gemma4ToolSyntax.declaration(map, s));
            ids.add(require("<tool|>"));
        }
        ids.add(turnClose);
        ids.addAll(newline);
        out.add(Batch.prefill(ids));
    }

    /** Runs a tool-syntax renderer: text runs accumulate, quotes emit the trusted id. */
    private void sinkInto(
            StringBuilder text, List<Integer> ids, Consumer<Gemma4ToolSyntax.Sink> render) {
        render.accept(
                new Gemma4ToolSyntax.Sink() {
                    @Override
                    public void text(String s) {
                        text.append(s);
                    }

                    @Override
                    public void quote() {
                        flushText(text, ids);
                        ids.add(require("<|\"|>"));
                    }
                });
        flushText(text, ids);
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
                            default -> false; // Reasoning: thought channel not ported
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
        return generationPrompt;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    /** Gemma's template names the assistant turn {@code model}. */
    private static String roleName(Role role) {
        return role.equals(Role.ASSISTANT) ? "model" : role.name();
    }

    private void flushText(StringBuilder text, List<Integer> ids) {
        if (text.isEmpty()) return;
        ids.addAll(tokenizer.encode(text.toString()).toList());
        text.setLength(0);
    }

    @Override
    public ReplyParser parser() {
        return ReplyParser.spans(
                tokenizer, "<|tool_call>", "<tool_call|>", Gemma4ToolSyntax::parseBlock);
    }

    /** Forced calls pin {@code call:name} - the notation's plain-byte opening. */
    @Override
    public Optional<String> callGrammar(List<Tool> tools) {
        if (tools.isEmpty()) return Optional.empty();
        return Optional.of(ToolCallSyntax.prefixPinGbnf("call:", tools));
    }
}
