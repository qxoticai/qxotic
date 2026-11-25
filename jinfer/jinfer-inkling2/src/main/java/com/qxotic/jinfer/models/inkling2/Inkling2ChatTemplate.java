package com.qxotic.jinfer.models.inkling2;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.MediaEncodingCache;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.PromptWriter;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;
import java.util.function.Consumer;

/**
 * Inkling 2 chat framing, token-exact with the GGUF's Jinja chat_template. A conversation is a list
 * of blocks {@code {role}{name?}{content kind}{body}<|end_message|>}:
 *
 * <ul>
 *   <li>with tools, first {@code <|message_system|>tool_declare<|content_xml|>} and the
 *       declarations as compact JSON with sorted keys;
 *   <li>before the first turn that is not a system turn, a system text block {@code Thinking effort
 *       level: N}: 0.9 with thinking on (the template's default, "high"), 0 with it off;
 *   <li>a text block per turn; a tool result is headed {@code <|message_tool|>} and the name of the
 *       function it answers;
 *   <li>an image is a block of its own in its turn, {@code <|content_image|>} and its projected
 *       rows, and the text around it is a text block on each side, as llama.cpp splits a turn;
 *   <li>an assistant turn is its reasoning block, its text block and a block per call, {@code
 *       <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}}, then {@code
 *       <|content_model_end_sampling|>}.
 * </ul>
 *
 * There is no generation prompt: the model opens its own blocks, and a reply is that same list.
 */
final class Inkling2ChatTemplate implements ChatTemplate {
    static final String END_OF_TURN = "<|content_model_end_sampling|>";

    private static final String MODEL = "<|message_model|>";
    private static final String THINKING = "<|content_thinking|>";
    private static final String TEXT = "<|content_text|>";
    private static final String INVOKE = "<|content_invoke_tool_json|>";
    private static final String END = "<|end_message|>";
    private static final String NAME_GBNF = "root ::= [a-zA-Z0-9_.-]+";

    private final Tokenizer tokenizer;
    private final Multimodal media;
    private final int system, user, model, tool, thinking, text, image, xml, invoke, end, endOfTurn;
    private ReplyLanguage.Selection auto; // memoized: tools-independent, built once

    /** {@code media} supplies the vision tower; a model loaded without its mmproj has none. */
    Inkling2ChatTemplate(Tokenizer tokenizer, Multimodal media) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.media = Objects.requireNonNull(media, "media");
        system = SpecialTokens.require(tokenizer, "<|message_system|>");
        user = SpecialTokens.require(tokenizer, "<|message_user|>");
        model = SpecialTokens.require(tokenizer, MODEL);
        tool = SpecialTokens.require(tokenizer, "<|message_tool|>");
        thinking = SpecialTokens.require(tokenizer, THINKING);
        text = SpecialTokens.require(tokenizer, TEXT);
        image = SpecialTokens.require(tokenizer, "<|content_image|>");
        xml = SpecialTokens.require(tokenizer, "<|content_xml|>");
        invoke = SpecialTokens.require(tokenizer, INVOKE);
        end = SpecialTokens.require(tokenizer, END);
        endOfTurn = SpecialTokens.require(tokenizer, END_OF_TURN);
    }

    /**
     * A reasoning block is closed by the mark that closes every block, so with thinking off only
     * {@code <|content_thinking|>} is masked.
     */
    @Override
    public ThinkMarkers thinkMarkers() {
        return new ThinkMarkers(THINKING, END, true);
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        return encode(conversation, batchCapacity, null, sink);
    }

    @Override
    public ReplyState encode(
            Conversation conversation,
            int batchCapacity,
            MediaEncodingCache mediaCache,
            Consumer<Batch> sink) {
        Objects.requireNonNull(conversation, "conversation");
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, mediaCache, sink);
        if (!conversation.tools().isEmpty()) {
            List<Object> specs = new ArrayList<>();
            for (Tool declared : conversation.tools()) specs.add(spec(declared));
            out.id(system).text("tool_declare").id(xml).text(JsonCodec.stringify(specs)).id(end);
            out.flush();
        }
        String effort = "Thinking effort level: " + (conversation.thinking() ? "0.9" : "0");
        boolean effortWritten = false;
        List<Content.ToolCall> open = new ArrayList<>(); // the calls no result has answered yet
        for (Message m : conversation.messages()) {
            if (!effortWritten && !m.role().equals(Role.SYSTEM)) {
                out.id(system).id(text).text(effort).id(end);
                effortWritten = true;
            }
            if (m.role().equals(Role.ASSISTANT)) {
                writeAssistant(out, m, open);
            } else if (m.role().equals(Role.TOOL)) {
                writeResults(out, m, open);
            } else if (m.role().equals(Role.SYSTEM)) {
                requireParts(m, Content.Text.class);
                out.id(system).id(text).text(m.text()).id(end);
            } else if (m.role().equals(Role.USER)) {
                writeUser(out, m);
            } else {
                throw new UnsupportedConversation("role " + m.role().name());
            }
            out.flush();
        }
        if (!effortWritten) out.id(system).id(text).text(effort).id(end);
        out.finish();
        return new ReplyState(IntSequence.empty(), parser(tokenizer));
    }

    /** A block per image and per run of text. */
    private void writeUser(PromptWriter out, Message m) {
        requireParts(m, Content.Text.class, Content.Media.class);
        List<Content> parts = m.content();
        for (int i = 0; i < parts.size(); ) {
            if (parts.get(i) instanceof Content.Media part) {
                Media.Image picture = image(part.media());
                MediaProjector<Media.Image> tower = tower();
                out.cachedMedia(
                        picture,
                        part.key(),
                        encoded ->
                                encoded.id(user)
                                        .id(image)
                                        .media(picture, part.key(), tower, false)
                                        .id(end));
                i++;
                continue;
            }
            StringBuilder run = new StringBuilder();
            for (; i < parts.size() && parts.get(i) instanceof Content.Text part; i++)
                run.append(part.text());
            out.id(user).id(text).text(run.toString()).id(end);
        }
    }

    private static Media.Image image(Media value) {
        if (value instanceof Media.Image picture) return picture;
        throw new UnsupportedConversation(
                "Inkling 2 takes no " + value.getClass().getSimpleName().toLowerCase(Locale.ROOT));
    }

    private MediaProjector<Media.Image> tower() {
        return media.projector(Media.Image.class)
                .orElseThrow(
                        () -> new UnsupportedConversation("image input needs the model's mmproj"));
    }

    @Override
    public int mediaPositions(Media value) {
        return tower().positions(image(value));
    }

    /** {@code description}, {@code name}, {@code parameters} and {@code type}, keys sorted. */
    private static Map<String, Object> spec(Tool declared) {
        Map<String, Object> function = ToolCallSyntax.functionObject(declared);
        Map<String, Object> spec = new TreeMap<>();
        spec.put("description", function.get("description") instanceof String s ? s : "");
        spec.put("name", declared.name());
        spec.put(
                "parameters",
                function.get("parameters") instanceof Map<?, ?> p ? sorted(p) : Map.of());
        spec.put("type", declared.definition().get("type") instanceof String s ? s : "function");
        return spec;
    }

    /** Python's {@code sort_keys}: every object of a JSON value in key order. */
    private static Object sorted(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> out = new TreeMap<>();
            map.forEach((key, v) -> out.put(String.valueOf(key), sorted(v)));
            return out;
        }
        if (value instanceof List<?> list)
            return list.stream().map(Inkling2ChatTemplate::sorted).toList();
        return value;
    }

    /**
     * Reasoning, text, then the calls. The text block is written even when empty unless the turn
     * has reasoning or calls, as the template does.
     */
    private void writeAssistant(PromptWriter out, Message m, List<Content.ToolCall> open) {
        requireParts(m, Content.Text.class, Content.Reasoning.class, Content.ToolCall.class);
        String reasoning = m.reasoning(), content = m.text();
        boolean calls = m.content().stream().anyMatch(Content.ToolCall.class::isInstance);
        if (!reasoning.isEmpty()) out.id(model).id(thinking).text(reasoning).id(end);
        if (!content.isEmpty() || !calls && reasoning.isEmpty())
            out.id(model).id(text).text(content).id(end);
        for (Content part : m.content()) {
            if (!(part instanceof Content.ToolCall call)) continue;
            open.add(call);
            Map<String, Object> envelope = new LinkedHashMap<>();
            envelope.put("name", call.name());
            envelope.put("args", new TreeMap<>(call.arguments()));
            out.id(model).text(call.name()).id(invoke).text(JsonCodec.stringify(envelope)).id(end);
        }
        out.id(endOfTurn);
    }

    /**
     * A text block per result, headed by the name of the call it answers. A conversation holds no
     * result without an open call.
     */
    private void writeResults(PromptWriter out, Message m, List<Content.ToolCall> open) {
        requireParts(m, Content.ToolResult.class);
        for (Content part : m.content()) {
            if (!(part instanceof Content.ToolResult result)) continue;
            // the call with the result's id, else the oldest one still open
            Content.ToolCall answered = open.getFirst();
            for (Content.ToolCall call : open)
                if (!result.callId().isEmpty() && call.id().equals(result.callId()))
                    answered = call;
            open.remove(answered);
            out.id(tool).text(answered.name()).id(text).text(result.text()).id(end);
        }
    }

    /** The part kinds a turn is framed from; anything else is refused. */
    private static void requireParts(Message m, Class<?>... kinds) {
        for (Content part : m.content())
            if (Arrays.stream(kinds).noneMatch(kind -> kind.isInstance(part)))
                throw new UnsupportedConversation(
                        m.role().name() + " turn: " + part.getClass().getSimpleName());
    }

    @Override
    public synchronized ReplyParser parser(Tokenizer tokenizer) {
        if (auto == null)
            auto = ReplyLanguage.Selection.of(reply(ReplyLanguage.free(), true), this.tokenizer);
        return auto.walk();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String grammar) {
        return constrainedReply(grammar, false);
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String grammar, boolean calls) {
        return Optional.of(
                ReplyLanguage.Selection.of(reply(ReplyLanguage.gbnf(grammar), calls), tokenizer));
    }

    /**
     * The reply: blocks, then the end of the turn. Reasoning is always free and {@code document} is
     * the body of a text block. Without {@code calls} a text block is required, after the
     * reasoning.
     */
    private static ReplyLanguage.Node reply(ReplyLanguage.Node document, boolean calls) {
        ReplyLanguage.Node header = ReplyLanguage.mark(MODEL), end = ReplyLanguage.mark(END);
        ReplyLanguage.Node reasoning =
                ReplyLanguage.think(
                        header, ReplyLanguage.mark(THINKING), ReplyLanguage.free(), end);
        ReplyLanguage.Node content =
                ReplyLanguage.content(header, ReplyLanguage.mark(TEXT), document, end);
        ReplyLanguage.Node call =
                ReplyLanguage.call(
                        Inkling2ChatTemplate::parseCall,
                        header,
                        ReplyLanguage.gbnf(NAME_GBNF),
                        ReplyLanguage.mark(INVOKE),
                        ReplyLanguage.free(),
                        end);
        ReplyLanguage.Node endOfTurn = ReplyLanguage.opt(ReplyLanguage.mark(END_OF_TURN));
        return calls
                ? ReplyLanguage.seq(
                        ReplyLanguage.rep(ReplyLanguage.alt(reasoning, content, call), 0, -1),
                        endOfTurn)
                : ReplyLanguage.seq(ReplyLanguage.rep(reasoning, 0, -1), content, endOfTurn);
    }

    /** Forced calls: the header carries an offered name and the schema binds the arguments. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        List<ReplyLanguage.Node> options = new ArrayList<>(callableTools.size());
        for (Tool offered : callableTools)
            options.add(
                    ReplyLanguage.call(
                            Inkling2ChatTemplate::parseCall,
                            ReplyLanguage.mark(MODEL),
                            ReplyLanguage.bytes(offered.name()),
                            ReplyLanguage.mark(INVOKE),
                            ReplyLanguage.bytes(
                                    "{\"name\":"
                                            + JsonCodec.stringify(offered.name())
                                            + ",\"args\":"),
                            ReplyLanguage.gbnf(Grammar.schemaGbnf(offered.parameters())),
                            ReplyLanguage.bytes("}"),
                            ReplyLanguage.mark(END)));
        return Optional.of(
                ReplyLanguage.Selection.of(
                        ReplyLanguage.seq(
                                new ReplyLanguage.Node.Alt(options),
                                ReplyLanguage.opt(ReplyLanguage.mark(END_OF_TURN))),
                        tokenizer));
    }

    /**
     * A call block's text, {@code NAME{"name":"NAME","args":{...}}}: the envelope names the call. A
     * payload without a named envelope is no call, and the reply continues.
     */
    @SuppressWarnings("unchecked")
    static List<Content.ToolCall> parseCall(String payload) {
        int brace = payload.indexOf('{');
        Map<String, Object> envelope =
                brace < 0 ? null : ToolCallSyntax.parseObject(payload.substring(brace));
        if (envelope == null || !(envelope.get("name") instanceof String name) || name.isEmpty())
            return List.of();
        Object args = envelope.get("args");
        return List.of(
                new Content.ToolCall(
                        "",
                        name,
                        args instanceof Map<?, ?> map ? (Map<String, Object>) map : Map.of()));
    }
}
