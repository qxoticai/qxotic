package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Hand-written gpt-oss (Harmony) chat framing, matching the GGUF chat_template's plain-conversation
 * shape.
 *
 * <p>Layout: a fixed system preamble once ({@code <|start|>system<|message|>{identity, cutoff,
 * date, reasoning, channels}<|end|>}), the conversation's system message as a developer block
 * ({@code <|start|>developer<|message|># Instructions\n\n{content}<|end|>}), then per turn {@code
 * <|start|>user<|message|>{content}<|end|>} and, for assistant history, {@code
 * <|start|>assistant<|channel|>final<|message|>{content}<|end|>} (the template drops CoT from
 * history - only the final channel is re-rendered). Generation prompt is {@code
 * <|start|>assistant}; the model then emits its own channel tokens ({@code
 * <|channel|>analysis<|message|>...}), so {@code thinking} is a no-op here - Harmony always
 * reasons, and the effort knob lives in the system preamble ({@code Reasoning: medium}).
 *
 * <p>Each text run between specials is ONE contiguous plain {@link Tokenizer#encode} (that is how a
 * rendered template tokenizes; specials force the only splits), and conversation content never goes
 * through special-aware encoding, so text cannot mint control tokens.
 *
 * <p>The preamble embeds the current date ({@code strftime_now("%Y-%m-%d")} in the template); it is
 * a constructor argument so tests are deterministic - the convenience constructor pins today,
 * matching what the template renders.
 */
public final class GptOssTurnTemplate implements TurnTemplate {

    static final String DEFAULT_IDENTITY =
            "You are ChatGPT, a large language model trained by OpenAI.";
    static final String DEFAULT_EFFORT = "medium";

    /** Appended to the system preamble when tools are offered (the template's tools branch). */
    static final String TOOLS_LINE =
            "\nCalls to these tools must go to the commentary channel: 'functions'.";

    private final Tokenizer tokenizer;
    private final List<Batch> conversationStart; // fixed preamble, encoded once
    private final Batch toolsPreamble; // the preamble + routing line, encoded once
    private final int start; // <|start|>
    private final int message; // <|message|>
    private final int channel; // <|channel|>
    private final int end; // <|end|>
    private final int call; // <|call|>
    private final TokenRuns proto; // compiled spelling table, forked per block/turn

    public GptOssTurnTemplate(Tokenizer tokenizer) {
        this(tokenizer, LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
    }

    public GptOssTurnTemplate(Tokenizer tokenizer, String currentDate) {
        this.tokenizer = tokenizer;
        this.start = SpecialTokens.require(tokenizer, "<|start|>");
        this.message = SpecialTokens.require(tokenizer, "<|message|>");
        this.channel = SpecialTokens.require(tokenizer, "<|channel|>");
        this.end = SpecialTokens.require(tokenizer, "<|end|>");
        this.call = SpecialTokens.require(tokenizer, "<|call|>");
        this.proto = new TokenRuns(tokenizer);
        String systemText =
                DEFAULT_IDENTITY
                        + "\n"
                        + "Knowledge cutoff: 2024-06\n"
                        + "Current date: "
                        + currentDate
                        + "\n\n"
                        + "Reasoning: "
                        + DEFAULT_EFFORT
                        + "\n\n"
                        + "# Valid channels: analysis, commentary, final. Channel must be included"
                        + " for every message.";
        this.conversationStart = List.of(block("system", systemText));
        this.toolsPreamble = block("system", systemText + TOOLS_LINE);
    }

    /** {@code <|start|>{role}<|message|>{body}<|end|>} - one channel-less block. */
    private Batch block(String role, String body) {
        return proto.fresh().id(start).text(role).id(message).text(body).id(end).batch();
    }

    /**
     * {@code <|start|>{header}<|channel|>{channelName}<|message|>{body}{close}} - one channeled
     * turn. The body splices {@code verbatim} generated ids when present (model-exact bytes),
     * otherwise re-encodes {@code body} plainly.
     */
    private Batch turn(
            String header, String channelName, String body, IntSequence verbatim, int close) {
        TokenRuns runs =
                proto.fresh().id(start).text(header).id(channel).text(channelName).id(message);
        if (verbatim != null) runs.verbatim(verbatim);
        else runs.text(body);
        return runs.id(close).batch();
    }

    /** The fixed Harmony system preamble: {@code <|start|>system<|message|>{...}<|end|>}. */
    @Override
    public List<Batch> conversationStart() {
        return conversationStart;
    }

    @Override
    public List<Batch> encodeTurn(Message m) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        if (m.role().equals(Role.SYSTEM)) { // conversation system -> developer block
            ids.addAll(tokenizer.encode("developer"));
            ids.add(message);
            ids.addAll(tokenizer.encode("# Instructions\n\n" + m.textOnly()));
        } else if (m.role().equals(Role.ASSISTANT)) { // history keeps only the final channel
            ids.addAll(tokenizer.encode("assistant"));
            ids.add(channel);
            ids.addAll(tokenizer.encode("final"));
            ids.add(message);
            ids.addAll(tokenizer.encode(m.textOnly()));
        } else {
            ids.addAll(tokenizer.encode(m.role().name()));
            ids.add(message);
            ids.addAll(tokenizer.encode(m.textOnly()));
        }
        ids.add(end);
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    /** {@code <|start|>assistant} - the model emits its own channel tokens from here. */
    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        ids.addAll(tokenizer.encode("assistant"));
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    /** Closes the open message: {@code <|end|>} (the {@code <|return|>} stop is never ingested). */
    @Override
    public List<Batch> closeTurn() {
        return List.of(Batch.prefill(new int[] {end}));
    }

    /**
     * The codec face, tools included - the template's whole-conversation flow: the system preamble
     * grows the commentary-routing line when tools are offered; the developer block carries {@code
     * # Instructions} and/or the {@code # Tools} TypeScript namespace; an assistant call turn is
     * {@code <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{args}
     * <|call|>}, preceded by its analysis message only while no later final answer exists; a tool
     * result is {@code <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>
     * {content}<|end|>}, its name resolved from the most recent call (the template's max-one-call-
     * per-message assumption; extra calls in one turn are not rendered, matching it). Reasoning on
     * a plain assistant turn is dropped from history (the template's stated CoT rule).
     */
    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        List<Tool> tools = conversation.tools();
        TurnTemplate.requireToolShapes(msgs);
        List<Batch> out = new ArrayList<>();
        if (tools.isEmpty()) out.addAll(conversationStart());
        else out.add(toolsPreamble);
        // the answered-round-trip boundary: calls at or after it keep their analysis message
        int lastFinal = -1;
        for (int i = 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.ASSISTANT)
                    && m.content().stream().noneMatch(p -> p instanceof Part.ToolCall)) {
                lastFinal = i;
            }
        }
        Message dev = Message.leadingSystem(msgs);
        String devText = dev == null ? "" : dev.textOnly();
        if (!devText.isEmpty() || !tools.isEmpty()) {
            StringBuilder body = new StringBuilder();
            if (!devText.isEmpty()) body.append("# Instructions\n\n").append(devText);
            if (!tools.isEmpty()) {
                if (!devText.isEmpty()) body.append("\n\n");
                body.append("# Tools\n\n").append(GptOssToolSyntax.namespace(tools));
            }
            out.add(block("developer", body.toString()));
        }
        String lastCall = null;
        for (int i = dev == null ? 0 : 1; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) {
                if (lastCall == null)
                    throw new UnsupportedConversation("tool result without a preceding call");
                StringBuilder body = new StringBuilder();
                for (Part p : m.content()) {
                    if (p instanceof Part.ToolResult r) body.append(r.text());
                }
                out.add(
                        turn(
                                "functions." + lastCall + " to=assistant",
                                "commentary",
                                body.toString(),
                                null,
                                end));
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                out.addAll(encodeTurn(m));
                continue;
            }
            Part.ToolCall first = null;
            for (Part p : m.content()) {
                if (p instanceof Part.ToolCall c) {
                    first = c;
                    break;
                }
            }
            if (first == null) { // history keeps only the final channel; CoT is dropped
                out.addAll(encodeTurn(new Message(Role.ASSISTANT, m.text())));
                lastCall = null;
                continue;
            }
            String content = m.text();
            Part.Reasoning thinking = m.reasoning();
            if (!content.isEmpty() && thinking != null)
                throw new UnsupportedConversation(
                        "assistant call turn with both content and thinking");
            if ((!content.isEmpty() || thinking != null) && i > lastFinal) {
                // the one place inference retains CoT: an unanswered tool-call round-trip
                String analysis = !content.isEmpty() ? content : thinking.text();
                out.add(
                        turn(
                                "assistant",
                                "analysis",
                                analysis,
                                thinking == null ? null : thinking.verbatim(),
                                end));
            }
            out.add(
                    turn(
                            "assistant to=functions." + first.name(),
                            "commentary json",
                            ToolCallSyntax.jinjaJson(first.arguments()),
                            first.verbatim(),
                            call));
            lastCall = first.name();
        }
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }

    // ---- the reply language: parse, constrain and force from ONE definition ----

    private static final String NAME_GBNF = "root ::= [a-zA-Z0-9_.-]+";

    private ReplyLanguage.Selection autoReply; // memoized: tools-independent, built once

    /**
     * The AUTO walk over the canonical Harmony reply language: analysis, preamble, final and call
     * messages behind the shared {@code <|channel|>} opener (candidacy splits them on the channel
     * name), each also reachable through a re-opened {@code <|start|>assistant} header, message
     * separators consumed at structure level. Argument bodies are FREE holes whose parser drops a
     * malformed payload without ending the reply - the old parser's leniency, kept.
     */
    @Override
    public ReplyParser parser() {
        if (autoReply == null) {
            autoReply =
                    ReplyLanguage.Selection.of(harmonyLanguage(ReplyLanguage.free()), tokenizer);
        }
        return autoReply.walk();
    }

    /**
     * The FINAL channel takes the hole; analysis and preamble stay free (the channel-scoping law).
     */
    @Override
    public Optional<ReplyLanguage.Selection> constrainedAuto(String contentGbnf) {
        return Optional.of(
                ReplyLanguage.Selection.of(
                        harmonyLanguage(ReplyLanguage.gbnf(contentGbnf)), tokenizer));
    }

    private ReplyLanguage.Node harmonyLanguage(ReplyLanguage.Node contentHole) {
        ReplyLanguage.Node sep =
                ReplyLanguage.alt(
                        ReplyLanguage.mark("<|end|>"),
                        ReplyLanguage.mark("<|call|>"),
                        ReplyLanguage.mark("<|return|>"));
        List<ReplyLanguage.Node> shapes = new ArrayList<>();
        for (boolean reopened : new boolean[] {false, true}) {
            shapes.add(
                    message(
                            reopened,
                            ReplyLanguage.Kind.THINK,
                            "analysis",
                            null,
                            ReplyLanguage.free()));
            shapes.add(message(reopened, ReplyLanguage.Kind.CONTENT, "final", null, contentHole));
            shapes.add(
                    message(
                            reopened,
                            ReplyLanguage.Kind.CONTENT,
                            "commentary",
                            null,
                            ReplyLanguage.free()));
            shapes.add(
                    message(
                            reopened,
                            ReplyLanguage.Kind.CALL,
                            null,
                            GptOssTurnTemplate::harmonyCalls,
                            ReplyLanguage.free()));
        }
        ReplyLanguage.Node msg = new ReplyLanguage.Node.Alt(shapes);
        return ReplyLanguage.rep(ReplyLanguage.seq(msg, ReplyLanguage.opt(sep)), 0, -1);
    }

    /** One canonical Harmony message shape; {@code channelName} null = the call header. */
    private ReplyLanguage.Node message(
            boolean reopened,
            ReplyLanguage.Kind kind,
            String channelName,
            java.util.function.Function<String, List<Part.ToolCall>> calls,
            ReplyLanguage.Node bodyHole) {
        List<ReplyLanguage.Node> body = new ArrayList<>();
        if (reopened) {
            body.add(ReplyLanguage.mark("<|start|>"));
            body.add(ReplyLanguage.bytes("assistant"));
        }
        body.add(ReplyLanguage.mark("<|channel|>"));
        if (channelName != null) {
            body.add(ReplyLanguage.bytes(channelName));
        } else {
            // ANY dotted recipient, not only functions.*: browser.search and friends are legal
            // Harmony - the payload parser filters, so a non-functions call drops silently
            // instead of ending the reply
            body.add(ReplyLanguage.bytes("commentary to="));
            body.add(ReplyLanguage.gbnf(NAME_GBNF));
        }
        // the constrain adornment (" <|constrain|>json", space optional - the old parser
        // defended the no-space variant explicitly) appears on CALL headers and, under a JSON
        // response format, on final/analysis headers too
        body.add(
                ReplyLanguage.opt(
                        ReplyLanguage.seq(
                                ReplyLanguage.opt(ReplyLanguage.bytes(" ")),
                                ReplyLanguage.mark("<|constrain|>"),
                                ReplyLanguage.bytes("json"))));
        body.add(ReplyLanguage.mark("<|message|>"));
        body.add(bodyHole); // region-final: the body closes on any control token (or accepts)
        return new ReplyLanguage.Node.Region(kind, calls, body);
    }

    /**
     * The forced-call language: per-tool call regions with SCHEMA-BOUND argument grammars - a
     * forced call can neither name an unoffered tool nor malform its payload. Replaces the
     * seed/pin/epilogue recipe whose free argument region failed roughly one REQUIRED run in three
     * on the 20B.
     */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> tools) {
        if (tools.isEmpty()) return Optional.empty();
        List<ReplyLanguage.Node> options = new ArrayList<>(tools.size());
        for (Tool tool : tools) {
            Map<String, Object> schema = tool.parameters();
            options.add(
                    ReplyLanguage.call(
                            GptOssTurnTemplate::harmonyCalls,
                            ReplyLanguage.mark("<|channel|>"),
                            ReplyLanguage.bytes("commentary to=functions." + tool.name() + " "),
                            ReplyLanguage.mark("<|constrain|>"),
                            ReplyLanguage.bytes("json"),
                            ReplyLanguage.mark("<|message|>"),
                            ReplyLanguage.gbnf(Grammar.schemaGbnf(schema))));
        }
        return Optional.of(
                ReplyLanguage.Selection.of(
                        ReplyLanguage.seq(
                                new ReplyLanguage.Node.Alt(options),
                                ReplyLanguage.opt(ReplyLanguage.mark("<|call|>"))),
                        tokenizer));
    }

    /**
     * The payload parser both languages share: {@code commentary to=functions.NAME [json]{args}}
     * captured as text, the args the FIRST JSON object. A payload that yields no name or no object
     * is NO CALL - the region completed, the reply continues (the old parser's malformed-drop
     * semantics).
     */
    static List<Part.ToolCall> harmonyCalls(String payload) {
        int at = payload.indexOf("to=functions.");
        if (at < 0) return List.of();
        int from = at + "to=functions.".length();
        int to = from;
        while (to < payload.length()
                && !Character.isWhitespace(payload.charAt(to))
                && payload.charAt(to) != '{') {
            to++;
        }
        String name = payload.substring(from, to);
        int brace = payload.indexOf('{', to);
        if (name.isEmpty() || brace < 0) return List.of();
        try {
            if (JsonCodec.parse(payload.substring(brace)) instanceof Map<?, ?> args) {
                @SuppressWarnings("unchecked")
                Map<String, Object> arguments = (Map<String, Object>) args;
                return List.of(new Part.ToolCall("", name, arguments));
            }
        } catch (RuntimeException malformed) {
            // a payload that never held a parseable object is no call
        }
        return List.of();
    }
}
