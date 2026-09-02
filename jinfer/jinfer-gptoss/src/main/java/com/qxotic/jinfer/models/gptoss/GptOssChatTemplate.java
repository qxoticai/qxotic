package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * gpt-oss's Harmony reply codec: prompt framing punts to the GGUF template's whole-render; this
 * declares the reply language - analysis, preamble, final and call messages behind the shared
 * {@code <|channel|>} opener (candidacy splits them on the channel name), each also reachable
 * through a re-opened {@code <|start|>assistant} header, message separators consumed at structure
 * level. Argument bodies are FREE holes whose parser drops a malformed payload without ending the
 * reply.
 */
public final class GptOssChatTemplate implements ChatTemplate {

    private static final String NAME_GBNF = "root ::= [a-zA-Z0-9_.-]+";

    private final Tokenizer tokenizer;

    public GptOssChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        throw new UnsupportedConversation("gpt-oss framing punts to the whole-render");
    }

    private ReplyLanguage.Selection autoReply; // memoized: tools-independent, built once

    @Override
    public synchronized ReplyParser parser(Tokenizer tokenizer) {
        if (autoReply == null) {
            autoReply =
                    ReplyLanguage.Selection.of(harmonyLanguage(ReplyLanguage.free()), tokenizer);
        }
        return autoReply.walk();
    }

    /** The FINAL channel takes the hole; analysis and preamble stay free (channel-scoping). */
    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String contentGbnf) {
        return Optional.of(
                ReplyLanguage.Selection.of(
                        harmonyLanguage(ReplyLanguage.gbnf(contentGbnf), false), tokenizer));
    }

    private ReplyLanguage.Node harmonyLanguage(ReplyLanguage.Node contentHole) {
        return harmonyLanguage(contentHole, true);
    }

    private ReplyLanguage.Node harmonyLanguage(ReplyLanguage.Node contentHole, boolean allowCalls) {
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
            shapes.add(
                    message(
                            reopened,
                            ReplyLanguage.Kind.CONTENT,
                            "commentary",
                            null,
                            ReplyLanguage.free()));
            if (allowCalls) {
                shapes.add(
                        message(reopened, ReplyLanguage.Kind.CONTENT, "final", null, contentHole));
                shapes.add(
                        message(
                                reopened,
                                ReplyLanguage.Kind.CALL,
                                null,
                                GptOssChatTemplate::harmonyCalls,
                                ReplyLanguage.free()));
            }
        }
        ReplyLanguage.Node msg = new ReplyLanguage.Node.Alt(shapes);
        ReplyLanguage.Node stream =
                ReplyLanguage.rep(ReplyLanguage.seq(msg, ReplyLanguage.opt(sep)), 0, -1);
        if (allowCalls) return stream;
        // tool-less: reasoning and preambles stay free, then ONE final message is REQUIRED -
        // an empty reply must not comply, and there are no calls to take instead
        ReplyLanguage.Node requiredFinal =
                ReplyLanguage.alt(
                        message(false, ReplyLanguage.Kind.CONTENT, "final", null, contentHole),
                        message(true, ReplyLanguage.Kind.CONTENT, "final", null, contentHole));
        return ReplyLanguage.seq(stream, requiredFinal, ReplyLanguage.opt(sep));
    }

    /** One canonical Harmony message shape; {@code channelName} null = the call header. */
    private ReplyLanguage.Node message(
            boolean reopened,
            ReplyLanguage.Kind kind,
            String channelName,
            Function<String, List<Content.ToolCall>> calls,
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
        // the constrain adornment (" <|constrain|>json", space optional) appears on CALL headers
        // and, under a JSON response format, on final/analysis headers too
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
     * forced call can neither name an unoffered tool nor malform its payload.
     */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        List<ReplyLanguage.Node> options = new ArrayList<>(callableTools.size());
        for (Tool tool : callableTools) {
            options.add(
                    ReplyLanguage.call(
                            GptOssChatTemplate::harmonyCalls,
                            ReplyLanguage.mark("<|channel|>"),
                            ReplyLanguage.bytes("commentary to=functions." + tool.name() + " "),
                            ReplyLanguage.mark("<|constrain|>"),
                            ReplyLanguage.bytes("json"),
                            ReplyLanguage.mark("<|message|>"),
                            ReplyLanguage.gbnf(Grammar.schemaGbnf(tool.parameters()))));
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
     * is NO CALL - the region completed, the reply continues.
     */
    static List<Content.ToolCall> harmonyCalls(String payload) {
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
        Map<String, Object> arguments = ToolCallSyntax.parseObject(payload.substring(brace));
        return arguments == null ? List.of() : List.of(new Content.ToolCall("", name, arguments));
    }
}
