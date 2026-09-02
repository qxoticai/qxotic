package com.qxotic.jinfer.models.llama;

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
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Mistral/Ministral's reply codec: prompt framing punts to the GGUF template's whole-render; this
 * declares the reply language - a call is {@code [TOOL_CALLS] name [ARGS] args-json}, the name and
 * args as free holes around the interior mark, close-LESS: the region exits when the next call
 * opens, the reply ends, or the payload's balance completes.
 */
public final class MistralChatTemplate implements ChatTemplate {

    private final Tokenizer tokenizer;
    private final IntSequence promptStart;

    public MistralChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        promptStart = IntSequence.of(SpecialTokens.require(tokenizer, "<s>"));
    }

    @Override
    public IntSequence promptStart() {
        return promptStart;
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        throw new UnsupportedConversation("Mistral framing punts to the whole-render");
    }

    private ReplyLanguage.Selection autoReply; // memoized: tools-independent, built once

    @Override
    public synchronized ReplyParser parser(Tokenizer tokenizer) {
        if (autoReply == null) {
            autoReply = ReplyLanguage.Selection.of(language(), this.tokenizer);
        }
        return autoReply.walk();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String contentGbnf) {
        ReplyLanguage.Node tree =
                ReplyLanguage.seq(
                        ReplyLanguage.content(ReplyLanguage.gbnf(contentGbnf)),
                        ReplyLanguage.opt(ReplyLanguage.mark("</s>")));
        return Optional.of(ReplyLanguage.Selection.of(tree, tokenizer));
    }

    /** The family's ordinary free-content tree: {@code (content | call)* </s>?}. */
    private static ReplyLanguage.Node language() {
        return ReplyLanguage.seq(
                ReplyLanguage.rep(
                        ReplyLanguage.alt(
                                ReplyLanguage.content(ReplyLanguage.free()),
                                ReplyLanguage.call(
                                        MistralChatTemplate::walkCalls,
                                        ReplyLanguage.mark("[TOOL_CALLS]"),
                                        ReplyLanguage.free(),
                                        ReplyLanguage.mark("[ARGS]"),
                                        ReplyLanguage.free())),
                        0,
                        -1),
                ReplyLanguage.opt(ReplyLanguage.mark("</s>")));
    }

    /**
     * The forced-call language: the header is wholly forced ({@code [TOOL_CALLS]name[ARGS]} -
     * nothing is sampled until the arguments), the arguments are schema-bound. There is no free
     * region between the seed and the name for the model to derail in.
     */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        List<ReplyLanguage.Node> options = new ArrayList<>(callableTools.size());
        for (Tool tool : callableTools) {
            options.add(
                    ReplyLanguage.call(
                            MistralChatTemplate::walkCalls,
                            ReplyLanguage.mark("[TOOL_CALLS]"),
                            ReplyLanguage.bytes(tool.name()),
                            ReplyLanguage.mark("[ARGS]"),
                            ReplyLanguage.gbnf(Grammar.schemaGbnf(tool.parameters()))));
        }
        return Optional.of(
                ReplyLanguage.Selection.of(
                        ReplyLanguage.seq(
                                new ReplyLanguage.Node.Alt(options),
                                ReplyLanguage.opt(ReplyLanguage.mark("</s>"))),
                        tokenizer));
    }

    /**
     * The walk's payload is {@code name{json}} - the interior {@code [ARGS]} mark is excluded from
     * capture, so the name ends where the object begins. A payload without a parseable object is no
     * call.
     */
    static List<Content.ToolCall> walkCalls(String payload) {
        int brace = payload.indexOf('{');
        if (brace <= 0) return List.of();
        String name = payload.substring(0, brace).strip();
        if (name.isEmpty()) return List.of();
        Map<String, Object> arguments = ToolCallSyntax.parseObject(payload.substring(brace));
        return arguments == null ? List.of() : List.of(new Content.ToolCall("", name, arguments));
    }
}
