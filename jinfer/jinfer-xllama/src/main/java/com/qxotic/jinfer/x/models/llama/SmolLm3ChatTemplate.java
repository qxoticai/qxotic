package com.qxotic.jinfer.x.models.llama;

import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.chat.ChatTemplate;
import com.qxotic.jinfer.x.chat.Conversation;
import com.qxotic.jinfer.x.chat.ReplyLanguage;
import com.qxotic.jinfer.x.chat.ReplyParser;
import com.qxotic.jinfer.x.chat.Tool;
import com.qxotic.jinfer.x.chat.ToolCallSyntax;
import com.qxotic.jinfer.x.chat.UnsupportedConversation;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * SmolLM3's reply codec: prompt framing punts to the GGUF template's whole-render; this declares
 * the reply language - an optional {@code <think>} span, content and {@code <tool_call>} JSON
 * envelopes interleaved, terminated by {@code <|im_end|>} - so the fallback keeps the family's call
 * parsing, constrained decoding and forced calls.
 */
public final class SmolLm3ChatTemplate implements ChatTemplate {

    private final Tokenizer tokenizer;

    public SmolLm3ChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        throw new UnsupportedConversation("SmolLM3 framing punts to the whole-render");
    }

    @Override
    public ReplyParser parser(Tokenizer tokenizer) {
        return spans().parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(
            String contentGbnf, List<Tool> callableTools) {
        return Optional.of(spans().constrainedAuto(contentGbnf, !callableTools.isEmpty()));
    }

    /** Forced calls: the envelope carries an OFFERED name, the schema binds the arguments. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(
                ReplyLanguage.Selection.of(
                        JsonEnvelopeReplies.forced(callableTools, "<|im_end|>"), tokenizer));
    }

    private ReplyLanguage.Spans spans; // the family's derived faces, markers written once

    private ReplyLanguage.Spans spans() {
        if (spans == null) {
            spans =
                    new ReplyLanguage.Spans(
                            "<think>",
                            "</think>",
                            "<tool_call>",
                            "</tool_call>",
                            ToolCallSyntax::parseBlock,
                            ReplyLanguage.mark("<|im_end|>"),
                            tokenizer);
        }
        return spans;
    }
}
