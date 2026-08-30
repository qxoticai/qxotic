package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Granite's reply codec: prompt framing punts to the GGUF template's whole-render; this declares
 * the reply language - an optional {@code <think>} span, content and {@code <tool_call>} spans
 * interleaved - so the fallback keeps the checkpoint's call parsing, constrained decoding and
 * forced calls. Granite 4.1 carries JSON call envelopes and {@code <|end_of_text|>}; Granite 4.2
 * carries function/parameter XML and {@code <|im_end|>}.
 */
public final class GraniteChatTemplate implements ChatTemplate {

    private final Tokenizer tokenizer;
    private final boolean functionXml;

    public GraniteChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.functionXml = SpecialTokens.find(tokenizer, "<|im_end|>").isPresent();
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        throw new UnsupportedConversation("Granite framing punts to the whole-render");
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

    /** Forced calls: the wire carries an offered name; the model supplies its arguments. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        if (functionXml) {
            return Optional.of(
                    spans().forcedCall(callableTools, tool -> "\n<function=" + tool.name()));
        }
        return Optional.of(
                ReplyLanguage.Selection.of(
                        JsonEnvelopeReplies.forced(callableTools, "<|end_of_text|>"), tokenizer));
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
                            functionXml
                                    ? ToolCallSyntax::parseFunctionXml
                                    : ToolCallSyntax::parseBlock,
                            ReplyLanguage.mark(functionXml ? "<|im_end|>" : "<|end_of_text|>"),
                            tokenizer);
        }
        return spans;
    }
}
