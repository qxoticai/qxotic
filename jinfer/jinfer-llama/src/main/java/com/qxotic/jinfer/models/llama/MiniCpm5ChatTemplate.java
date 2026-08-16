package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * MiniCPM5's reply codec: prompt framing punts to the GGUF template's whole-render; this declares
 * the reply language - an optional {@code <think>} span, content and {@code <function} XML spans
 * interleaved, terminated by {@code <|im_end|>}. The {@code </param>} closers are SPECIALS inside
 * the payload, and a marker-pair call span claims interior control tokens AS THEIR SPELLINGS -
 * exactly the decoded text {@link MiniCpmToolSyntax#parsePayload} expects.
 */
public final class MiniCpm5ChatTemplate implements ChatTemplate {

    private final Tokenizer tokenizer;

    public MiniCpm5ChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        throw new UnsupportedConversation("MiniCPM5 framing punts to the whole-render");
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

    /** Forced calls: the header carries an OFFERED name, the arguments stay the model's own. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(spans().forcedCall(callableTools, tool -> " name=\"" + tool.name()));
    }

    private ReplyLanguage.Spans spans; // the family's derived faces, markers written once

    private ReplyLanguage.Spans spans() {
        if (spans == null) {
            spans =
                    new ReplyLanguage.Spans(
                            "<think>",
                            "</think>",
                            "<function",
                            "</function>",
                            MiniCpmToolSyntax::parsePayload,
                            ReplyLanguage.mark("<|im_end|>"),
                            tokenizer);
        }
        return spans;
    }
}
