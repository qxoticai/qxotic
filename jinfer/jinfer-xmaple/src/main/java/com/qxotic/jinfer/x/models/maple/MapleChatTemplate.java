package com.qxotic.jinfer.x.models.maple;

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

/** Maple's ChatML reply codec; GGUF Jinja remains the prompt-framing authority. */
public final class MapleChatTemplate implements ChatTemplate {
    private final Tokenizer tokenizer;
    private ReplyLanguage.Spans spans;

    public MapleChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        throw new UnsupportedConversation("Maple framing punts to the whole-render");
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

    private ReplyLanguage.Spans spans() {
        if (spans == null)
            spans =
                    new ReplyLanguage.Spans(
                            "<think>",
                            "</think>",
                            "<tool_call>",
                            "</tool_call>",
                            ToolCallSyntax::parseBlock,
                            ReplyLanguage.mark("<|im_end|>"),
                            tokenizer);
        return spans;
    }
}
