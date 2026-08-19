package com.qxotic.jinfer.models.lfm2;

import static com.qxotic.jinfer.chat.ReplyLanguage.mark;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.MediaEncodingCache;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.PromptWriter;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/** Token-exact native codec for the LFM 2.5 ChatML template. */
public final class Lfm2ChatTemplate implements ChatTemplate {
    private static final String THINK_OPEN = "<think>";
    private static final String THINK_CLOSE = "</think>";
    private static final String CALL_OPEN = "<|tool_call_start|>";
    private static final String CALL_CLOSE = "<|tool_call_end|>";

    private final Tokenizer tokenizer;
    private final Lfm2Vision vision;
    private final boolean promptOpensThinking;
    private final IntSequence promptStart;
    private final int turnOpen;
    private final int turnClose;
    private final int callOpen;
    private final int callClose;
    private final int thinkOpen;
    private final int thinkClose;
    private final ReplyLanguage.Spans replyLanguage;

    /** Builds the native template with the checkpoint's generation-prompt behavior. */
    public static Lfm2ChatTemplate fromGguf(Tokenizer tokenizer, GGUF gguf) {
        Objects.requireNonNull(gguf, "gguf");
        String source = gguf.getStringOrDefault("tokenizer.chat_template", "");
        return new Lfm2ChatTemplate(tokenizer, source.contains("<|im_start|>assistant\\n<think>"));
    }

    public Lfm2ChatTemplate(Tokenizer tokenizer, boolean promptOpensThinking) {
        this(tokenizer, null, promptOpensThinking);
    }

    public Lfm2ChatTemplate(Lfm2 model, boolean promptOpensThinking) {
        this(model.tokenizer(), model.vision(), promptOpensThinking);
    }

    Lfm2ChatTemplate(Tokenizer tokenizer, Lfm2Vision vision, boolean promptOpensThinking) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.vision = vision;
        this.promptOpensThinking = promptOpensThinking;
        promptStart = IntSequence.of(SpecialTokens.require(tokenizer, "<|startoftext|>"));
        turnOpen = SpecialTokens.require(tokenizer, "<|im_start|>");
        turnClose = SpecialTokens.require(tokenizer, "<|im_end|>");
        callOpen = SpecialTokens.require(tokenizer, CALL_OPEN);
        callClose = SpecialTokens.require(tokenizer, CALL_CLOSE);
        thinkOpen = SpecialTokens.find(tokenizer, THINK_OPEN).orElse(-1);
        thinkClose = SpecialTokens.find(tokenizer, THINK_CLOSE).orElse(-1);
        if (promptOpensThinking && (thinkOpen < 0 || thinkClose < 0)) {
            throw new IllegalArgumentException(
                    "prompt opens thinking but the tokenizer lacks think markers");
        }
        replyLanguage =
                new ReplyLanguage.Spans(
                        THINK_OPEN,
                        THINK_CLOSE,
                        CALL_OPEN,
                        CALL_CLOSE,
                        Lfm2ToolCodec::parse,
                        mark("<|im_end|>"),
                        tokenizer);
    }

    @Override
    public IntSequence promptStart() {
        return promptStart;
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
        for (Message message : conversation.messages())
            requireSupported(message, spliceable(message));

        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, mediaCache, sink);
        List<Message> messages = conversation.messages();
        int first = 0;
        String system = "";
        if (!messages.isEmpty() && messages.getFirst().role().equals(Role.SYSTEM)) {
            system = text(messages.getFirst());
            first = 1;
        }

        out.verbatim(promptStart());
        if (!system.isEmpty() || !conversation.tools().isEmpty()) {
            out.id(turnOpen).text("system\n").text(system);
            if (!system.isEmpty() && !conversation.tools().isEmpty()) out.text("\n");
            if (!conversation.tools().isEmpty())
                out.text(Lfm2ToolCodec.renderTools(conversation.tools()));
            out.id(turnClose).text("\n");
            out.flush();
        }

        int lastUser = lastUser(messages);
        for (int i = first; i < messages.size(); i++) {
            Message message = messages.get(i);
            if (spliceable(message)) splice(out, message);
            else writeTurn(out, message, i > lastUser, batchCapacity);
            out.flush();
        }

        out.id(turnOpen).text("assistant\n");
        IntSequence replyPrefix = IntSequence.empty();
        if (conversation.thinking() && promptOpensThinking) {
            out.id(thinkOpen);
            replyPrefix = IntSequence.of(thinkOpen);
        }
        out.finish();

        boolean claimCalls = !conversation.tools().isEmpty();
        ReplyParser parser =
                ReplyParser.spans(
                        tokenizer,
                        CALL_OPEN,
                        CALL_CLOSE,
                        Lfm2ToolCodec::parse,
                        THINK_OPEN,
                        THINK_CLOSE,
                        claimCalls);
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    @Override
    public ReplyParser parser(Tokenizer tokenizer) {
        return replyLanguage.parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(
            String contentGbnf, List<Tool> callableTools) {
        return Optional.of(replyLanguage.constrainedAuto(contentGbnf, !callableTools.isEmpty()));
    }

    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(replyLanguage.forcedCall(callableTools, tool -> "[" + tool.name()));
    }

    private void writeTurn(
            PromptWriter out, Message message, boolean afterLastUser, int batchCapacity) {
        out.id(turnOpen).text(message.role().name()).text("\n");
        if (message.role().equals(Role.ASSISTANT)) {
            boolean typedReasoning =
                    message.content().stream().anyMatch(Content.Reasoning.class::isInstance);
            if (afterLastUser) writeReasoning(out, message);
            String visible = text(message);
            if (!afterLastUser) visible = stripThinking(visible);
            if (typedReasoning && !afterLastUser) visible = visible.strip();
            out.text(visible);
            List<Content.ToolCall> calls = calls(message);
            if (!calls.isEmpty()) {
                out.id(callOpen).text(Lfm2ToolCodec.renderCalls(calls)).id(callClose);
            }
        } else {
            for (Content part : message.content()) {
                switch (part) {
                    case Content.Text value -> out.text(value.text());
                    case Content.ToolResult value -> out.text(value.text());
                    case Content.Media value -> writeMedia(out, value, batchCapacity);
                    default -> throw new IllegalStateException("unsupported " + part);
                }
            }
        }
        out.id(turnClose).text("\n");
    }

    private void writeMedia(PromptWriter out, Content.Media content, int batchCapacity) {
        if (!(content.value() instanceof Media.Image image) || vision == null)
            throw new IllegalStateException("unsupported media after validation");
        ContentKey contentKey = content.contentKey();
        out.cachedMedia(
                image,
                contentKey,
                encoded -> {
                    Lfm2VisionPreprocess.Plan plan = vision.plan(image);
                    encoded.id(require("<|image_start|>"));
                    for (Lfm2VisionPreprocess.Part part : plan.parts()) {
                        String marker =
                                part.thumbnail()
                                        ? "<|img_thumbnail|>"
                                        : "<|img_row_"
                                                + part.row()
                                                + "_col_"
                                                + part.column()
                                                + "|>";
                        encoded.id(require(marker));
                        vision.embed(
                                part,
                                batchCapacity,
                                rows ->
                                        encoded.batch(
                                                Batch.embeddings(
                                                        rows,
                                                        Math.toIntExact(rows.shape().flatAt(0)),
                                                        true,
                                                        contentKey)));
                    }
                    encoded.id(require("<|image_end|>"));
                });
    }

    @Override
    public int mediaPositions(Media media) {
        if (vision == null)
            throw new UnsupportedOperationException("image input needs an mmproj sidecar");
        if (media instanceof Media.Image image) return vision.positions(image);
        throw new UnsupportedOperationException(
                media.getClass().getSimpleName() + " is not supported by LFM2-VL");
    }

    private void writeReasoning(PromptWriter out, Message message) {
        StringBuilder text = new StringBuilder();
        for (Content part : message.content()) {
            if (!(part instanceof Content.Reasoning reasoning) || reasoning.content().isEmpty())
                continue;
            text.append(reasoningText(reasoning));
        }
        if (!text.isEmpty())
            out.id(requireThinkOpen()).text(text.toString()).id(requireThinkClose());
    }

    private void splice(PromptWriter out, Message message) {
        out.id(turnOpen).text("assistant\n");
        splice(out, message.content());
        out.id(turnClose).text("\n");
    }

    private void splice(PromptWriter out, List<Content> content) {
        for (Content part : content) {
            switch (part) {
                case Content.Text text -> out.verbatim(text.verbatim());
                case Content.ToolCall call ->
                        out.id(callOpen).verbatim(call.verbatim()).id(callClose);
                case Content.Reasoning reasoning -> {
                    out.id(requireThinkOpen());
                    splice(out, reasoning.content());
                    out.id(requireThinkClose());
                }
                default -> throw new IllegalStateException("unspliceable " + part);
            }
        }
    }

    private boolean spliceable(Message message) {
        return message.role().equals(Role.ASSISTANT)
                && !message.content().isEmpty()
                && verbatim(message.content());
    }

    private boolean verbatim(List<Content> content) {
        for (Content part : content) {
            boolean exact =
                    switch (part) {
                        case Content.Text text -> !text.verbatim().isEmpty();
                        case Content.ToolCall call -> !call.verbatim().isEmpty();
                        case Content.Reasoning reasoning ->
                                thinkOpen >= 0
                                        && thinkClose >= 0
                                        && !reasoning.content().isEmpty()
                                        && verbatim(reasoning.content());
                        default -> false;
                    };
            if (!exact) return false;
        }
        return true;
    }

    private void requireSupported(Message message, boolean exactReplay) {
        for (Content part : message.content()) {
            boolean supported =
                    part instanceof Content.Text
                            || (message.role().equals(Role.ASSISTANT)
                                    && (part instanceof Content.Reasoning
                                            || part instanceof Content.ToolCall))
                            || (message.role().equals(Role.TOOL)
                                    && part instanceof Content.ToolResult)
                            || (message.role().equals(Role.USER)
                                    && part instanceof Content.Media media
                                    && media.value() instanceof Media.Image
                                    && vision != null);
            if (!supported)
                throw new UnsupportedConversation(
                        part instanceof Content.Media && vision == null
                                ? "media on a text-only load"
                                : message.role().name()
                                        + " turn: "
                                        + part.getClass().getSimpleName());
            if (!exactReplay && part instanceof Content.Reasoning reasoning)
                reasoningText(reasoning);
        }
    }

    private static String text(Message message) {
        StringBuilder out = new StringBuilder();
        for (Content part : message.content()) {
            if (part instanceof Content.Text text) out.append(text.text());
            else if (part instanceof Content.ToolResult result) out.append(result.text());
        }
        return out.toString();
    }

    private static String reasoningText(Content.Reasoning reasoning) {
        StringBuilder out = new StringBuilder();
        for (Content part : reasoning.content()) {
            if (!(part instanceof Content.Text text))
                throw new UnsupportedConversation(
                        "reasoning contains " + part.getClass().getSimpleName());
            out.append(text.text());
        }
        return out.toString();
    }

    private static List<Content.ToolCall> calls(Message message) {
        List<Content.ToolCall> calls = new ArrayList<>();
        for (Content part : message.content())
            if (part instanceof Content.ToolCall call) calls.add(call);
        return calls;
    }

    private static int lastUser(List<Message> messages) {
        int last = -1;
        for (int i = 0; i < messages.size(); i++)
            if (messages.get(i).role().equals(Role.USER)) last = i;
        return last;
    }

    private static String stripThinking(String content) {
        int close = content.lastIndexOf(THINK_CLOSE);
        return close < 0 ? content : content.substring(close + THINK_CLOSE.length()).strip();
    }

    private int require(String spelling) {
        return SpecialTokens.require(tokenizer, spelling);
    }

    private int requireThinkOpen() {
        if (thinkOpen < 0) throw new UnsupportedConversation("tokenizer has no <think> token");
        return thinkOpen;
    }

    private int requireThinkClose() {
        if (thinkClose < 0) throw new UnsupportedConversation("tokenizer has no </think> token");
        return thinkClose;
    }
}
