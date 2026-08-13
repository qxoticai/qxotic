package com.qxotic.jinfer.x.models.gemma4;

import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Embedder;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jinfer.x.boundary.MultiModal;
import com.qxotic.jinfer.x.chat.ChatTemplate;
import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.Conversation;
import com.qxotic.jinfer.x.chat.Message;
import com.qxotic.jinfer.x.chat.PromptWriter;
import com.qxotic.jinfer.x.chat.ReplyLanguage;
import com.qxotic.jinfer.x.chat.ReplyParser;
import com.qxotic.jinfer.x.chat.Role;
import com.qxotic.jinfer.x.chat.Tool;
import com.qxotic.jinfer.x.chat.UnsupportedConversation;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/** Gemma 4 chat framing with structural image and audio input and the tool round-trip. */
public final class Gemma4ChatTemplate implements ChatTemplate {
    private static final String CHANNEL_OPEN = "<|channel>";
    private static final String CHANNEL_CLOSE = "<channel|>";
    private static final String CALL_OPEN = "<|tool_call>";
    private static final String CALL_CLOSE = "<tool_call|>";
    private static final String RESPONSE_OPEN = "<|tool_response>";
    private static final String RESPONSE_CLOSE = "<tool_response|>";
    private static final String DECLARATION_OPEN = "<|tool>";
    private static final String DECLARATION_CLOSE = "<tool|>";

    private final Tokenizer tokenizer;
    private final MultiModal media;
    private final boolean scaffoldsNonThinking;
    private final int bos;
    private final int turnOpen;
    private final int turnClose;
    private final int imageOpen;
    private final int imageClose;
    private final int audioOpen;
    private final int audioClose;
    private final IntSequence noThinkingPrefix;
    private final IntSequence thoughtTail;

    public Gemma4ChatTemplate(Tokenizer tokenizer) {
        this(tokenizer, null, false);
    }

    public Gemma4ChatTemplate(Gemma4 model, boolean scaffoldsNonThinking) {
        this(model.tokenizer(), model, scaffoldsNonThinking);
    }

    public Gemma4ChatTemplate(Tokenizer tokenizer, MultiModal media, boolean scaffoldsNonThinking) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.media = media;
        this.scaffoldsNonThinking = scaffoldsNonThinking;
        bos = SpecialTokens.require(tokenizer, "<bos>");
        turnOpen = SpecialTokens.require(tokenizer, "<|turn>");
        turnClose = SpecialTokens.require(tokenizer, "<turn|>");
        imageOpen = SpecialTokens.require(tokenizer, "<|image>");
        imageClose = SpecialTokens.require(tokenizer, "<image|>");
        audioOpen = SpecialTokens.require(tokenizer, "<|audio>");
        audioClose = SpecialTokens.require(tokenizer, "<audio|>");
        int channelOpen = SpecialTokens.require(tokenizer, CHANNEL_OPEN);
        IntSequence.Builder prefix = IntSequence.newBuilder();
        prefix.add(channelOpen);
        prefix.addAll(tokenizer.encode("thought\n"));
        prefix.add(SpecialTokens.require(tokenizer, CHANNEL_CLOSE));
        noThinkingPrefix = prefix.build();
        if (scaffoldsNonThinking) {
            IntSequence.Builder tail = IntSequence.newBuilder();
            tail.add(channelOpen);
            tail.addAll(tokenizer.encode("thought\n"));
            thoughtTail = tail.build();
        } else {
            thoughtTail = IntSequence.empty();
        }
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        Objects.requireNonNull(conversation, "conversation");
        requireSupported(conversation);
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        out.id(bos);
        List<Message> msgs = conversation.messages();
        boolean systemFirst = !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM);
        int start = 0;
        if (systemFirst || !conversation.tools().isEmpty()) {
            systemBlock(out, systemFirst ? msgs.get(0) : null, conversation.tools());
            out.flush();
            if (systemFirst) start = 1;
        }
        // upstream template fix 35b4173: reasoning is PRESERVED in turns after the last user
        // message (the in-flight tool loop) and stripped everywhere before it
        int lastUserIdx = -1;
        for (int i = 0; i < msgs.size(); i++) {
            if (msgs.get(i).role().equals(Role.USER)) lastUserIdx = i;
        }
        // the template's per-message state: 'call'/'response' leave the model turn OPEN
        String prev = null;
        Role prevNonToolRole = null;
        boolean openTail = false; // did the FINAL emitted turn stay open?
        for (int i = start; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) continue; // folded into its call turn below
            prev = null;
            openTail = false;
            boolean assistant = m.role().equals(Role.ASSISTANT);
            boolean continuation = assistant && Role.ASSISTANT.equals(prevNonToolRole);
            prevNonToolRole = m.role();
            List<Content.ToolCall> calls =
                    m.content().stream()
                            .filter(Content.ToolCall.class::isInstance)
                            .map(Content.ToolCall.class::cast)
                            .toList();
            Content.Reasoning reasoning = reasoningOf(assistant ? m : null);
            String thought =
                    reasoning != null && i > lastUserIdx && !reasoning.text().isEmpty()
                            ? reasoning.text()
                            : null;
            // turn-tag balance (upstream fix): an assistant turn does not close when the next
            // non-tool message is also assistant - the turns merge into one model turn
            Role nextRole = null;
            for (int j = i + 1; j < msgs.size(); j++) {
                if (!msgs.get(j).role().equals(Role.TOOL)) {
                    nextRole = msgs.get(j).role();
                    break;
                }
            }
            boolean continuesIntoNext = assistant && Role.ASSISTANT.equals(nextRole);
            if (calls.isEmpty() && !continuation && reasoning == null && !continuesIntoNext) {
                writeTurn(out, m);
                out.flush();
                continue;
            }
            if (m.content().stream().anyMatch(Content.Media.class::isInstance))
                throw new UnsupportedConversation("media in a tool-call/continuation model turn");
            if (!continuation) {
                out.id(turnOpen).text(roleName(m.role())).text("\n");
            }
            if (thought != null) {
                // <|channel>thought\n{reasoning}\n<channel|> - before calls and content
                out.id(require(CHANNEL_OPEN))
                        .text("thought\n")
                        .text(thought)
                        .text("\n")
                        .id(require(CHANNEL_CLOSE));
            }
            for (Content.ToolCall call : calls) {
                out.id(require(CALL_OPEN));
                sinkInto(out, s -> Gemma4ToolSyntax.call(call.name(), call.arguments(), s));
                out.id(require(CALL_CLOSE));
            }
            // forward-fold the consecutive tool-role results, names resolved from the calls.
            // A tool turn's result arrives as Content.ToolResult (typed API) OR Content.Text
            // (the server's lowering shape); dropping the Text form silently starved the model
            // of every served tool result.
            boolean responses = false;
            int nthResult = 0; // results fold in call order: the id-less Text shape resolves
            for (int j = i + 1; j < msgs.size() && msgs.get(j).role().equals(Role.TOOL); j++) {
                for (Content part : msgs.get(j).content()) {
                    String callId;
                    String resultText;
                    if (part instanceof Content.ToolResult r) {
                        callId = r.callId();
                        resultText = r.text();
                    } else if (part instanceof Content.Text t && !t.text().isEmpty()) {
                        callId = "";
                        resultText = t.text();
                    } else {
                        continue;
                    }
                    out.id(require(RESPONSE_OPEN));
                    String name =
                            callId.isEmpty() && nthResult < calls.size()
                                    ? calls.get(nthResult).name()
                                    : resolveName(calls, callId);
                    nthResult++;
                    sinkInto(out, s -> Gemma4ToolSyntax.response(name, resultText, s));
                    out.id(require(RESPONSE_CLOSE));
                    responses = true;
                    prev = "response";
                }
            }
            if (!calls.isEmpty() && !responses) prev = "call";
            String content = m.text().strip(); // the lenient view: call parts render above
            out.text(content);
            if ("call".equals(prev)) {
                out.id(require(RESPONSE_OPEN)); // awaiting results: the turn stays open
            } else if (continuesIntoNext) {
                // no close, no flush: the next assistant message continues this model turn and
                // juxtaposed text BPE-merges exactly like the whole render
                continue;
            } else if (!(responses && content.isEmpty() && nextRole == null)) {
                // close unless the conversation ENDS on folded responses with no answer yet
                // (then the open turn is the generation surface); a following non-assistant
                // turn always closes this one (upstream turn-tag balance: next_nt.found)
                out.id(turnClose).text("\n");
                out.flush();
            } else {
                openTail = true; // trailing folded responses: the model turn stays open
            }
        }
        IntSequence replyPrefix = IntSequence.empty();
        if (openTail) {
            // trailing folded responses left the turn open. Thinking on: the model RESUMES
            // thinking - the prompt opens the channel and the reply starts inside it, so the
            // prefix IS this tail. Thinking off (or no channel scaffolding): the model answers
            // directly in the open turn - the reference emits nothing.
            if (conversation.thinking() && scaffoldsNonThinking) {
                out.verbatim(thoughtTail);
                replyPrefix = thoughtTail;
            }
        } else if (!"call".equals(prev)) {
            // normal end - and the llama.cpp patch: a CLOSED final turn always reopens
            // <|turn>model regardless of what prev reads ("call" stays open at the await marker)
            out.id(turnOpen).text("model\n");
            if (!conversation.thinking() && scaffoldsNonThinking) {
                out.verbatim(noThinkingPrefix);
                replyPrefix = noThinkingPrefix;
            }
        }
        out.finish();

        ReplyParser parser = spans().parser();
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    /** {@code <|turn>system\n} + trimmed system text + one declaration block per tool. */
    private void systemBlock(PromptWriter out, Message system, List<Tool> tools) {
        out.id(turnOpen).text("system\n");
        if (system != null) out.text(system.text().strip());
        for (Tool tool : tools) {
            out.id(require(DECLARATION_OPEN));
            sinkInto(out, s -> Gemma4ToolSyntax.declaration(tool.definition(), s));
            out.id(require(DECLARATION_CLOSE));
        }
        out.id(turnClose).text("\n");
    }

    /** Runs a tool-syntax renderer: text runs accumulate, quotes emit the trusted id. */
    private void sinkInto(PromptWriter out, Consumer<Gemma4ToolSyntax.Sink> render) {
        render.accept(
                new Gemma4ToolSyntax.Sink() {
                    @Override
                    public void text(String s) {
                        out.text(s);
                    }

                    @Override
                    public void quote() {
                        out.id(require(Gemma4ToolSyntax.QUOTE));
                    }
                });
    }

    private static Content.Reasoning reasoningOf(Message message) {
        if (message == null) return null;
        for (Content part : message.content()) {
            if (part instanceof Content.Reasoning reasoning) return reasoning;
        }
        return null;
    }

    private static String resolveName(List<Content.ToolCall> calls, String callId) {
        for (Content.ToolCall call : calls) {
            if (call.id().equals(callId)) return call.name();
        }
        return calls.size() == 1 ? calls.get(0).name() : "unknown";
    }

    private int require(String name) {
        return SpecialTokens.require(tokenizer, name);
    }

    /** The part shapes this port frames byte-exactly; anything else is rejected. */
    private void requireSupported(Conversation conversation) {
        for (Message m : conversation.messages()) {
            boolean toolTurn = m.role().equals(Role.TOOL);
            boolean assistant = m.role().equals(Role.ASSISTANT);
            for (Content part : m.content()) {
                boolean ok =
                        switch (part) {
                            case Content.Text t -> true;
                            case Content.Media b -> !toolTurn;
                            case Content.ToolCall c -> assistant;
                            case Content.ToolResult r -> toolTurn;
                            // upstream fix 35b4173: reasoning renders in turns after the last
                            // user message (thought channel), stripped before it
                            case Content.Reasoning r -> assistant;
                        };
                if (!ok)
                    throw new UnsupportedConversation(
                            m.role().name() + " turn: " + part.getClass().getSimpleName());
                if (part instanceof Content.Media && media == null)
                    throw new UnsupportedConversation("media on a text-only load");
            }
        }
    }

    private void writeTurn(PromptWriter out, Message message) {
        boolean hasMedia = message.content().stream().anyMatch(Content.Media.class::isInstance);
        out.id(turnOpen).text(roleName(message.role())).text("\n");
        if (!hasMedia) {
            for (Content part : message.content()) requireText(message, part);
            out.text(message.text().strip());
        } else {
            for (Content part : message.content()) {
                switch (part) {
                    case Content.Text text -> out.text(text.text());
                    case Content.Media value -> writeMedia(out, value);
                    default -> requireText(message, part);
                }
            }
        }
        out.id(turnClose).text("\n");
    }

    private void writeMedia(PromptWriter out, Content.Media content) {
        switch (content.value()) {
            case Media.Image image ->
                    writeMedia(
                            out,
                            image,
                            content.contentKey(),
                            Media.Image.class,
                            imageOpen,
                            imageClose,
                            true);
            case Media.Audio audio ->
                    writeMedia(
                            out,
                            audio,
                            content.contentKey(),
                            Media.Audio.class,
                            audioOpen,
                            audioClose,
                            false);
            case Media.Video ignored ->
                    throw new UnsupportedConversation("Gemma 4 video framing is not ported");
        }
    }

    private <M extends Media> void writeMedia(
            PromptWriter out,
            M value,
            byte[] contentKey,
            Class<M> type,
            int open,
            int close,
            boolean bidirectional) {
        Embedder<M> embedder = media == null ? null : media.embedder(type).orElse(null);
        if (embedder == null)
            throw new UnsupportedConversation(
                    type.getSimpleName().toLowerCase() + " input is not supported by this model");
        out.id(open).media(value, contentKey, embedder, bidirectional).id(close);
    }

    /** Best-effort media positions via the modality's embedder plan (no encoding). */
    @Override
    public int mediaPositions(Media m) {
        return switch (m) {
            case Media.Image img -> plan(Media.Image.class, img);
            case Media.Audio aud -> plan(Media.Audio.class, aud);
            default ->
                    throw new UnsupportedOperationException(
                            m.getClass().getSimpleName() + " is not supported by this model");
        };
    }

    private <M extends Media> int plan(Class<M> type, M m) {
        Embedder<M> embedder = media == null ? null : media.embedder(type).orElse(null);
        if (embedder == null)
            throw new UnsupportedOperationException(
                    type.getSimpleName().toLowerCase() + " input is not supported by this model");
        return embedder.positions(m);
    }

    private static void requireText(Message message, Content part) {
        if (!(part instanceof Content.Text))
            throw new UnsupportedConversation(
                    message.role().name() + " turn: " + part.getClass().getSimpleName());
    }

    private static String roleName(Role role) {
        return role.equals(Role.ASSISTANT) ? "model" : role.name();
    }

    private ReplyLanguage.Spans spans; // the family's derived faces, markers written once

    @Override
    public ReplyParser parser(Tokenizer tokenizer) {
        return spans().parser();
    }

    private ReplyLanguage.Spans spans() {
        if (spans == null) {
            List<ReplyLanguage.Node> ends = new ArrayList<>();
            ends.add(ReplyLanguage.mark("<turn|>"));
            ends.add(ReplyLanguage.mark("<end_of_turn>"));
            ends.add(ReplyLanguage.mark("<|endoftext|>"));
            if (tokenizer.vocabulary().contains("<eos>")) {
                ends.add(ReplyLanguage.markId("<eos>", tokenizer.vocabulary().id("<eos>")));
            }
            spans =
                    new ReplyLanguage.Spans(
                            CHANNEL_OPEN,
                            CHANNEL_CLOSE,
                            CALL_OPEN,
                            CALL_CLOSE,
                            Gemma4ToolSyntax::parseBlock,
                            new ReplyLanguage.Node.Alt(ends),
                            tokenizer);
        }
        return spans;
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
        return Optional.of(spans().forcedCall(callableTools, tool -> "call:" + tool.name()));
    }
}
