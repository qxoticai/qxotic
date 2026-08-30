package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
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
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;

/** Token-exact native codec for the Llama 3.2 plain-conversation template. */
public final class Llama32ChatTemplate implements ChatTemplate {
    public static final String DEFAULT_DATE = "26 Jul 2024";

    private final Tokenizer tokenizer;
    private final String systemPreamble;
    private final IntSequence promptStart;
    private final int startHeader;
    private final int endHeader;
    private final int endTurn;

    public Llama32ChatTemplate(Tokenizer tokenizer) {
        this(tokenizer, DEFAULT_DATE);
    }

    public Llama32ChatTemplate(Tokenizer tokenizer, String date) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.systemPreamble =
                "Cutting Knowledge Date: December 2023\nToday Date: "
                        + Objects.requireNonNull(date, "date")
                        + "\n\n";
        promptStart = IntSequence.of(SpecialTokens.require(tokenizer, "<|begin_of_text|>"));
        startHeader = SpecialTokens.require(tokenizer, "<|start_header_id|>");
        endHeader = SpecialTokens.require(tokenizer, "<|end_header_id|>");
        endTurn = SpecialTokens.require(tokenizer, "<|eot_id|>");
    }

    @Override
    public IntSequence promptStart() {
        return promptStart;
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        Objects.requireNonNull(conversation, "conversation");
        if (!conversation.tools().isEmpty())
            throw new UnsupportedConversation("Llama 3.2 tool framing is not ported");

        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        List<Message> messages = conversation.messages();
        int first = 0;
        String system = "";
        if (!messages.isEmpty() && messages.getFirst().role().equals(Role.SYSTEM)) {
            system = text(messages.getFirst()).strip();
            first = 1;
        }

        out.verbatim(promptStart());
        writeTurn(out, Role.SYSTEM, systemPreamble + system, null);
        out.flush();
        for (int i = first; i < messages.size(); i++) {
            Message message = messages.get(i);
            String value = text(message).strip();
            Content.Text exact = exactText(message, value);
            writeTurn(out, message.role(), value, exact);
            out.flush();
        }
        out.id(startHeader).text(Role.ASSISTANT.name()).id(endHeader).text("\n\n");
        out.finish();

        IntSequence replyPrefix = IntSequence.empty();
        ReplyParser parser = ReplyParser.spans(tokenizer);
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    public int endTurnToken() {
        return endTurn;
    }

    @Override
    public ReplyParser parser(Tokenizer tokenizer) {
        return new BareToolCallParser(tokenizer);
    }

    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        List<ReplyLanguage.Node> bodies = new ArrayList<>(callableTools.size());
        for (Tool tool : callableTools) {
            bodies.add(
                    ReplyLanguage.seq(
                            ReplyLanguage.bytes(
                                    "{\"name\": " + ToolCallSyntax.jinjaJson(tool.name()) + ", "),
                            ReplyLanguage.bytes("\"parameters\": "),
                            ReplyLanguage.gbnf(Grammar.schemaGbnf(tool.parameters())),
                            ReplyLanguage.bytes("}")));
        }
        return Optional.of(
                ReplyLanguage.Selection.of(
                        ReplyLanguage.seq(
                                ReplyLanguage.call(
                                        ToolCallSyntax::parseBlock,
                                        new ReplyLanguage.Node.Alt(bodies)),
                                ReplyLanguage.opt(
                                        ReplyLanguage.alt(
                                                ReplyLanguage.mark("<|eot_id|>"),
                                                ReplyLanguage.mark("<|eom_id|>")))),
                        tokenizer));
    }

    private void writeTurn(PromptWriter out, Role role, String text, Content.Text exact) {
        out.id(startHeader).text(role.name()).id(endHeader).text("\n\n");
        if (exact == null) out.text(text);
        else out.verbatim(exact.verbatim());
        out.id(endTurn);
    }

    private static String text(Message message) {
        StringBuilder out = new StringBuilder();
        for (Content part : message.content()) {
            if (!(part instanceof Content.Text text))
                throw new UnsupportedConversation(
                        message.role() + " message contains " + part.getClass().getSimpleName());
            out.append(text.text());
        }
        return out.toString();
    }

    private static Content.Text exactText(Message message, String stripped) {
        if (message.content().size() != 1) return null;
        Content.Text text = (Content.Text) message.content().getFirst();
        return !text.verbatim().isEmpty() && text.text().equals(stripped) ? text : null;
    }

    /**
     * Llama's call has no marker: hold only the short ambiguous prefix, then either stream normal
     * content or keep a confirmed call atomic until it can be parsed.
     */
    private static final class BareToolCallParser implements ReplyParser {
        private static final Fragment EMPTY = new Fragment("", IntSequence.empty());

        private enum State {
            UNDECIDED,
            CALL,
            CONTENT
        }

        private enum Prefix {
            POSSIBLE,
            CALL,
            INVALID
        }

        private enum Match {
            PARTIAL,
            COMPLETE,
            INVALID
        }

        private final Tokenizer tokenizer;
        private final ReplyParser content;
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        private IntSequence.Builder ids = IntSequence.newBuilder();
        private State state = State.UNDECIDED;
        private Message finished;

        BareToolCallParser(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            this.content = ReplyParser.spans(tokenizer);
        }

        @Override
        public void seed(IntSequence seed) {
            if (finished != null) throw new IllegalStateException("parser already finished");
            if (state == State.CONTENT) {
                content.seed(seed);
                return;
            }
            seed.forEachInt(this::hold);
            Prefix prefix = prefix();
            if (prefix == Prefix.INVALID) {
                IntSequence held = ids.build();
                clear();
                state = State.CONTENT;
                content.seed(held);
            } else if (prefix == Prefix.CALL) {
                state = State.CALL;
            }
        }

        @Override
        public Fragment feed(int token) {
            if (finished != null) throw new IllegalStateException("parser already finished");
            if (state == State.CONTENT) return content.feed(token);
            if (SpecialTokens.isSpecial(tokenizer, token)) return EMPTY;
            hold(token);
            if (state == State.CALL) return EMPTY;
            Prefix prefix = prefix();
            if (prefix == Prefix.CALL) state = State.CALL;
            if (prefix != Prefix.INVALID) return EMPTY;

            IntSequence held = ids.build();
            clear();
            state = State.CONTENT;
            StringBuilder text = new StringBuilder();
            IntSequence.Builder verbatim = IntSequence.newBuilder();
            held.forEachInt(
                    id -> {
                        Fragment fragment = content.feed(id);
                        text.append(fragment.text());
                        fragment.tokens().forEachInt(verbatim::add);
                    });
            return new Fragment(text.toString(), verbatim.build());
        }

        @Override
        public boolean reasoning() {
            return state == State.CONTENT && content.reasoning();
        }

        @Override
        public Channel channel() {
            return state == State.CONTENT ? content.channel() : Channel.TOOL_CALL;
        }

        @Override
        public Channel pending() {
            return state == State.CONTENT ? content.pending() : null;
        }

        @Override
        public Set<Channel> outputChannels() {
            return state == State.CONTENT ? content.outputChannels() : Set.of(Channel.TOOL_CALL);
        }

        @Override
        public boolean ended() {
            return state == State.CONTENT && content.ended();
        }

        @Override
        public Message finish() {
            if (finished != null) return finished;
            if (state != State.CONTENT) {
                String wire = bytes.toString(StandardCharsets.UTF_8);
                List<Content.ToolCall> calls = ToolCallSyntax.parseBlock(wire);
                if (!calls.isEmpty()) {
                    finished = new Message(Role.ASSISTANT, new ArrayList<>(calls));
                    return finished;
                }
                IntSequence held = ids.build();
                clear();
                state = State.CONTENT;
                held.forEachInt(content::feed);
            }
            return finished = content.finish();
        }

        private void hold(int token) {
            ids.add(token);
            bytes.writeBytes(tokenizer.decodeBytes(new int[] {token}));
        }

        private Prefix prefix() {
            String text = bytes.toString(StandardCharsets.UTF_8);
            int at = skipWhitespace(text, 0);
            if (at == text.length()) return Prefix.POSSIBLE;
            if (text.charAt(at++) != '{') return Prefix.INVALID;
            at = skipWhitespace(text, at);

            Match name = match(text, at, "\"name\"");
            if (name == Match.COMPLETE) return colon(text, at + "\"name\"".length());
            Match type = match(text, at, "\"type\"");
            if (type == Match.COMPLETE) return functionType(text, at + "\"type\"".length());
            return name == Match.PARTIAL || type == Match.PARTIAL
                    ? Prefix.POSSIBLE
                    : Prefix.INVALID;
        }

        private static Prefix colon(String text, int at) {
            at = skipWhitespace(text, at);
            if (at == text.length()) return Prefix.POSSIBLE;
            return text.charAt(at) == ':' ? Prefix.CALL : Prefix.INVALID;
        }

        private static Prefix functionType(String text, int at) {
            at = skipWhitespace(text, at);
            if (at == text.length()) return Prefix.POSSIBLE;
            if (text.charAt(at++) != ':') return Prefix.INVALID;
            at = skipWhitespace(text, at);
            return switch (match(text, at, "\"function\"")) {
                case PARTIAL -> Prefix.POSSIBLE;
                case COMPLETE -> Prefix.CALL;
                case INVALID -> Prefix.INVALID;
            };
        }

        private static Match match(String text, int at, String literal) {
            for (int i = 0; i < literal.length(); i++) {
                if (at == text.length()) return Match.PARTIAL;
                if (text.charAt(at++) != literal.charAt(i)) return Match.INVALID;
            }
            return Match.COMPLETE;
        }

        private static int skipWhitespace(String text, int at) {
            while (at < text.length() && Character.isWhitespace(text.charAt(at))) at++;
            return at;
        }

        private void clear() {
            bytes.reset();
            ids = IntSequence.newBuilder();
        }
    }
}
