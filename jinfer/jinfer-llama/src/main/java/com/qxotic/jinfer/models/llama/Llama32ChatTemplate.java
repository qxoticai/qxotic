package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.PromptWriter;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/** Token-exact native codec for the Llama 3.2 plain-conversation template. */
public final class Llama32ChatTemplate implements ChatTemplate {
    public static final String DEFAULT_DATE = "26 Jul 2024";

    private final Tokenizer tokenizer;
    private final String systemPreamble;
    private final int bos;
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
        bos = SpecialTokens.require(tokenizer, "<|begin_of_text|>");
        startHeader = SpecialTokens.require(tokenizer, "<|start_header_id|>");
        endHeader = SpecialTokens.require(tokenizer, "<|end_header_id|>");
        endTurn = SpecialTokens.require(tokenizer, "<|eot_id|>");
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

        out.id(bos);
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
}
