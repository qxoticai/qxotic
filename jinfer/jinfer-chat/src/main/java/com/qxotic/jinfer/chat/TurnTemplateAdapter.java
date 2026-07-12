package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.llm.GgufTokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * Bridges a legacy {@link TurnTemplate} onto the {@link ChatTemplate} codec so unported models work
 * through the new interface, bit-identical to the per-turn path: normalize, weld tools and the
 * leading system message into the preamble, encode each turn, append the assistant scaffold; decode
 * through {@link SpanReplyDecoder} with the template's own {@link ToolCallDetector}.
 *
 * <p>Per-turn templates re-render history from structure unconditionally, so verbatim splicing is a
 * native-codec capability this bridge does not have. Deleted when the last model ports.
 */
public final class TurnTemplateAdapter implements ChatTemplate {

    private final TurnTemplate turn;
    private final GgufTokenizer tokenizer;

    public TurnTemplateAdapter(TurnTemplate turn, GgufTokenizer tokenizer) {
        this.turn = turn;
        this.tokenizer = tokenizer;
    }

    /** The adapted per-turn template (turn-aligned drivers refine through this during porting). */
    public TurnTemplate turnTemplate() {
        return turn;
    }

    @Override
    public boolean supports(Conversation conversation) {
        if (!conversation.tools().isEmpty() && !turn.supportsTools()) return false;
        for (Message message : conversation.messages()) {
            for (Part part : message.content()) {
                // Per-turn templates predate ToolResult parts (tool results arrive as tool-role
                // turns); Reasoning history is a native-codec concern.
                if (part instanceof Part.ToolResult || part instanceof Part.Reasoning) return false;
            }
        }
        return true;
    }

    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> messages = turn.normalize(conversation.messages());
        List<Batch> out = new ArrayList<>();
        List<Message> turns = messages;
        if (!conversation.tools().isEmpty()) {
            turns = TurnTemplate.encodePreamble(turn, messages, conversation.tools(), out);
        } else {
            out.addAll(turn.conversationStart());
        }
        for (Message message : turns) out.addAll(turn.encodeTurn(message));
        out.addAll(turn.generationPrompt(conversation.thinking()));
        return out;
    }

    @Override
    public ReplyDecoder decoder() {
        return new SpanReplyDecoder(tokenizer, turn.toolCallDetector().orElse(null));
    }
}
