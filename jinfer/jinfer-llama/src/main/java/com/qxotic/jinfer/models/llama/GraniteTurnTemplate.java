package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * Hand-written Granite 4.1 chat framing, token-exact with the GGUF's Jinja {@code
 * tokenizer.chat_template} for plain conversations (no tools/documents) and validated against it
 * offline (GraniteTurnTemplateOracle).
 *
 * <p>Layout: no bos ({@code add_bos_token} is false and the template never emits one), no default
 * system message; every turn — system, user and assistant alike — frames as {@code
 * <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n}; generation prompt {@code
 * <|start_of_role|>assistant<|end_of_role|>} with nothing after it. Content is NOT trimmed (the
 * template has no {@code | trim}) and is tokenized as one contiguous plain-encoded run between the
 * role-close and turn-close specials, exactly as the rendered template tokenizes. The {@code
 * thinking} flag is ignored: the vocab carries think tokens but the template has no reasoning
 * scaffold.
 *
 * <p>Empty system messages are the caller's to omit (the template drops them; a turn here would
 * still frame). Two domains: the three role/turn markers are emitted as trusted ids; everything
 * else goes through plain {@link Tokenizer#encode} so conversation text can never mint control
 * tokens.
 */
public final class GraniteTurnTemplate implements TurnTemplate {

    private final Tokenizer tokenizer;
    private final int startRole; // <|start_of_role|>
    private final int endRole; // <|end_of_role|>
    private final int endText; // <|end_of_text|>
    private final IntSequence newline; // encode("\n"), constant
    private final List<Batch>
            generationPrompt; // <|start_of_role|>assistant<|end_of_role|>, constant
    private final List<Batch> closeTurn; // <|end_of_text|>\n, constant

    public GraniteTurnTemplate(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.startRole = SpecialTokens.require(tokenizer, "<|start_of_role|>");
        this.endRole = SpecialTokens.require(tokenizer, "<|end_of_role|>");
        this.endText = SpecialTokens.require(tokenizer, "<|end_of_text|>");
        this.newline = tokenizer.encode("\n");
        IntSequence gen =
                IntSequence.of(startRole)
                        .concat(tokenizer.encode("assistant"))
                        .concat(IntSequence.of(endRole));
        this.generationPrompt = List.of(Batch.prefill(gen.toArray()));
        IntSequence close = IntSequence.of(endText).concat(newline);
        this.closeTurn = List.of(Batch.prefill(close.toArray()));
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(); // no bos, no preamble
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        // <|start_of_role|> {role} <|end_of_role|> {content} <|end_of_text|> \n
        IntSequence ids =
                IntSequence.of(startRole)
                        .concat(tokenizer.encode(message.role().name()))
                        .concat(IntSequence.of(endRole))
                        .concat(tokenizer.encode(message.textOnly()))
                        .concat(IntSequence.of(endText))
                        .concat(newline);
        return List.of(Batch.prefill(ids.toArray()));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return generationPrompt;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    // The tools system message, split where its literal <tools> / <tool_call> spellings sit:
    // both marker families are control specials in this vocab, so the template-authored
    // instruction text emits them as trusted ids (matching the render+rescan).
    static final String TOOLS_INTRO =
            "You are a helpful assistant with access to the following tools. You may call one or"
                    + " more tools to assist with the user query.\n\n"
                    + "You are provided with function signatures within ";
    static final String XML_TAGS = " XML tags:\n";
    static final String TOOLS_OUTRO_HEAD =
            "\n\nFor each tool call, return a json object with function name and arguments"
                    + " within ";
    static final String TOOLS_EXAMPLE_BODY =
            "\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n";
    static final String TOOLS_OUTRO_TAIL =
            ". If a tool does not exist in the provided list of tools, notify the user that you do"
                    + " not have the ability to fulfill the request.";

    /**
     * The codec face, tools included - the template's flow: the tools message (per-tool {@code
     * tojson} of the WHOLE envelope, framed by the fixed instructions whose literal {@code
     * <tool_call>} spellings emit as trusted ids) joins the system turn; an assistant call turn is
     * its content then {@code <tool_call>\n{"name": "N", "arguments": A}\n</tool_call>} per call;
     * consecutive tool results fold into ONE user turn of {@code \n<tool_response>\n...} blocks.
     * The template has no reasoning scaffold, so {@link Part.Reasoning} on history is dropped.
     * Plain conversations keep the oracle-validated per-turn fold.
     */
    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        List<Tool> tools = conversation.tools();
        if (tools.isEmpty() && msgs.stream().allMatch(GraniteTurnTemplate::textOnly)) {
            return TurnTemplate.super.encode(conversation);
        }
        TurnTemplate.requireToolShapes(msgs);
        int toolCall = SpecialTokens.require(tokenizer, "<tool_call>");
        int endToolCall = SpecialTokens.require(tokenizer, "</tool_call>");
        int toolResponse = SpecialTokens.require(tokenizer, "<tool_response>");
        int endToolResponse = SpecialTokens.require(tokenizer, "</tool_response>");

        Message sys =
                !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM) ? msgs.get(0) : null;
        int first = sys != null ? 1 : 0;
        IntSequence.Builder ids = IntSequence.newBuilder();
        StringBuilder text = new StringBuilder();
        if (sys != null || !tools.isEmpty()) {
            ids.add(startRole);
            text.append("system");
            emit(ids, text, endRole);
            if (sys != null) text.append(sys.textOnly());
            if (!tools.isEmpty()) {
                int toolsOpen = SpecialTokens.require(tokenizer, "<tools>");
                int toolsClose = SpecialTokens.require(tokenizer, "</tools>");
                if (sys != null) text.append("\n\n");
                text.append(TOOLS_INTRO);
                emit(ids, text, toolsOpen);
                emit(ids, text, toolsClose);
                text.append(XML_TAGS);
                emit(ids, text, toolsOpen);
                for (Tool t : tools) {
                    text.append('\n')
                            .append(ToolCallSyntax.jinjaJson(JsonCodec.parse(t.rawJson())));
                }
                text.append('\n');
                emit(ids, text, toolsClose);
                text.append(TOOLS_OUTRO_HEAD);
                emit(ids, text, toolCall);
                emit(ids, text, endToolCall);
                text.append(XML_TAGS);
                emit(ids, text, toolCall);
                text.append(TOOLS_EXAMPLE_BODY);
                emit(ids, text, endToolCall);
                text.append(TOOLS_OUTRO_TAIL);
            }
            emit(ids, text, endText);
            text.append('\n');
        }
        for (int i = first; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) {
                if (i == first || !msgs.get(i - 1).role().equals(Role.TOOL)) {
                    emit(ids, text, startRole);
                    text.append("user");
                    emit(ids, text, endRole);
                }
                text.append('\n');
                emit(ids, text, toolResponse);
                text.append('\n');
                for (Part p : m.content()) {
                    if (p instanceof Part.ToolResult r) text.append(r.text());
                }
                text.append('\n');
                emit(ids, text, endToolResponse);
                boolean nextIsTool =
                        i + 1 < msgs.size() && msgs.get(i + 1).role().equals(Role.TOOL);
                if (!nextIsTool) {
                    emit(ids, text, endText);
                    text.append('\n');
                }
                continue;
            }
            emit(ids, text, startRole);
            text.append(m.role().name());
            emit(ids, text, endRole);
            String content = m.text(); // Reasoning dropped: the template has no think scaffold
            text.append(content);
            if (m.role().equals(Role.ASSISTANT)) {
                boolean firstCall = true;
                for (Part p : m.content()) {
                    if (!(p instanceof Part.ToolCall call)) continue;
                    if (!firstCall || !content.isEmpty()) text.append('\n');
                    firstCall = false;
                    emit(ids, text, toolCall);
                    text.append("\n{\"name\": \"")
                            .append(call.name())
                            .append("\", \"arguments\": ")
                            .append(ToolCallSyntax.jinjaJson(call.arguments()))
                            .append("}\n");
                    emit(ids, text, endToolCall);
                }
            }
            emit(ids, text, endText);
            text.append('\n');
        }
        flush(ids, text);
        List<Batch> out = new ArrayList<>();
        out.add(Batch.prefill(ids.build().toArray()));
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }

    private static boolean textOnly(Message m) {
        return m.content().stream().allMatch(p -> p instanceof Part.Text);
    }

    /** Flushes the pending text run plainly, then emits one trusted id. */
    private void emit(IntSequence.Builder ids, StringBuilder text, int id) {
        flush(ids, text);
        ids.add(id);
    }

    private void flush(IntSequence.Builder ids, StringBuilder text) {
        if (text.isEmpty()) return;
        ids.addAll(tokenizer.encode(text.toString()));
        text.setLength(0);
    }

    /** Tool calls parse from the {@code <tool_call>} span's JSON {@code {name, arguments}}. */
    @Override
    public ReplyParser parser() {
        if (SpecialTokens.find(tokenizer, "<tool_call>").isEmpty()) {
            return ReplyParser.spans(tokenizer);
        }
        return ReplyParser.spans(
                tokenizer, "<tool_call>", "</tool_call>", ToolCallSyntax::parseBlock);
    }

    /** Forced calls seed {@code <tool_call>} (seeding only - no pin hook yet). */
    @Override
    public int[] callSeed() {
        return SpecialTokens.find(tokenizer, "<tool_call>").stream().toArray();
    }
}
