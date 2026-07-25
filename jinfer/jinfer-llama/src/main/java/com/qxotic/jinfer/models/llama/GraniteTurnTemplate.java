package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TokenRuns;
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
    private final TokenRuns proto; // compiled spelling table, forked per encode

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
        this.proto = new TokenRuns(tokenizer);
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

    // The tools system message, the template's own prefix/suffix pair: TokenRuns.trusted mints
    // the literal <tools> / <tool_call> spellings (control specials in this vocab) as ids,
    // matching the render+rescan.
    static final String TOOLS_PREFIX =
            "You are a helpful assistant with access to the following tools. You may call one or"
                    + " more tools to assist with the user query.\n\n"
                    + "You are provided with function signatures within <tools></tools> XML"
                    + " tags:\n<tools>";
    static final String TOOLS_SUFFIX =
            "\n"
                + "</tools>\n\n"
                + "For each tool call, return a json object with function name and arguments within"
                + " <tool_call></tool_call> XML tags:\n"
                + "<tool_call>\n"
                + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                + "</tool_call>. If a tool does not exist in the provided list of tools, notify the"
                + " user that you do not have the ability to fulfill the request.";

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
        TokenRuns runs = proto.fresh();
        if (sys != null || !tools.isEmpty()) {
            runs.id(startRole).text("system").id(endRole);
            if (sys != null) runs.text(sys.textOnly());
            if (!tools.isEmpty()) {
                if (sys != null) runs.text("\n\n");
                runs.trusted(TOOLS_PREFIX);
                for (Tool t : tools) {
                    runs.text("\n").text(ToolCallSyntax.jinjaJson(JsonCodec.parse(t.rawJson())));
                }
                runs.trusted(TOOLS_SUFFIX);
            }
            runs.id(endText).text("\n");
        }
        for (int i = first; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) {
                if (i == first || !msgs.get(i - 1).role().equals(Role.TOOL)) {
                    runs.id(startRole).text("user").id(endRole);
                }
                runs.text("\n").id(toolResponse).text("\n");
                for (Part p : m.content()) {
                    if (p instanceof Part.ToolResult r) runs.text(r.text());
                }
                runs.text("\n").id(endToolResponse);
                boolean nextIsTool =
                        i + 1 < msgs.size() && msgs.get(i + 1).role().equals(Role.TOOL);
                if (!nextIsTool) runs.id(endText).text("\n");
                continue;
            }
            runs.id(startRole).text(m.role().name()).id(endRole);
            String content = m.text(); // Reasoning dropped: the template has no think scaffold
            runs.text(content);
            if (m.role().equals(Role.ASSISTANT)) {
                boolean firstCall = true;
                for (Part p : m.content()) {
                    if (!(p instanceof Part.ToolCall call)) continue;
                    if (!firstCall || !content.isEmpty()) runs.text("\n");
                    firstCall = false;
                    runs.id(toolCall)
                            .text("\n{\"name\": \"")
                            .text(call.name())
                            .text("\", \"arguments\": ")
                            .text(ToolCallSyntax.jinjaJson(call.arguments()))
                            .text("}\n")
                            .id(endToolCall);
                }
            }
            runs.id(endText).text("\n");
        }
        List<Batch> out = new ArrayList<>();
        out.add(runs.batch());
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }

    private static boolean textOnly(Message m) {
        return m.content().stream().allMatch(p -> p instanceof Part.Text);
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
