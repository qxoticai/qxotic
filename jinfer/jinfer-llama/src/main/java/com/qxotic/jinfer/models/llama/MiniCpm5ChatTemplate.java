package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * Hand-written MiniCPM5 chat framing (ChatML dialect over the {@code llama} graph): {@code <s>}
 * bos, {@code <|im_start|>role\n...<|im_end|>\n} turns, and the family's XML function-call wire -
 * declarations as {@code tojson} signatures inside trusted {@code <tools>} ids, calls as {@code
 * <function name="N"><param name="K">V</param></function>} where {@code <function}, {@code
 * </param>} and {@code </function>} are control specials (the render+rescan mints them; {@code
 * <param} opens stay plain text), CDATA-wrapped values per the template's rule. Tool results fold
 * into one user turn of trusted {@code <tool_response>} blocks.
 *
 * <p>Reasoning follows the Qwen-style last-query rule: {@code <think>} blocks are re-rendered only
 * on assistant turns AFTER the last real user message (the active tool round-trip); earlier CoT is
 * dropped. The generation prompt appends {@code <think>\n} (thinking) or the closed empty pair. The
 * template's {@code <tool_def_sep>} system-placeholder is not ported (plain concat only).
 */
public final class MiniCpm5ChatTemplate implements ChatTemplate {

    // The declarations block, split where trusted spellings sit (<tools> ids 12/13; the
    // guidelines' literal <function ... </function> mentions ids 18/19, </param> id 21)
    static final String DEFS_INTRO = "# Tools\n\nYou are provided with function signatures within ";
    static final String XML_TAGS = " XML tags:\n";
    static final String GUIDE_A =
            "\n\nTool usage guidelines:\n- You may call zero or more functions. If no function"
                    + " calls are needed, just answer normally and do not include any ";
    static final String GUIDE_DOTS = " ... ";
    static final String GUIDE_B = ".\n- When calling a function, return an XML object within ";
    static final String GUIDE_C = " using:\n";
    static final String GUIDE_EXAMPLE =
            " name=\"function-name\"><param name=\"param-name\">param-value";
    static final String GUIDE_D =
            "\n- param-value may be multi-line. If it contains <, & or newline characters, wrap it"
                    + " in a CDATA block: <param name=\"param-name\"><![CDATA[...multi-line"
                    + " value...]]>";

    private final Tokenizer tokenizer;
    private final int bos; // <s>
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    private final int toolsOpen; // <tools>
    private final int toolsClose; // </tools>
    private final int function; // <function
    private final int endFunction; // </function>
    private final int endParam; // </param>
    private final int toolResponse; // <tool_response>
    private final int endToolResponse; // </tool_response>

    public MiniCpm5ChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.bos = SpecialTokens.require(tokenizer, "<s>");
        this.imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        this.imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        this.think = SpecialTokens.require(tokenizer, "<think>");
        this.endThink = SpecialTokens.require(tokenizer, "</think>");
        this.toolsOpen = SpecialTokens.require(tokenizer, "<tools>");
        this.toolsClose = SpecialTokens.require(tokenizer, "</tools>");
        this.function = SpecialTokens.require(tokenizer, "<function");
        this.endFunction = SpecialTokens.require(tokenizer, "</function>");
        this.endParam = SpecialTokens.require(tokenizer, "</param>");
        this.toolResponse = SpecialTokens.require(tokenizer, "<tool_response>");
        this.endToolResponse = SpecialTokens.require(tokenizer, "</tool_response>");
    }

    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        TurnTemplate.requireToolShapes(msgs);
        List<Tool> tools = conversation.tools();

        Message sys =
                !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM) ? msgs.get(0) : null;
        int lastQuery = msgs.size() - 1;
        for (int i = msgs.size() - 1; i >= 0; i--) {
            if (msgs.get(i).role().equals(Role.USER)) {
                lastQuery = i;
                break;
            }
        }

        IntSequence.Builder ids = IntSequence.newBuilder();
        StringBuilder text = new StringBuilder();
        ids.add(bos);
        if (!tools.isEmpty()) {
            ids.add(imStart);
            text.append("system\n");
            if (sys != null) text.append(sys.textOnly()).append("\n\n");
            text.append(DEFS_INTRO);
            emit(ids, text, toolsOpen);
            emit(ids, text, toolsClose);
            text.append(XML_TAGS);
            emit(ids, text, toolsOpen);
            for (Tool t : tools) {
                text.append('\n').append(ToolCallSyntax.jinjaJson(JsonCodec.parse(t.rawJson())));
            }
            text.append('\n');
            emit(ids, text, toolsClose);
            text.append(GUIDE_A);
            emit(ids, text, function);
            text.append(GUIDE_DOTS);
            emit(ids, text, endFunction);
            text.append(GUIDE_B);
            emit(ids, text, function);
            text.append(GUIDE_DOTS);
            emit(ids, text, endFunction);
            text.append(GUIDE_C);
            emit(ids, text, function);
            text.append(GUIDE_EXAMPLE);
            emit(ids, text, endParam);
            emit(ids, text, endFunction);
            text.append(GUIDE_D);
            emit(ids, text, endParam);
            emit(ids, text, imEnd);
            text.append('\n');
        } else if (sys != null) {
            ids.add(imStart);
            text.append("system\n").append(sys.textOnly());
            emit(ids, text, imEnd);
            text.append('\n');
        }

        for (int i = sys != null ? 1 : 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) {
                if (i == 0 || !msgs.get(i - 1).role().equals(Role.TOOL)) {
                    emit(ids, text, imStart);
                    text.append("user"); // no newline: the response block brings its own
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
                    emit(ids, text, imEnd);
                    text.append('\n');
                }
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                emit(ids, text, imStart);
                text.append(m.role().name()).append('\n').append(m.textOnly());
                emit(ids, text, imEnd);
                text.append('\n');
                continue;
            }
            emit(ids, text, imStart);
            text.append("assistant\n");
            String content = m.text();
            Part.Reasoning reasoning = m.reasoning();
            if (reasoning != null && i > lastQuery) {
                emit(ids, text, think);
                text.append('\n').append(strip(reasoning.text(), '\n')).append('\n');
                emit(ids, text, endThink);
                text.append("\n\n").append(stripLeading(content));
            } else {
                text.append(content);
            }
            boolean first = true;
            for (Part p : m.content()) {
                if (!(p instanceof Part.ToolCall call)) continue;
                if (!first || !content.isEmpty()) text.append('\n');
                first = false;
                emit(ids, text, function);
                text.append(" name=\"").append(call.name()).append("\">");
                for (var arg : call.arguments().entrySet()) {
                    text.append("<param name=\"").append(arg.getKey()).append("\">");
                    text.append(MiniCpmToolSyntax.paramValue(arg.getValue()));
                    emit(ids, text, endParam);
                }
                emit(ids, text, endFunction);
            }
            emit(ids, text, imEnd);
            text.append('\n');
        }

        emit(ids, text, imStart);
        text.append("assistant\n");
        if (conversation.thinking()) {
            emit(ids, text, think);
            text.append('\n');
        } else {
            emit(ids, text, think);
            text.append("\n\n");
            emit(ids, text, endThink);
            text.append("\n\n");
        }
        flush(ids, text);
        List<Batch> out = new ArrayList<>();
        out.add(Batch.prefill(ids.build().toArray()));
        return out;
    }

    private static String strip(String s, char c) {
        int a = 0, b = s.length();
        while (a < b && s.charAt(a) == c) a++;
        while (b > a && s.charAt(b - 1) == c) b--;
        return s.substring(a, b);
    }

    private static String stripLeading(String s) {
        int i = 0;
        while (i < s.length() && s.charAt(i) == '\n') i++;
        return s.substring(i);
    }

    private void emit(IntSequence.Builder ids, StringBuilder text, int id) {
        flush(ids, text);
        ids.add(id);
    }

    private void flush(IntSequence.Builder ids, StringBuilder text) {
        if (text.isEmpty()) return;
        ids.addAll(tokenizer.encode(text.toString()));
        text.setLength(0);
    }

    /**
     * Calls parse from the span between the trusted {@code <function} / {@code </function>} ids.
     */
    @Override
    public ReplyParser parser() {
        return ReplyParser.spans(
                tokenizer, "<function", "</function>", MiniCpmToolSyntax::parsePayload);
    }

    /** Forced calls seed {@code <function} (seeding only - no pin hook yet). */
    @Override
    public int[] callSeed() {
        return new int[] {function};
    }

    /** The generation prompt opens the think span (or its closed pair): pre-feed it. */
    @Override
    public int[] replySeed(boolean thinking) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(think);
        if (thinking) {
            ids.addAll(tokenizer.encode("\n"));
        } else {
            ids.addAll(tokenizer.encode("\n\n"));
            ids.add(endThink);
            ids.addAll(tokenizer.encode("\n\n"));
        }
        return ids.build().toArray();
    }
}
