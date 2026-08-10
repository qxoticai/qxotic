package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

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

    // The declarations block, the template's own strings: TokenRuns.trusted mints the literal
    // <tools> / <function / </function> / </param> spellings (control specials in this vocab;
    // <param opens are NOT in the rescan's kept set and stay plain).
    static final String DEFS_INTRO =
            "# Tools\n\nYou are provided with function signatures within <tools></tools> XML"
                    + " tags:\n<tools>";
    static final String GUIDELINES =
            "\n</tools>\n\nTool usage guidelines:\n- You may call zero or more functions. If no"
                    + " function calls are needed, just answer normally and do not include any"
                    + " <function ... </function>.\n- When calling a function, return an XML object"
                    + " within <function ... </function> using:\n<function"
                    + " name=\"function-name\"><param name=\"param-name\">param-value</param>"
                    + "</function>\n- param-value may be multi-line. If it contains <, & or newline"
                    + " characters, wrap it in a CDATA block: <param"
                    + " name=\"param-name\"><![CDATA[...multi-line value...]]></param>";

    private final Tokenizer tokenizer;
    private final int bos; // <s>
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    private final int function; // <function
    private final int endFunction; // </function>
    private final int endParam; // </param>
    private final int toolResponse; // <tool_response>
    private final int endToolResponse; // </tool_response>
    private final TokenRuns proto; // compiled spelling table, forked per encode

    public MiniCpm5ChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.bos = SpecialTokens.require(tokenizer, "<s>");
        this.imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        this.imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        this.think = SpecialTokens.require(tokenizer, "<think>");
        this.endThink = SpecialTokens.require(tokenizer, "</think>");
        this.function = SpecialTokens.require(tokenizer, "<function");
        this.endFunction = SpecialTokens.require(tokenizer, "</function>");
        this.endParam = SpecialTokens.require(tokenizer, "</param>");
        this.toolResponse = SpecialTokens.require(tokenizer, "<tool_response>");
        this.endToolResponse = SpecialTokens.require(tokenizer, "</tool_response>");
        this.proto = new TokenRuns(tokenizer);
    }

    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        TurnTemplate.requireToolShapes(msgs);
        List<Tool> tools = conversation.tools();

        Message sys = Message.leadingSystem(msgs);
        int lastQuery = msgs.size() - 1;
        for (int i = msgs.size() - 1; i >= 0; i--) {
            if (msgs.get(i).role().equals(Role.USER)) {
                lastQuery = i;
                break;
            }
        }

        // cache-boundary law (ChatTemplate): preamble, turns, scaffold-last are separate
        // batches. Every boundary sits at an <|im_start|> special, so the split is token-exact
        // with the whole render (BPE cannot merge across a special).
        List<Batch> out = new ArrayList<>();
        TokenRuns runs = proto.fresh();
        runs.id(bos);
        if (!tools.isEmpty()) {
            runs.id(imStart).text("system\n");
            if (sys != null) runs.text(sys.textOnly()).text("\n\n");
            runs.trusted(DEFS_INTRO);
            for (Tool t : tools) {
                runs.text("\n").text(ToolCallSyntax.jinjaJson(JsonCodec.parse(t.rawJson())));
            }
            runs.trusted(GUIDELINES);
            runs.id(imEnd).text("\n");
        } else if (sys != null) {
            runs.id(imStart).text("system\n").text(sys.textOnly());
            runs.id(imEnd).text("\n");
        }

        for (int i = sys != null ? 1 : 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            boolean toolContinuation =
                    m.role().equals(Role.TOOL) && i > 0 && msgs.get(i - 1).role().equals(Role.TOOL);
            if (!toolContinuation) { // a folded tool run stays inside its turn's batch
                out.addAll(runs.batches());
                runs = proto.fresh();
            }
            if (m.role().equals(Role.TOOL)) {
                if (!toolContinuation) {
                    runs.id(imStart).text("user"); // no newline: the response block brings its own
                }
                runs.text("\n").id(toolResponse).text("\n");
                for (Part p : m.content()) {
                    if (p instanceof Part.ToolResult r) runs.text(r.text());
                }
                runs.text("\n").id(endToolResponse);
                boolean nextIsTool =
                        i + 1 < msgs.size() && msgs.get(i + 1).role().equals(Role.TOOL);
                if (!nextIsTool) runs.id(imEnd).text("\n");
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                runs.id(imStart).text(m.role().name()).text("\n").text(m.textOnly());
                runs.id(imEnd).text("\n");
                continue;
            }
            runs.id(imStart).text("assistant\n");
            String content = m.text();
            Part.Reasoning reasoning = m.reasoning();
            if (reasoning != null && i > lastQuery) {
                runs.id(think).text("\n").text(strip(reasoning.text(), '\n')).text("\n");
                runs.id(endThink).text("\n\n").text(stripLeading(content));
            } else {
                runs.text(content);
            }
            boolean first = true;
            for (Part p : m.content()) {
                if (!(p instanceof Part.ToolCall call)) continue;
                if (!first || !content.isEmpty()) runs.text("\n");
                first = false;
                runs.id(function).text(" name=\"").text(call.name()).text("\">");
                for (var arg : call.arguments().entrySet()) {
                    runs.text("<param name=\"").text(arg.getKey()).text("\">");
                    runs.text(MiniCpmToolSyntax.paramValue(arg.getValue()));
                    runs.id(endParam);
                }
                runs.id(endFunction);
            }
            runs.id(imEnd).text("\n");
        }

        out.addAll(runs.batches());
        runs = proto.fresh();
        runs.id(imStart).text("assistant\n");
        if (conversation.thinking()) runs.id(think).text("\n");
        else runs.id(think).text("\n\n").id(endThink).text("\n\n");
        out.addAll(runs.batches()); // scaffold last: the cache's own-block convention
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

    /**
     * Calls parse from the span between the trusted {@code <function} / {@code </function>} ids.
     */
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

    /**
     * The reply-language walk; the {@code </param>} closers are SPECIALS inside the payload, and a
     * marker-pair call span claims interior control tokens AS THEIR SPELLINGS - exactly the decoded
     * text the old span parser fed {@link MiniCpmToolSyntax#parsePayload}.
     */
    @Override
    public ReplyParser parser() {
        return spans().parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedAuto(
            String contentGbnf, boolean toolsOffered) {
        return Optional.of(spans().constrainedAuto(contentGbnf, toolsOffered));
    }

    /** Forced calls: the header carries an OFFERED name, the arguments stay the model's own. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> tools) {
        if (tools.isEmpty()) return Optional.empty();
        return Optional.of(spans().forcedCall(tools, tool -> " name=\"" + tool.name()));
    }

    /** The generation prompt opens the think span (or its closed pair): pre-feed it. */
    @Override
    public int[] replySeed(boolean thinking) {
        return TurnTemplate.reasonSeed(tokenizer, thinking);
    }
}
