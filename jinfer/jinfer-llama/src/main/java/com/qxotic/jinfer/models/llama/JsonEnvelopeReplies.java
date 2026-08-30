package com.qxotic.jinfer.models.llama;

import static com.qxotic.jinfer.chat.ReplyLanguage.bytes;
import static com.qxotic.jinfer.chat.ReplyLanguage.call;
import static com.qxotic.jinfer.chat.ReplyLanguage.gbnf;
import static com.qxotic.jinfer.chat.ReplyLanguage.mark;
import static com.qxotic.jinfer.chat.ReplyLanguage.opt;
import static com.qxotic.jinfer.chat.ReplyLanguage.seq;

import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.Grammar;
import java.util.ArrayList;
import java.util.List;

/**
 * The ChatML JSON-envelope forced-call language shared by SmolLM3 and Granite: per offered tool,
 * the envelope bytes carry the name and the schema grammar binds the arguments.
 */
final class JsonEnvelopeReplies {

    private JsonEnvelopeReplies() {}

    /** Per-tool spans: the envelope carries the name, the schema grammar the arguments. */
    static ReplyLanguage.Node forced(List<Tool> tools, String terminator) {
        List<ReplyLanguage.Node> options = new ArrayList<>(tools.size());
        for (Tool tool : tools) {
            options.add(
                    call(
                            ToolCallSyntax::parseBlock,
                            mark("<tool_call>"),
                            bytes(
                                    "\n{\"name\": "
                                            + ToolCallSyntax.jinjaJson(tool.name())
                                            + ", \"arguments\": "),
                            gbnf(Grammar.schemaGbnf(tool.parameters())),
                            bytes("}\n"),
                            mark("</tool_call>")));
        }
        return seq(new ReplyLanguage.Node.Alt(options), opt(mark(terminator)));
    }
}
