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
 * The ChatML JSON-envelope reply language shared by SmolLM3 and Granite: an optional {@code
 * <think>} span, content and {@code <tool_call>} JSON spans interleaved, one terminator. The AUTO
 * call span is a classic marker pair around a free payload parsed by {@link
 * ToolCallSyntax#parseBlock} - exactly the old span detector's contract, so malformed payloads drop
 * without ending the reply and the span's verbatim ids ride the call. The FORCED language binds
 * each offered tool's name into the envelope bytes and its arguments to the tool's schema.
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
                            bytes("\n{\"name\": \"" + tool.name() + "\", \"arguments\": "),
                            gbnf(Grammar.schemaGbnf(tool.parameters())),
                            bytes("}\n"),
                            mark("</tool_call>")));
        }
        return seq(new ReplyLanguage.Node.Alt(options), opt(mark(terminator)));
    }
}
