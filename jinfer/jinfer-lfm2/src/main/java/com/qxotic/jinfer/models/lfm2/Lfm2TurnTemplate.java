package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.GgufTokenizer;
import java.util.List;
import java.util.Map;

/**
 * Hand-written LFM2.5 chat framing (ChatML dialect), byte-exact with the GGUF's Jinja {@code
 * tokenizer.chat_template} and validated against it offline.
 *
 * <p>Layout: {@code <|startoftext|>} once, then per turn {@code
 * <|im_start|>{role}\n{content}<|im_end|>\n}, generation prompt {@code <|im_start|>assistant\n}.
 * The role header and the content are tokenized as ONE contiguous plain-encoded run — that is how
 * the rendered template tokenizes (specials force the only splits), and BPE merges across the
 * header/content boundary, so encoding them separately would drift.
 *
 * <p>Two domains: {@code <|startoftext|>}/{@code <|im_start|>}/{@code <|im_end|>} are emitted as
 * trusted ids; everything else goes through plain {@link GgufTokenizer#encode} so conversation text
 * can never mint control tokens. Matching the template, a historical assistant turn keeps only the
 * text after its last {@code </think>} (trimmed).
 *
 * <p>Tools weld into the system turn ({@code List of tools: [...]}); assistant tool-call turns
 * append {@code <|tool_call_start|>[pythonic]<|tool_call_end|>}; tool results are a {@code tool}
 * turn. Out of scope, matching the model: media parts (text-only port). Empty system messages are
 * the caller's to omit (the template drops them; a turn here would still frame).
 */
public final class Lfm2TurnTemplate implements TurnTemplate {

    private final GgufTokenizer tokenizer;
    private final int bos; // <|startoftext|>
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int tcStart; // <|tool_call_start|>
    private final int tcEnd; // <|tool_call_end|>
    private final int[] newline; // encode("\n"), constant
    private final List<Batch> generationPrompt; // <|im_start|>assistant\n, constant
    private final List<Batch> closeTurn; // <|im_end|>\n, constant

    public Lfm2TurnTemplate(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> special = tokenizer.getSpecialTokens();
        this.bos = tokenizer.requiredSpecial("<|startoftext|>");
        this.imStart = tokenizer.requiredSpecial("<|im_start|>");
        this.imEnd = tokenizer.requiredSpecial("<|im_end|>");
        this.tcStart = tokenizer.requiredSpecial("<|tool_call_start|>");
        this.tcEnd = tokenizer.requiredSpecial("<|tool_call_end|>");
        this.newline = ids(tokenizer.encode("\n"));
        List<Integer> gen = new java.util.ArrayList<>(List.of(imStart));
        gen.addAll(tokenizer.encode("assistant\n"));
        this.generationPrompt = List.of(Batch.prefill(gen));
        List<Integer> close = new java.util.ArrayList<>(List.of(imEnd));
        close.addAll(tokenizer.encode("\n"));
        this.closeTurn = List.of(Batch.prefill(close));
    }

    private static int[] ids(List<Integer> l) {
        int[] a = new int[l.size()];
        for (int i = 0; i < a.length; i++) a[i] = l.get(i);
        return a;
    }

    @Override
    public boolean supportsTools() {
        return true;
    }

    @Override
    public java.util.Optional<com.qxotic.jinfer.chat.ToolCallDetector> toolCallDetector() {
        return java.util.Optional.of(new Lfm2ToolCallDetector(tokenizer));
    }

    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[] {bos}));
    }

    /**
     * The template welds tools into the system turn: {@code system_prompt = {system content}\nList
     * of tools: [{tool0 json}, {tool1 json}]}, rendered once as a system turn. When tools are
     * present but there is no system message, a system turn is still emitted with only the tool
     * list. Each tool is its raw request JSON verbatim (Jinja {@code tool | tojson}).
     */
    @Override
    public List<Batch> conversationStart(Preamble preamble) {
        List<Integer> ids = new java.util.ArrayList<>(List.of(bos));
        String system = preamble.system().map(Message::text).orElse("");
        if (!preamble.tools().isEmpty()) {
            StringBuilder tools = new StringBuilder("List of tools: [");
            List<Tool> list = preamble.tools();
            for (int t = 0; t < list.size(); t++) {
                if (t > 0) tools.append(", ");
                tools.append(list.get(t).rawJson());
            }
            tools.append(']');
            system = system.isEmpty() ? tools.toString() : system + "\n" + tools;
        }
        if (!system.isEmpty()) {
            ids.add(imStart);
            ids.addAll(tokenizer.encode("system\n" + system));
            ids.add(imEnd);
            for (int nl : newline) ids.add(nl);
        }
        return List.of(Batch.prefill(ids));
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        Role role = message.role();
        String content = textContent(message);
        if (role.equals(Role.ASSISTANT)) content = stripThinking(content);
        List<Part.ToolCall> calls = toolCalls(message);
        // <|im_start|>{role}\n{content}[<|tool_call_start|>[calls]<|tool_call_end|>]<|im_end|>\n
        // Each plain span (header+content, and the bracketed call list) is a separate contiguous
        // run: the trusted markers split them, exactly as encodeWithSpecialTokens rescans the
        // rendered string.
        List<Integer> ids = new java.util.ArrayList<>();
        ids.add(imStart);
        ids.addAll(tokenizer.encode(role.name() + "\n" + content));
        if (!calls.isEmpty()) {
            ids.add(tcStart);
            ids.addAll(tokenizer.encode("[" + ToolCallSyntax.renderPythonic(calls) + "]"));
            ids.add(tcEnd);
        }
        ids.add(imEnd);
        for (int nl : newline) ids.add(nl);
        return List.of(Batch.prefill(ids));
    }

    /** The message's text parts concatenated (media/tool-call parts excluded). */
    private static String textContent(Message message) {
        StringBuilder sb = new StringBuilder();
        for (Part part : message.content()) if (part instanceof Part.Text t) sb.append(t.text());
        return sb.toString();
    }

    /** The tool-call parts of a (usually assistant) message, in order. */
    private static List<Part.ToolCall> toolCalls(Message message) {
        List<Part.ToolCall> calls = new java.util.ArrayList<>();
        for (Part part : message.content()) if (part instanceof Part.ToolCall c) calls.add(c);
        return calls;
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return generationPrompt;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    /**
     * The template keeps only the text after the last {@code </think>} in historical assistant
     * turns: {@code content.split("</think>")[-1] | trim}.
     */
    private static String stripThinking(String content) {
        int at = content.lastIndexOf("</think>");
        return at < 0 ? content : content.substring(at + "</think>".length()).strip();
    }
}
