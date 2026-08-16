package com.qxotic.jinfer.x.cli;

import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.chat.Channel;
import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.llm.Generator;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.PrintStream;
import java.util.HexFormat;
import java.util.Locale;

/**
 * The terminal half of one generation - the part every CLI mode shares, whichever of them assembled
 * the request: prompt echo, delta streaming, think routing (inline vs stderr) and the stderr timing
 * summary. The PARSE is the engine's ({@link ChatEngine.ReplySink} deltas arrive UTF-8-safe and
 * channel-tagged); what remains here is pure presentation, which is all a terminal ever wanted.
 */
final class Turn implements ChatEngine.ReplySink {

    private static final String ANSI_GREY = "\033[90m";
    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_RESET = "\033[0m";

    private final Tokenizer tokenizer;
    private final Options options;
    private final boolean rawLane;
    private final StringBuilder text = new StringBuilder();
    private boolean inReasoning;

    private Turn(Tokenizer tokenizer, Options options, boolean rawLane) {
        this.tokenizer = tokenizer;
        this.options = options;
        this.rawLane = rawLane;
    }

    /**
     * A turn over one prepared request: echoes the prompt when {@code --echo}, then serves as the
     * generation's {@link ChatEngine.ReplySink}.
     */
    static Turn start(Tokenizer tokenizer, ChatEngine.Prepared prepared, Options options) {
        Turn turn = new Turn(tokenizer, options, false);
        if (options.echo()) {
            echoPrompt(tokenizer, Batch.tokenIds(prepared.encoded().prompt()));
        }
        return turn;
    }

    /**
     * As {@link #start} over raw tokens - the {@code --raw-prompt} lane, whose prompt is the CLI's
     * own. Raw deltas carry NO parser, so control tokens (the family's end-of-turn, which the
     * generator feeds to its listener before the stop check) arrive as literal spellings; this lane
     * filters them from display, exactly like the old CLI's printer did.
     */
    static Turn startRaw(Tokenizer tokenizer, int[] promptTokens, Options options) {
        Turn turn = new Turn(tokenizer, options, true);
        if (options.echo()) {
            echoPrompt(tokenizer, promptTokens);
        }
        return turn;
    }

    @Override
    public void on(ChatEngine.Delta delta) {
        if (options.echo()) {
            delta.tokens()
                    .forEachInt(t -> System.err.print(escape(tokenizer.decode(new int[] {t}))));
        }
        if (rawLane && isControl(delta)) {
            return; // a parser-less delta of pure special tokens is control, not content
        }
        text.append(delta.text());
        if (!options.stream()) {
            return; // nothing streams - finish() prints the whole reply at once
        }
        if (delta.channel() == Channel.REASONING) {
            if (!options.think()) {
                return;
            }
            if (!inReasoning) {
                onThinkingStart();
                inReasoning = true;
            }
            thoughtOut().print(delta.text());
            return;
        }
        if (inReasoning) {
            onThinkingEnd();
            inReasoning = false;
        }
        System.out.print(delta.text());
        System.out.flush();
    }

    /**
     * The stderr summary every turn ends with - context fill, the two speeds, and where the prompt
     * came from (a cache tier the old CLI never could see) - then the whole reply when nothing
     * streamed.
     */
    void finish(ChatEngine.Completion completion, int contextCapacity) {
        if (inReasoning) {
            onThinkingEnd();
            inReasoning = false;
        }
        Generator.GenerationResult result = completion.result();
        if (result != null) {
            int promptTokens = completion.promptTokens();
            int evaluated = Math.max(0, promptTokens - completion.restoredTokens());
            int generated = result.completionTokens() + (result.stopToken().isPresent() ? 1 : 0);
            long promptNanos = Math.max(1, result.promptTime().toNanos());
            long decodeNanos = Math.max(1, result.decodeTime().toNanos());
            String prefix = options.colors() ? ANSI_CYAN : "";
            String suffix = options.colors() ? ANSI_RESET : "";
            System.err.printf(
                    "%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)"
                            + " cache: %s, %d restored%s%s%n",
                    prefix,
                    promptTokens + generated,
                    contextCapacity,
                    evaluated / (promptNanos / 1e9),
                    evaluated,
                    generated / (decodeNanos / 1e9),
                    generated,
                    completion.tier().name().toLowerCase(Locale.ROOT),
                    completion.restoredTokens(),
                    acceptance(completion),
                    suffix);
        }
        if (!options.stream()) {
            System.out.println(text);
        }
    }

    /** " accept: A/D (P%)" when the pass speculated, "" otherwise. */
    private static String acceptance(ChatEngine.Completion completion) {
        return completion
                .speculated()
                .map(
                        s ->
                                s.drafted() == 0
                                        ? " accept: -"
                                        : String.format(
                                                " accept: %d/%d (%.0f%%)",
                                                s.accepted(),
                                                s.drafted(),
                                                100.0 * s.accepted() / s.drafted()))
                .orElse("");
    }

    /** Every token of this delta is special - a control fragment the raw lane must not display. */
    private boolean isControl(ChatEngine.Delta delta) {
        IntSequence tokens = delta.tokens();
        for (int i = 0; i < tokens.length(); i++) {
            if (!SpecialTokens.isSpecial(tokenizer, tokens.intAt(i))) {
                return false;
            }
        }
        return true;
    }

    private PrintStream thoughtOut() {
        return options.thinkInline() ? System.out : System.err;
    }

    private void onThinkingStart() {
        if (options.colors()) {
            thoughtOut().print(ANSI_GREY);
        }
        thoughtOut().println("[Start thinking]");
    }

    private void onThinkingEnd() {
        thoughtOut().println();
        thoughtOut().println("[End thinking]");
        if (options.colors()) {
            thoughtOut().print(ANSI_RESET);
        }
        thoughtOut().println();
    }

    /** {@code --echo}: the prompt tokens to stderr, control characters escaped. */
    static void echoPrompt(Tokenizer tokenizer, int[] promptTokens) {
        for (int token : promptTokens) {
            System.err.print(escape(tokenizer.decode(new int[] {token})));
        }
    }

    /** Escape control characters (except newline) so token echo cannot distort the terminal. */
    private static String escape(String str) {
        StringBuilder chars = new StringBuilder();
        str.codePoints()
                .forEach(
                        cp -> {
                            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
                            } else {
                                chars.appendCodePoint(cp);
                            }
                        });
        return chars.toString();
    }
}
