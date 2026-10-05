package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;

/**
 * The terminal half of one generation - the part every CLI mode shares, whichever of them assembled
 * the request: prompt echo, delta streaming, think routing (inline vs stderr) and the stderr timing
 * summary. The PARSE is the engine's ({@link ChatEngine.ReplySink} deltas arrive UTF-8-safe and
 * channel-tagged); what remains here is pure presentation, which is all a terminal ever wanted.
 *
 * <p>{@code --no-stream} is the same rendering, replayed in {@link #finish}: one path decides what
 * goes where, so the two modes cannot disagree.
 */
final class Turn implements ChatEngine.ReplySink, AutoCloseable {

    private static final String ANSI_GREY = "\033[90m";
    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_RESET = "\033[0m";

    private final Tokenizer tokenizer;
    private final boolean stream, echo, thinkInline, thoughtColors, errorColors;
    private final Main.IO io;
    private final boolean rawLane;
    // --echo writes every generated token to stderr; rendering a token again on the same screen
    // interleaves the two copies word by word ("Thinking Thinking Process Process")
    private final boolean echoedThoughts, echoedContent;
    private final List<ChatEngine.Delta> buffered = new ArrayList<>(); // --no-stream
    private boolean inReasoning;
    private final Options options;
    private boolean reasoned, answered;

    Turn(Tokenizer tokenizer, Options options, boolean rawLane, Main.IO io) {
        this.tokenizer = tokenizer;
        this.stream = options.stream;
        this.echo = options.echo;
        this.thinkInline = options.thinkInline;
        this.thoughtColors = options.colors(io, thinkInline ? 1 : 2);
        this.errorColors = options.colors(io, 2);
        this.io = io;
        this.rawLane = rawLane;
        this.echoedContent = echo && io.isTerminal(1) && io.isTerminal(2);
        this.echoedThoughts = echo && (!thinkInline || echoedContent);
        this.options = options;
        if (!stream) io.err().println("Generating response ...");
    }

    /**
     * A turn over one prepared request: echoes the prompt when {@code --echo}, then serves as the
     * generation's {@link ChatEngine.ReplySink}.
     */
    static Turn start(
            Tokenizer tokenizer, ChatEngine.Prepared prepared, Options options, Main.IO io) {
        Turn turn = new Turn(tokenizer, options, false, io);
        if (options.echo) {
            echoPrompt(tokenizer, Batch.tokenIds(prepared.encoded().prompt()), io.err());
        }
        return turn;
    }

    /**
     * As {@link #start} over raw tokens - the {@code --raw-prompt} lane, whose prompt is the CLI's
     * own. Raw deltas carry NO parser, so control tokens (the family's end-of-turn, which the
     * generator feeds to its listener before the stop check) arrive as literal spellings; this lane
     * filters them from display, exactly like the old CLI's printer did.
     */
    static Turn startRaw(Tokenizer tokenizer, int[] promptTokens, Options options, Main.IO io) {
        Turn turn = new Turn(tokenizer, options, true, io);
        if (options.echo) {
            echoPrompt(tokenizer, promptTokens, io.err());
        }
        return turn;
    }

    @Override
    public void on(ChatEngine.Delta delta) {
        if (echo) {
            delta.tokens().forEachInt(t -> io.err().print(escape(tokenizer.decode(new int[] {t}))));
        }
        if (rawLane && isControl(delta)) {
            return; // a parser-less delta of pure special tokens is control, not content
        }
        if (stream) emit(delta);
        else buffered.add(delta);
    }

    /**
     * The one rendering path: a delta to its stream, thinking framed. With {@code --echo}, what the
     * echo already put on a screen is not rendered there twice: thoughts bound for stderr, and the
     * answer too when stdout is the same terminal (a redirected stdout still gets it).
     */
    private void emit(ChatEngine.Delta delta) {
        if (delta.channel() == Channel.REASONING) {
            reasoned |= !delta.text().isBlank();
            if (echoedThoughts) return;
            if (!inReasoning) {
                onThinkingStart();
                inReasoning = true;
            }
            thoughtOut().print(delta.text());
        } else {
            answered |= !delta.text().isBlank();
            close();
            if (!echoedContent) io.out().print(delta.text());
        }
        checkOutput();
    }

    void finish(ChatEngine.Completion completion, ChatEngine engine) {
        finish(
                completion,
                engine.contextCapacity(),
                engine.maxReasoningTokens(
                        options.think, options.maxOutputTokens, options.maxReasoningTokens));
    }

    /**
     * The stderr summary every turn ends with - why the reply stopped early, if it did, then
     * context fill, the two speeds, and where the prompt came from - then the whole reply when
     * nothing streamed.
     */
    void finish(ChatEngine.Completion completion, int contextCapacity, int reasoningCap) {
        if (!stream) {
            buffered.forEach(this::emit);
            buffered.clear();
        }
        close();
        io.out().println(); // the reply's line ends on stdout, streamed or not
        checkOutput();
        Generator.GenerationResult result = completion.result();
        if (result != null) {
            int evaluated = Math.max(0, completion.promptTokens() - completion.restoredTokens());
            int generated = generated(result);
            long promptNanos = result.promptTime().toNanos();
            long decodeNanos = result.decodeTime().toNanos();
            String prefix = errorColors ? ANSI_CYAN : "";
            String suffix = errorColors ? ANSI_RESET : "";
            notices(completion, contextCapacity, reasoningCap);
            io.err()
                    .printf(
                            Locale.ROOT,
                            "%scontext: %d/%d prompt: %s generation: %s cache: %s, %d"
                                    + " restored%s%s%n",
                            prefix,
                            used(completion),
                            contextCapacity,
                            speed(evaluated, promptNanos),
                            speed(generated, decodeNanos),
                            completion.tier().name().toLowerCase(Locale.ROOT),
                            completion.restoredTokens(),
                            acceptance(completion),
                            suffix);
        }
    }

    /** The context this turn leaves filled: its prompt, its reply and the stop token. */
    static int used(ChatEngine.Completion completion) {
        Generator.GenerationResult result = completion.result();
        return completion.promptTokens() + (result == null ? 0 : generated(result));
    }

    private static int generated(Generator.GenerationResult result) {
        return result.completionTokens() + (result.stopToken().isPresent() ? 1 : 0);
    }

    /** Restore reasoning framing even when generation throws before finish(). */
    @Override
    public void close() {
        if (inReasoning) {
            onThinkingEnd();
            inReasoning = false;
        }
    }

    private void checkOutput() {
        // checkError flushes, so this is also the streaming flush point.
        if (io.out().checkError())
            throw new UncheckedIOException(
                    new IOException("cannot write generated text to stdout"));
    }

    /**
     * Below this many tokens a rate measures fixed latency (the first step, a batch that is mostly
     * padding), not throughput: a 1-token prefill "at 4 tokens/s" only alarms.
     */
    static final int MIN_RATE_TOKENS = 32;

    /** "12.34 tokens/s (512)" over enough tokens to mean throughput, else "17 tokens in 0.73 s". */
    static String speed(int tokens, long nanos) {
        double seconds = Math.max(1, nanos) / 1e9;
        if (tokens >= MIN_RATE_TOKENS)
            return String.format(Locale.ROOT, "%.2f tokens/s (%d)", tokens / seconds, tokens);
        return String.format(
                Locale.ROOT, "%d token%s in %.2f s", tokens, tokens == 1 ? "" : "s", seconds);
    }

    /**
     * Why the model was cut short, if it was: the reasoning cap leaves no mark in the visible text,
     * and a reply stopped at the context wall or the output budget may have no answer at all.
     */
    private void notices(ChatEngine.Completion completion, int capacity, int reasoningCap) {
        if (reasoningCap > 0 && completion.reasoningTokens() >= reasoningCap)
            io.err()
                    .println(
                            "reasoning: cut at its cap of "
                                    + reasoningCap
                                    + " tokens; raise --max-reasoning-tokens (-1: uncapped)");
        if (completion.result().finishReason() != Generator.FinishReason.LENGTH) return;
        String unanswered = reasoned && !answered ? " during the reasoning, no answer" : "";
        if (used(completion) >= capacity || options.maxOutputTokens < 0)
            io.err()
                    .println(
                            "stopped: the context is full ("
                                    + capacity
                                    + " tokens)"
                                    + unanswered
                                    + "; raise --context-capacity");
        else
            io.err()
                    .println(
                            "stopped: --max-output-tokens "
                                    + options.maxOutputTokens
                                    + " reached"
                                    + unanswered);
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
        return thinkInline ? io.out() : io.err();
    }

    private void onThinkingStart() {
        if (thoughtColors) {
            thoughtOut().print(ANSI_GREY);
        }
        thoughtOut().println("[Start thinking]");
    }

    private void onThinkingEnd() {
        thoughtOut().println();
        thoughtOut().println("[End thinking]");
        if (thoughtColors) {
            thoughtOut().print(ANSI_RESET);
        }
        thoughtOut().println();
    }

    /** {@code --echo}: the prompt tokens to stderr, control characters escaped. */
    static void echoPrompt(Tokenizer tokenizer, int[] promptTokens, PrintStream out) {
        for (int token : promptTokens) {
            out.print(escape(tokenizer.decode(new int[] {token})));
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
