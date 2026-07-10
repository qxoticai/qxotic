package com.qxotic.jinfer.chat;

import java.util.List;

/**
 * A model's tool-call detector: turns the stream of generated token ids into structured {@link
 * Part.ToolCall}s. Owned by the {@link TurnTemplate} because the format is per-model - LFM2.5
 * delimits calls with {@code <|tool_call_start|>}/{@code <|tool_call_end|>} and a Pythonic-or-JSON
 * payload, Qwen with {@code <tool_call>}/{@code </tool_call>} and JSON, gpt-oss with a harmony
 * channel. This is the token-id analogue of SGLang's per-model {@code Detector}, and it detects on
 * ids rather than decoded text: a call boundary is a TRUSTED special token, so conversation content
 * can never fake one (the same two-domain invariant the templates enforce when encoding). Only the
 * payload inside a span is parsed as text, where the model's own grammar applies.
 *
 * <p>Stateful and single-pass. Feed every generated token in order via {@link #accept}; drive it
 * one token at a time for streaming, or feed the whole reply for one-shot. {@link #calls} returns
 * the calls completed so far.
 */
public interface ToolCallDetector {

    /**
     * Consume the next generated token.
     *
     * @return {@code true} if the token belongs to a tool call (an open/close marker or a token
     *     inside a span) and the caller must therefore keep it out of the visible content stream;
     *     {@code false} for any other token, which the caller routes normally.
     */
    boolean accept(int token);

    /** The tool calls completed so far. Grows as spans close; never returns null. */
    List<Part.ToolCall> calls();
}
