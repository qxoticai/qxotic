package com.qxotic.jinfer.chat;

import java.util.List;

/**
 * The decode direction of a {@link ChatTemplate}: a stateful, single-use parser that turns one
 * generated reply token stream into structured {@link Part}s. Subsumes tool-call detection AND the
 * streaming think-demux - reply structure is template grammar, decoded in one place, on token ids
 * (a span boundary is a TRUSTED special token, so conversation content can never fake one).
 *
 * <p>Feed EVERY sampled token in order, specials and the trailing stop token included (recognized
 * stop and turn-close specials are inert). Deltas are UTF-8 safe: multi-byte sequences split across
 * tokens are buffered, never emitted as replacement characters. Stop STRINGS stay outside the
 * decoder - the caller applies its stop-aware holdback to {@link Part.Text} deltas and aborts the
 * token loop; the decoder is purely structural.
 */
public interface ReplyDecoder {

    /**
     * Consume the next generated token. Returns 0..n DELTA parts - fragments of the logical span in
     * progress: {@link Part.Text} for visible content, {@link Part.Reasoning} (wrapping the
     * fragment) for think-span content, {@link Part.ToolCall} when a call span completes (whole
     * calls only; nothing is emitted mid-span so an in-progress call never leaks). Consecutive
     * deltas of the same kind belong to the same logical span until a different kind appears; a
     * Reasoning delta with EMPTY content is the span-close event (emitted only for an explicit
     * close marker, never by {@link #finish()}).
     */
    List<Part> feed(int token);

    /**
     * Flush incomplete UTF-8 bytes and close open spans (an unterminated think span is still
     * reasoning). Returns the final deltas. Idempotent.
     */
    List<Part> finish();

    /**
     * The coalesced reply after {@link #finish()}: adjacent same-kind deltas merged, think-span
     * content nested under {@link Part.Reasoning} nodes, each part carrying its verbatim payload
     * ids. Role is always assistant.
     */
    Message message();
}
