package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

/**
 * Incremental UTF-8 assembly for {@link ReplyParser} implementations: token bytes accumulate until
 * they decode to whole code points (a multi-byte sequence split across tokens is buffered, never
 * emitted as replacement characters), with the contributing token ids tracked alongside so every
 * emitted fragment carries its verbatim payload ids.
 */
public final class PendingUtf8 {

    /** A decoded run of whole code points and the token ids whose bytes produced it. */
    public record Fragment(String text, IntSequence ids) {}

    private final ByteArrayOutputStream pendingBytes = new ByteArrayOutputStream();
    private IntSequence.Builder pendingIds = IntSequence.newBuilder();
    private final CharsetDecoder utf8 =
            StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT);

    /**
     * Buffer one token's bytes, then the completed fragment - or null while a code point still
     * spans tokens.
     */
    public Fragment add(byte[] bytes, int token) {
        pendingBytes.writeBytes(bytes);
        pendingIds.add(token);
        try {
            utf8.reset();
            CharBuffer chars = utf8.decode(ByteBuffer.wrap(pendingBytes.toByteArray()));
            if (chars.isEmpty()) return null;
            pendingBytes.reset();
            return new Fragment(chars.toString(), takeIds());
        } catch (CharacterCodingException incomplete) {
            return null; // wait for a later token to complete the sequence
        }
    }

    /**
     * Drain the buffer at a span boundary or end of reply - a truncated trailing sequence decodes
     * permissively (replacement characters). Null when nothing is pending.
     */
    public Fragment flush() {
        if (pendingBytes.size() == 0) return null;
        String text = pendingBytes.toString(StandardCharsets.UTF_8);
        pendingBytes.reset();
        return new Fragment(text, takeIds());
    }

    private IntSequence takeIds() {
        IntSequence ids = pendingIds.build();
        pendingIds = IntSequence.newBuilder();
        return ids;
    }
}
