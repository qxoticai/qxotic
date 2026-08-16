package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

/** Incremental UTF-8 decoding with the token ids that produced each complete fragment. */
final class PendingUtf8 {
    record Fragment(String text, IntSequence ids) {}

    private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    private IntSequence.Builder ids = IntSequence.newBuilder();
    private final CharsetDecoder utf8 =
            StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT);

    Fragment add(byte[] next, int token) {
        bytes.writeBytes(next);
        ids.add(token);
        byte[] value = bytes.toByteArray();
        ByteBuffer input = ByteBuffer.wrap(value);
        CharBuffer output = CharBuffer.allocate(value.length);
        utf8.reset();
        CoderResult result = utf8.decode(input, output, false);
        if (result.isUnderflow() && input.hasRemaining()) return null;

        String text;
        if (result.isError()) {
            text = new String(value, StandardCharsets.UTF_8);
        } else {
            output.flip();
            text = output.toString();
        }
        if (text.isEmpty()) return null;
        bytes.reset();
        return new Fragment(text, takeIds());
    }

    Fragment flush() {
        if (bytes.size() == 0) return null;
        String text = bytes.toString(StandardCharsets.UTF_8);
        bytes.reset();
        return new Fragment(text, takeIds());
    }

    private IntSequence takeIds() {
        IntSequence value = ids.build();
        ids = IntSequence.newBuilder();
        return value;
    }
}
