package ai.llm4j.test;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

class StreamingDecoder {
    private final CharsetDecoder decoder;
    private ByteBuffer remainingBytes;

    private static final ByteBuffer EMPTY = ByteBuffer.allocate(0);

    public StreamingDecoder() {
        this.decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPLACE)
                .onMalformedInput(CodingErrorAction.REPLACE);
        this.remainingBytes = EMPTY;
    }

    public void reset() {
        this.remainingBytes = EMPTY;
    }

    public String flush() {
        return decode(new byte[0], true);
    }

    public String decode(byte[] bytes, boolean endOfInput) {
        // Combine remaining bytes with new input
        ByteBuffer input = ByteBuffer.allocate(remainingBytes.remaining() + bytes.length);
        input.put(remainingBytes);
        input.put(bytes);
        input.flip();

        // Create output buffer
        CharBuffer output = CharBuffer.allocate(input.remaining());

        // Decode
        decoder.decode(input, output, endOfInput);

        // Store remaining bytes
        remainingBytes = ByteBuffer.allocate(input.remaining());
        remainingBytes.put(input);
        remainingBytes.flip();

        // Return decoded string
        output.flip();
        return output.toString();
    }
}
