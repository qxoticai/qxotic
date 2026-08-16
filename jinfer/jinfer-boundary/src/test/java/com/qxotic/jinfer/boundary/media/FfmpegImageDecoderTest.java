package com.qxotic.jinfer.boundary.media;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Test;

class FfmpegImageDecoderTest {

    @Test
    void parsesACompletePpmAndRejectsTruncation() throws Exception {
        byte[] header = "P6\n1 1\n255\n".getBytes(StandardCharsets.US_ASCII);
        byte[] ppm = new byte[header.length + 3];
        System.arraycopy(header, 0, ppm, 0, header.length);
        ppm[header.length] = (byte) 255;

        var image = FfmpegImageDecoder.parsePpm(ppm);
        assertEquals(1, image.width());
        assertEquals(1, image.height());
        assertThrows(IOException.class, () -> FfmpegImageDecoder.parsePpm(header));
    }

    @Test
    void hostileDimensionsFailAsDecodeErrors() {
        for (String dimensions : List.of("0 1", "1 -1", "2147483647 2147483647")) {
            byte[] ppm = ("P6\n" + dimensions + "\n255\n").getBytes(StandardCharsets.US_ASCII);
            assertThrows(IOException.class, () -> FfmpegImageDecoder.parsePpm(ppm), dimensions);
        }
    }
}
