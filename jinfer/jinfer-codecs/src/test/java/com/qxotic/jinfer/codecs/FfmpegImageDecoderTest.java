package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.media.Media;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class FfmpegImageDecoderTest {

    @Test
    void theSignatureNamesTheDemuxer() {
        assertEquals(
                "png_pipe",
                FfmpegImageDecoder.demuxer(new byte[] {(byte) 0x89, 'P', 'N', 'G', 0, 0, 0, 0}));
        assertEquals(
                "jpeg_pipe",
                FfmpegImageDecoder.demuxer(new byte[] {(byte) 0xff, (byte) 0xd8, (byte) 0xff}));
        assertEquals(
                "gif", FfmpegImageDecoder.demuxer("GIF89a".getBytes(StandardCharsets.US_ASCII)));
        assertEquals(
                "webp_pipe",
                FfmpegImageDecoder.demuxer(
                        "RIFF\0\0\0\0WEBPVP8 ".getBytes(StandardCharsets.US_ASCII)));
        assertEquals(
                "bmp_pipe", FfmpegImageDecoder.demuxer("BM".getBytes(StandardCharsets.US_ASCII)));
        assertEquals(
                "tiff_pipe",
                FfmpegImageDecoder.demuxer("II*\0".getBytes(StandardCharsets.US_ASCII)));
        assertNull(
                FfmpegImageDecoder.demuxer("hello".getBytes(StandardCharsets.US_ASCII)),
                "unknown bytes are probed");
        assertNull(FfmpegImageDecoder.demuxer(new byte[0]));
    }

    @Test
    void aCorruptPngIsRefusedNotProbedAsSomethingElse() throws Exception {
        Assumptions.assumeTrue(ffmpegOnPath(), "needs ffmpeg on PATH");
        FfmpegImageDecoder decoder = new FfmpegImageDecoder();
        byte[] corrupt = new byte[24];
        corrupt[0] = (byte) 0x89;
        corrupt[1] = 'P';
        corrupt[2] = 'N';
        corrupt[3] = 'G';
        IOException refused = assertThrows(IOException.class, () -> decoder.decode(corrupt));
        assertTrue(refused.getMessage().contains("ffmpeg"), refused.getMessage());
        BufferedImage red = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) red.setRGB(x, y, 0xff0000);
        ByteArrayOutputStream png = new ByteArrayOutputStream();
        ImageIO.write(red, "png", png);
        Media.Image image = decoder.decode(png.toByteArray());
        assertEquals(8, image.width());
        assertEquals(8, image.height());
    }

    private static boolean ffmpegOnPath() {
        try {
            return new ProcessBuilder("ffmpeg", "-version")
                            .redirectErrorStream(true)
                            .start()
                            .waitFor()
                    == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

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
