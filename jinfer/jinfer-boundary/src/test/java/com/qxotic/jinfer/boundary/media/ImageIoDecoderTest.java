package com.qxotic.jinfer.boundary.media;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.zip.CRC32;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

class ImageIoDecoderTest {

    @Test
    void decodesOrdinaryImagesButRejectsHostileDimensionsBeforePixels() throws IOException {
        ByteArrayOutputStream png = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB), "png", png);
        var image = new ImageIoDecoder().decode(png.toByteArray());
        assertEquals(1, image.width());
        assertEquals(1, image.height());

        IOException failure =
                assertThrows(
                        IOException.class,
                        () -> new ImageIoDecoder().decode(hugePng(png.toByteArray())));
        assertTrue(failure.getMessage().contains("pixel limit"), failure.getMessage());
    }

    private static byte[] hugePng(byte[] small) {
        byte[] png = small.clone();
        ByteBuffer.wrap(png).putInt(16, 5_000).putInt(20, 5_000);
        CRC32 crc = new CRC32();
        crc.update(png, 12, 17); // IHDR type + data
        ByteBuffer.wrap(png).putInt(29, (int) crc.getValue());
        return png;
    }
}
