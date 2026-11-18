package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.Base64;
import java.util.zip.CRC32;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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

    @Test
    void aPathThatIsNotAnImageSaysWhatItIs(@TempDir Path dir) throws IOException {
        Path missing = dir.resolve("typo.png");
        assertThrows(NoSuchFileException.class, () -> new ImageIoDecoder().load(missing));
        IOException folder = assertThrows(IOException.class, () -> new ImageIoDecoder().load(dir));
        assertTrue(folder.getMessage().endsWith("is a directory"), folder.getMessage());
        Path text = Files.writeString(dir.resolve("notes.png"), "not an image");
        IOException notImage =
                assertThrows(IOException.class, () -> new ImageIoDecoder().load(text));
        assertTrue(notImage.getMessage().contains("cannot read"), notImage.getMessage());
    }

    /** WEBP is not a JDK format: the refusal names the decoder that reads it. */
    @Test
    void aFormatTheJdkCannotReadNamesTheRemedy() {
        byte[] webp =
                Base64.getDecoder().decode("UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==");
        IOException refused =
                assertThrows(IOException.class, () -> new ImageIoDecoder().decode(webp));
        assertTrue(
                refused.getMessage().endsWith("-Djinfer.imageDecoder=ffmpeg reads the rest"),
                refused.getMessage());
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
