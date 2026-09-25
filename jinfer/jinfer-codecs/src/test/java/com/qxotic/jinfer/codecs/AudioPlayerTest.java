package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

class AudioPlayerTest {

    @Test
    void eachPlatformUsesItsOwnPlayerAndStreamingTransport(@TempDir Path dir) {
        Path wav = dir.resolve("speech.wav");
        assertEquals("afplay", AudioPlayer.fileCommands("Mac OS X", wav)[0][0]);
        assertEquals("aplay", AudioPlayer.fileCommands("Linux", wav)[0][0]);
        assertEquals("powershell.exe", AudioPlayer.fileCommands("Windows 11", wav)[0][0]);
        for (String os : List.of("Mac OS X", "Linux", "Windows 11"))
            assertEquals("ffplay", AudioPlayer.fileCommands(os, wav)[1][0]);
        assertTrue(AudioPlayer.streamsWav("Mac OS X"));
        assertTrue(
                AudioPlayer.streamsWav("Windows 11"), "Windows streaming needs no ffplay install");
        assertFalse(AudioPlayer.streamsWav("Linux"));
    }

    @Test
    @EnabledOnOs(OS.WINDOWS)
    void windowsLoadsUnicodePathsAndReportsBadWavs(@TempDir Path dir) throws Exception {
        Path wav = dir.resolve("O'Brien says こんにちは.wav");
        Files.write(wav, AudioCodec.wav(new Media.Audio(new float[2400], 24000, 1)));
        String[] command = AudioPlayer.fileCommands("Windows 11", wav)[0];
        int encoded = command.length - 1;
        String script =
                new String(Base64.getDecoder().decode(command[encoded]), StandardCharsets.UTF_16LE);
        // Exercise the real PowerShell/.NET load and disposal, replacing only the audible call.
        script =
                script.replace(
                        "$player.PlaySync();",
                        "[Console]::Write([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($player.SoundLocation)));");
        command[encoded] =
                Base64.getEncoder().encodeToString(script.getBytes(StandardCharsets.UTF_16LE));
        byte[] reported = Base64.getDecoder().decode(Subprocess.run(List.of(command), null));
        assertEquals(wav.toAbsolutePath().toString(), new String(reported, StandardCharsets.UTF_8));

        // Load must stop before PlaySync can substitute a system beep for a corrupt file.
        Files.writeString(wav, "not a WAV");
        String[] invalid = AudioPlayer.fileCommands("Windows 11", wav)[0];
        IOException failure =
                assertThrows(IOException.class, () -> Subprocess.run(List.of(invalid), null));
        assertTrue(failure.getMessage().contains("exited"), failure.getMessage());
    }
}
