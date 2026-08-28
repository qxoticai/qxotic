package com.qxotic.jinfer.examples.inflect2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/**
 * WHICH player runs is decided by the OS, by trying each in turn, so there is nothing here to test
 * about the search. What is worth pinning is how each one is ASKED - the flags and the quoting that
 * a version bump or a space in a path would otherwise break silently.
 */
class PlayerTest {

    @Test
    void ffplayUsesTheDemuxerOptionsNotTheShorthands() {
        String line = String.join(" ", Player.ffplay(24000));
        // ffplay 9 removed -ac, and -ar means something else here: this is the stable spelling
        assertTrue(line.contains("-sample_rate 24000"), line);
        assertTrue(line.contains("-ch_layout mono"), line);
        assertFalse(line.contains(" -ac "), line);
        assertFalse(line.contains(" -ar "), line);
    }

    @Test
    void aplayIsToldTheSameFormat() {
        assertArrayEquals("aplay -f S16_LE -r 24000 -c 1 -".split(" "), Player.aplay(24000));
    }

    @Test
    void afplayIsHandedThePath() {
        assertArrayEquals(
                new String[] {"afplay", "/tmp/speech.wav"},
                Player.afplay(Path.of("/tmp/speech.wav")));
    }

    @Test
    void soundPlayerPlaysSynchronouslyAndQuotesThePath() {
        String[] command = Player.soundPlayer(Path.of("C:\\Temp\\it's here.wav"));
        String script = command[command.length - 1];
        // PlaySync, or powershell exits and cuts the audio off mid-word
        assertTrue(script.contains("PlaySync()"), script);
        // a quote in the path is doubled, which is how PowerShell escapes one
        assertTrue(script.contains("'C:\\Temp\\it''s here.wav'"), script);
    }

    // ── a player that quits ───────────────────────────────────────────────

    @Test
    void aDeadPlayerReportsItsStatusNotAStreamClosed() throws IOException {
        // `false` exits at once, so its stdin is a dead pipe by the time we close it. The bytes
        // that close() flushes into it used to throw "Stream closed" out of --play, naming our
        // own pipe for a failure one process away; the player's exit status is the diagnosis.
        Player player = new Player("false");
        player.offer(new byte[4096]); // may or may not reach the pipe: either is fine

        Player.Failed failed = assertThrows(Player.Failed.class, player::close);
        assertEquals(1, failed.status, "the player's status, so it can become ours");
        assertFalse(failed.getMessage().contains("Stream closed"), failed.getMessage());
    }

    @Test
    void aPlayerThatSucceedsClosesQuietly() throws IOException {
        Player player = new Player("true");
        player.offer(new byte[4096]);
        player.close();
    }
}
