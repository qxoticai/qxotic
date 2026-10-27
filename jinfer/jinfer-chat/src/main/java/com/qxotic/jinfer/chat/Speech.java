package com.qxotic.jinfer.chat;

/**
 * A spoken reply, requested: speech with its transcript interleaved, or speech alone. A request
 * without one is answered in text, and the reply never switches into speech on its own. Only a
 * model that speaks takes one; the speech arrives through {@link ChatEngine.ReplySink#onSpeech} and
 * as the reply's {@link Content.Speech}.
 *
 * @param transcript whether the words come as text too, interleaved with the speech
 */
public record Speech(boolean transcript) {

    /** Speech and the words as text: a voice assistant's reply. */
    public static Speech withTranscript() {
        return new Speech(true);
    }

    /** Speech alone. */
    public static Speech only() {
        return new Speech(false);
    }
}
