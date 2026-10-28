package com.qxotic.jinfer.chat;

/**
 * What a reply is made of, named on {@link ChatEngine.Request#output}: text, audio, or both. A
 * request names its output and the reply holds exactly that - a text reply never switches into
 * speech on its own. Text arrives as {@link ChatEngine.Delta}s and {@link Content.Text}; audio as
 * {@link ChatEngine.ReplySink#onMedia} parts and {@link Content.Speech}.
 */
public enum Modality {
    /** Words: the transcript, when the reply also speaks. */
    TEXT,
    /** Speech in the model's own voice. */
    AUDIO
}
