package com.qxotic.jinfer.chat;

/**
 * What a reply is made of, named on {@link ChatEngine.Request#output}. A request names its output
 * and the reply holds exactly that - a text reply never switches into speech on its own. Text
 * arrives as {@link ChatEngine.Delta}s and {@link Content.Text}; generated media as {@link
 * ChatEngine.ReplySink#onMedia} parts, speech also as {@link Content.Speech}. A model refuses an
 * output it cannot produce.
 */
public enum Modality {
    /** Words: the transcript, when the reply also speaks. */
    TEXT,
    /** Speech in the model's own voice. */
    AUDIO,
    /** Generated images; no port draws yet. */
    IMAGE,
    /** Generated video; no port films yet. */
    VIDEO
}
