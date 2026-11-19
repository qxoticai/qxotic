package com.qxotic.jinfer.chat;

/**
 * What a reply is made of, set on {@link ChatEngine.Request#output}: text alone, the default, never
 * switches into speech; text and audio is a spoken reply with its transcript; audio alone is speech
 * without one. The request decides, never the prompt. A model never produces what it lacks, and
 * refuses a request when it can produce none of them - audio alone from a model that cannot speak;
 * text and audio from it is answered in text. Text arrives as {@link ChatEngine.Delta}s and {@link
 * Content.Text}; generated media as {@link ChatEngine.ReplySink#onMedia} parts, speech also whole
 * as the reply's {@link Content.Media}.
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
