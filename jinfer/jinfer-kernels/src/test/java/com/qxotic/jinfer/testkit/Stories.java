// Shared long-prompt text for the wrapped-window cache cases.
package com.qxotic.jinfer.testkit;

public final class Stories {

    private Stories() {}

    /** ~1800 tokens: a codeword recall probe buried before a long filler story - histories long
     *  enough to wrap every model's sliding window. */
    public static String pelican() {
        StringBuilder story = new StringBuilder("Remember this codeword: PELICAN. Now read this story. ");
        for (int i = 0; i < 60; i++) {
            story.append("Chapter ").append(i).append(": the river wound through the valley, past mills and ")
                 .append("orchards, while travelers traded stories of distant mountain passes and the long winter. ");
        }
        story.append("After the story, tell me in one short sentence what the story is about.");
        return story.toString();
    }
}
