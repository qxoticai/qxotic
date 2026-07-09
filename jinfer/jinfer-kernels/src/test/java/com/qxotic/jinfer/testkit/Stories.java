// Shared long-prompt text for the long-history cache cases.
package com.qxotic.jinfer.testkit;

public final class Stories {

    private Stories() {}

    /**
     * ~1800 tokens: a codeword recall probe buried before a long filler story - histories long
     * enough to wrap every model's sliding window.
     */
    public static String pelican() {
        StringBuilder story =
                new StringBuilder("Remember this codeword: PELICAN. Now read this story. ");
        for (int i = 0; i < 60; i++) {
            story.append("Chapter ")
                    .append(i)
                    .append(": the river wound through the valley, past mills and ")
                    .append(
                            "orchards, while travelers traded stories of distant mountain passes"
                                    + " and the long winter. ");
        }
        story.append("After the story, tell me in one short sentence what the story is about.");
        return story.toString();
    }

    /**
     * ~2400 tokens: 90 numbered survey entries behind a summarize preamble - a long history for
     * ring-less models; probe with "How many entries were there? One number."
     */
    public static String expeditionLog() {
        StringBuilder story = new StringBuilder("Summarize the following notes.\n");
        for (int i = 0; i < 90; i++) {
            story.append("Entry ")
                    .append(i)
                    .append(": the expedition logged river depth, canopy density, ")
                    .append("and soil acidity at station ")
                    .append(i)
                    .append(
                            "; readings were nominal and the weather held clear through the"
                                    + " afternoon.\n");
        }
        return story.toString();
    }
}
