package com.llm4j.model.granite;

import com.llm4j.model.ChatFormat;
import com.llm4j.tokenizers.Tokenizer;
import com.llm4j.tokenizers.Vocabulary;
import com.llm4j.tokenizers.IntSequence;

import java.util.List;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Utility tailored for the Chat Markup Language (ChatML) prompt format.
 */
public class GraniteChatFormat extends ChatFormat {

    public final int endOfText;
    public final int fimPrefix;
    public final int fimMiddle;
    public final int fimSuffix;
    public final int fimPad;
    public final int filename;
    public final int ghStars;
    public final int issueStart;
    public final int issueComment;
    public final int issueClosed;
    public final int jupyterStart;
    public final int jupyterText;
    public final int jupyterCode;
    public final int jupyterOutput;
    public final int emptyOutput;
    public final int commitBefore;
    public final int commitMsg;
    public final int commitAfter;
    public final int reponame;
    public final int startOfRole;
    public final int endOfRole;
    public final int toolCall;

    private final Set<Integer> stopTokens;

    public GraniteChatFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();
        this.endOfText = vocabulary.id("<|end_of_text|>");
        this.startOfRole = vocabulary.id("<|start_of_role|>");
        this.endOfRole = vocabulary.id("<|end_of_role|>");
        this.toolCall = vocabulary.id("<|tool_call|>");

        this.fimPrefix = vocabulary.id("<fim_prefix>");
        this.fimMiddle = vocabulary.id("<fim_middle>");
        this.fimSuffix = vocabulary.id("<fim_suffix>");
        this.fimPad = vocabulary.id("<fim_pad>");
        this.filename = vocabulary.id("<filename>");
        this.ghStars = vocabulary.id("<gh_stars>");
        this.issueStart = vocabulary.id("<issue_start>");
        this.issueComment = vocabulary.id("<issue_comment>");
        this.issueClosed = vocabulary.id("<issue_closed>");
        this.jupyterStart = vocabulary.id("<jupyter_start>");
        this.jupyterText = vocabulary.id("<jupyter_text>");
        this.jupyterCode = vocabulary.id("<jupyter_code>");
        this.jupyterOutput = vocabulary.id("<jupyter_output>");
        this.emptyOutput = vocabulary.id("<empty_output>");
        this.commitBefore = vocabulary.id("<commit_before>");
        this.commitMsg = vocabulary.id("<commit_msg>");
        this.commitAfter = vocabulary.id("<commit_after>");
        this.reponame = vocabulary.id("<reponame>");

        this.stopTokens = Set.of(endOfText);
    }

    @Override
    public Set<Integer> stopTokens() {
        return stopTokens;
    }

    @Override
    public OptionalInt endOfText() {
        return OptionalInt.of(this.endOfText);
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(startOfRole);
        builder.addAll(this.tokenizer.encode(role.name()));
        builder.add(this.endOfRole);
        return builder.build();
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        validateRole(message.role());
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(message.role()));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        builder.add(endOfText);
        builder.addAll(this.tokenizer.encode("\n"));
        return builder.build();
    }

    @Override
    public boolean isValidRole(Role role) {
        return super.isValidRole(role)
                || List.of(
                        AVAILABLE_TOOLS.name(),
                        ASSISTANT_TOOL_CALL.name(),
                        TOOL_RESPONSE.name())
                .contains(role.name());

    }

    public static Role AVAILABLE_TOOLS = new Role("available_tools");
    public static Role ASSISTANT_TOOL_CALL = new Role("assistant_tool_call");
    public static Role TOOL_RESPONSE = new Role("tool_response");
}
