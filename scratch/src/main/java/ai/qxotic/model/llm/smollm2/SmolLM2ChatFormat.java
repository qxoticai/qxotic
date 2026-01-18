package ai.qxotic.model.llm.smollm2;

import ai.qxotic.model.llm.ChatFormat;
import ai.qxotic.tokenizers.IntSequence;
import ai.qxotic.tokenizers.Tokenizer;
import ai.qxotic.tokenizers.Vocabulary;
import java.util.OptionalInt;
import java.util.Set;

/** Utility tailored for Llama 3 instruct prompt format. */
public class SmolLM2ChatFormat extends ChatFormat {

    private final Set<Integer> stopTokens;

    private final int endoftext;
    private final int imStart;
    private final int imEnd;
    private final int repoName;
    private final int reponame;
    private final int fileSep;
    private final int filename;
    private final int ghStars;
    private final int issueStart;
    private final int issueComment;
    private final int issueClosed;
    private final int jupyterStart;
    private final int jupyterText;
    private final int jupyterCode;
    private final int jupyterOutput;
    private final int jupyterScript;
    private final int emptyOutput;

    public SmolLM2ChatFormat(Tokenizer tokenizer) {
        super(tokenizer);
        Vocabulary vocabulary = tokenizer.vocabulary();

        this.endoftext = vocabulary.id("<|endoftext|>");
        this.imStart = vocabulary.id("<|im_start|>");
        this.imEnd = vocabulary.id("<|im_end|>");
        this.repoName = vocabulary.id("<repo_name>");
        this.reponame = vocabulary.id("<reponame>");
        this.fileSep = vocabulary.id("<file_sep>");
        this.filename = vocabulary.id("<filename>");
        this.ghStars = vocabulary.id("<gh_stars>");
        this.issueStart = vocabulary.id("<issue_start>");
        this.issueComment = vocabulary.id("<issue_comment>");
        this.issueClosed = vocabulary.id("<issue_closed>");
        this.jupyterStart = vocabulary.id("<jupyter_start>");
        this.jupyterText = vocabulary.id("<jupyter_text>");
        this.jupyterCode = vocabulary.id("<jupyter_code>");
        this.jupyterOutput = vocabulary.id("<jupyter_output>");
        this.jupyterScript = vocabulary.id("<jupyter_script>");
        this.emptyOutput = vocabulary.id("<empty_output>");

        this.stopTokens = Set.of(endoftext, imEnd);
    }

    @Override
    public Set<Integer> stopTokens() {
        return this.stopTokens;
    }

    @Override
    public IntSequence encodeHeader(Role role) {
        validateRole(role);
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.add(this.imStart);
        builder.addAll(this.tokenizer.encode(role.name()));
        builder.addAll(this.tokenizer.encode("\n"));
        return builder.build();
    }

    @Override
    public IntSequence encodeMessage(Message message) {
        validateRole(message.role());
        IntSequence.Builder builder = IntSequence.newBuilder();
        builder.addAll(encodeHeader(message.role()));
        builder.addAll(this.tokenizer.encode(message.textContent().strip()));
        builder.add(this.imEnd);
        builder.addAll(this.tokenizer.encode("\n"));
        return builder;
    }

    @Override
    public OptionalInt endOfText() {
        return OptionalInt.of(this.endoftext);
    }
}
