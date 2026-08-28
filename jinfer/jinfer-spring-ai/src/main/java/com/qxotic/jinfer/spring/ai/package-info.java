/**
 * Spring AI adapters for in-process jinfer models.
 *
 * <p>{@link com.qxotic.jinfer.spring.ai.JinferChatModel} supplies blocking and streaming chat,
 * tools, structured output and multimodal prompts; configure generation through {@link
 * com.qxotic.jinfer.spring.ai.JinferChatOptions}. {@link
 * com.qxotic.jinfer.spring.ai.JinferEmbeddingModel} supplies embeddings, and {@link
 * com.qxotic.jinfer.spring.ai.JinferDocumentPostProcessor} reranks retrieved documents.
 *
 * <p>THE GRAMMAR GUARANTEES THE FORM; THE PROMPT SETS THE PLAN. A mask says "not that token" at
 * each step - it never says "here is the target". The model decides WHAT to write from the prompt
 * alone, and the mask edits how that decision is spelled. When the two agree the grammar never
 * fires; when they disagree the grammar wins on syntax and the CONTENT pays, because a greedy
 * decoder cannot revise a prefix it has committed to. Asked to extract "Apollo 11 launched on 16
 * July 1969" into a record with a {@code LocalDate}, with the schema given ONLY as a grammar,
 * LFM2.5-2.6B answered {@code "1677-11-19"}: it began writing the DAY, the date rule admits only
 * digits, and that 16 was stranded in the year field. Perfectly valid JSON against the schema,
 * which is what makes it dangerous - nothing throws.
 *
 * <p>On this stack that gap is usually filled for you: {@code BeanOutputConverter} states the
 * schema in the prompt itself. It does NOT on the two paths that hand the job to the provider -
 * {@code useProviderStructuredOutput}, and setting {@link
 * com.qxotic.jinfer.spring.ai.JinferChatOptions#getOutputSchema} yourself - and this adapter does
 * not append anything to your prompt. On those paths, say the shape you are about to enforce.
 * Measured on the equivalent langchain4j battery, stating it is worth 36 vs 32 of 39 on LFM2.5-2.6B
 * and 34 vs 28 on gemma-4-26B: the effect does not shrink with model size, it grows.
 *
 * <p>SAY IT IN PLAIN WORDS, not schema jargon: "Write dates as YYYY-MM-DD" fixed the case above
 * where the same schema carrying {@code "format": "date"} did not, reliably. An example of the
 * output you want is better still, and better yet is to CONSTRAIN WHAT YOU ALREADY ASKED FOR -
 * "answer yes or no" against a two-way choice cannot diverge. Put it in your system prompt and weld
 * it with {@code withCachedPrompt}: a fixed schema in a fixed prefix is prefilled once, where the
 * same text on each request is re-prefilled every time.
 *
 * <p>The grammar is still doing what a prompt cannot: an unconstrained run of that same request
 * returned the right date wrapped in a {@code ```json} fence, and elsewhere returned prose. Fences,
 * preambles, invented fields, truncated strings and malformed JSON are unrepresentable rather than
 * unlikely. Prompt for CORRECT, constrain for WELL-FORMED.
 *
 * <p>DESCRIBE EVERY TOOL. A local model decides from the declaration alone whether a tool applies,
 * and a small checkpoint skips an undescribed one and answers from its own knowledge instead.
 * Spring AI builds the declaration by reflecting the method, so a tool without {@code description},
 * without {@code @ToolParam(description = "...")} on its parameters, or compiled without {@code
 * -parameters} (which leaves the argument names as {@code arg0}, {@code arg1}) reaches the model
 * with nothing to reason about. Measured, same model and question: given {@code add(arg0, arg1)}
 * with no descriptions, LFM2.5-8B-A1B answered "the provided tools are not applicable for basic
 * arithmetic calculations" and called nothing; given a described {@code add(a, b)} it called the
 * tool on the first try.
 *
 * <p>Builders accept a local path, a model reference, or an already loaded model. One adapter is
 * one serial inference pipeline. Chat caching stays intentionally small: {@code retainSessions}
 * keeps recent conversations live, {@code promptCache} mounts one persisted catalog, and {@code
 * withCachedPrompt} defines a reusable prefix.
 */
package com.qxotic.jinfer.spring.ai;
