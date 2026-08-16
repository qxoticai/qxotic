/**
 * Model boundaries and the lifetime rules shared by every architecture.
 *
 * <p>{@link com.qxotic.jinfer.x.boundary.Model} binds immutable configuration and weights to a
 * mutable {@link com.qxotic.jinfer.x.boundary.RuntimeState}. Context models add bounded,
 * incremental ingestion; language, embedding, reranking and speech interfaces add only their
 * model-specific projection.
 *
 * <p>States are single serial pipelines. Public model operations hold {@link
 * com.qxotic.jinfer.x.boundary.RuntimeState#exclusively(java.lang.Runnable) exclusive access}; a
 * concurrent operation fails fast, while {@code close()} waits for an active operation and then
 * releases owned resources exactly once. Same-thread nesting is allowed for model composition.
 *
 * <p>Memory follows one rule: whoever supplies an arena owns it. A state created without an arena
 * owns and closes its internal arena. A state created with a {@link
 * com.qxotic.jota.memory.MemoryArena} borrows it; the caller must close that arena only after the
 * state and every operation using it. Projected {@link com.qxotic.jota.memory.MemoryView}s are
 * borrowed unless an API explicitly says otherwise. Callback APIs delimit their validity: copy
 * inside the callback to retain a result.
 */
package com.qxotic.jinfer.x.boundary;
