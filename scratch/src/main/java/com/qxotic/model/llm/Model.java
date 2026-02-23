package com.qxotic.model.llm;

/**
 * Generic interface for transformer language models that process tokens sequentially using
 * attention and feed-forward layers. This interface provides the core functionality needed to
 * perform inference with transformer-based language models, including token ingestion, state
 * management, and logits computation.
 *
 * <p>The model operates in three main steps: 1. State initialization for the inference session 2.
 * Token ingestion to process input sequences 3. Logits computation to generate probability
 * distributions over the vocabulary
 *
 * @param <Configuration> Configuration parameters including model architecture details like
 *     embedding dimensions, number of layers, attention heads, and context length. This defines the
 *     model's structure and capabilities.
 * @param <Weights> Model weights including token embeddings, attention weights, feed-forward
 *     weights, and layer norms. These are the learned parameters that determine the model's
 *     behavior.
 * @param <State> Mutable model state including activation buffers and key-value caches for
 *     attention. This maintains the running state during sequence processing.
 */
public interface Model<Configuration, Weights, State> {

    /**
     * Returns the model's configuration parameters that define its architecture and capabilities.
     * The configuration includes essential parameters such as: - Embedding dimensions - Number of
     * transformer layers - Number of attention heads - Maximum context length - Vocabulary size -
     * Other architecture-specific parameters
     *
     * @return The model configuration containing architecture parameters
     */
    Configuration configuration();

    /**
     * Creates a new mutable state object for model inference. The state object holds all temporary
     * buffers and caches needed during sequence processing, including attention key-value caches
     * and layer activations.
     *
     * @param maxBatchSize The maximum batch size this state should support. This determines the
     *     size of internal buffers and caches.
     * @return A new State object initialized for the specified batch size
     */
    State createNewState(int maxBatchSize);

    /**
     * Processes a sequence of input tokens and updates the model state accordingly. This method
     * performs the forward pass through the embedding layer and transformer layers, updating the
     * key-value caches and other state information.
     *
     * @param weights The model weights to use for processing
     * @param state The current model state to update
     * @param tokens Array of input token IDs to process
     */
    void ingestTokens(Weights weights, State state, int[] tokens);

    /**
     * Computes logits (unnormalized probabilities) for the next token given the current model
     * state. This method typically performs the final linear projection to vocabulary size and can
     * be followed by a softmax operation to obtain normalized probabilities.
     *
     * @param weights The model weights to use for computation
     * @param state The current model state containing the necessary activations
     */
    void computeLogits(Weights weights, State state);
}
