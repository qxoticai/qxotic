/**
 * Tensor operations and transformations.
 *
 * <p><b>What is a Tensor?</b>
 *
 * <p>In Jota, a Tensor is not merely a multi-dimensional array of data - it is a <em>representation
 * of computation</em>. Tensors are:
 *
 * <ul>
 *   <li><b>Lazy:</b> Operations on tensors do not execute immediately. Instead, they build a
 *       computation graph that is only materialized when explicitly requested.
 *   <li><b>Opaque:</b> Operations return Tensors, not values. Calling {@code tensor.sum()} does not
 *       return a scalar value - it returns a new Tensor representing the sum operation. The actual
 *       computation happens later when the tensor is materialized.
 *   <li><b>Compute:</b> Tensors are a means of computation, not just storage. A Tensor represents
 *       an operation or a chain of operations that can be compiled into optimized kernels.
 *   <li><b>Location-agnostic:</b> Tensors can exist anywhere - in system RAM, on a GPU, distributed
 *       across devices, or even be fully virtual (representing computations without backing
 *       storage).
 * </ul>
 *
 * <p><b>Example: Lazy Evaluation</b>
 *
 * <pre>
 * Tensor a = ...;
 * Tensor b = ...;
 * Tensor c = a.add(b);        // No computation happens here
 * Tensor d = c.sum();         // Still no computation
 * d.materialize();            // Now the entire graph is compiled and executed
 * </pre>
 *
 * <p><b>Controlled Indexing</b>
 *
 * <p>Jota supports structured indexing operations that preserve laziness, such as {@code slice},
 * {@code gather}, and {@code scatter}. Unlike other libraries where arbitrary indexing (e.g.,
 * {@code tensor[i][j]}) forces immediate materialization, Jota's indexing operations return new
 * Tensors that become part of the computation graph. Indiscriminate array-style indexing is not
 * supported to maintain the lazy evaluation model.
 *
 * <p><b>Key Operations</b>
 *
 * <ul>
 *   <li>Mathematical operations (add, multiply, etc.)
 *   <li>Shape transformations (reshape, transpose, broadcast)
 *   <li>Reduction operations (sum, mean, max)
 *   <li>Element-wise operations
 *   <li>Structured indexing (slice, gather, scatter)
 * </ul>
 *
 * @since 0.1.0
 */
package com.qxotic.jota.tensor;
