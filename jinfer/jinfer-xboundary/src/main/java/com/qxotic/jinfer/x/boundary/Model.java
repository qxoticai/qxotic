package com.qxotic.jinfer.x.boundary;

/** A loaded model's immutable configuration and weights, and its runtime-state type. */
public interface Model<C, W, S extends RuntimeState> {

    C configuration();

    W weights();
}
