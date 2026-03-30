package com.qxotic.tokenizers.impl;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.IntSequences;
import java.util.Objects;

/**
 * An abstract base implementation of the {@link IntSequence} interface that provides common
 * functionality for sequences of integers.
 *
 * <p>This class implements the basic operations like {@link #toString()}, {@link #equals(Object)},
 * {@link #hashCode()}, and {@link #compareTo(IntSequence)} that are common to all integer sequence
 * implementations. Subclasses need only implement the core methods defined in the {@link
 * IntSequence} interface.
 *
 * @see IntSequence
 */
public abstract class AbstractIntSequence implements IntSequence {

    /**
     * Returns a string representation of this sequence using default delimiters. The elements are
     * separated by ", " and enclosed in square brackets.
     *
     * @return a string representation of this sequence
     * @see IntSequence#toString(CharSequence, CharSequence, CharSequence)
     */
    @Override
    public String toString() {
        return toString(", ", "[", "]");
    }

    /**
     * Computes a hash code for this sequence. The hash code is computed using the standard
     * algorithm for sequences: for each element, multiply the running hash by 31 and add the
     * element value.
     *
     * @return a hash code value for this sequence
     */
    @Override
    public int hashCode() {
        int hash = 1;
        for (int i = 0; i < length(); ++i) {
            hash = hash * 31 + intAt(i);
        }
        return hash;
    }

    /**
     * Compares this sequence with another object for equality. Two sequences are considered equal
     * if they have the same length and contain the same elements in the same order.
     *
     * @param other the object to compare with
     * @return true if the specified object represents a sequence equivalent to this sequence
     */
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        return other instanceof IntSequence
                && IntSequences.contentEquals(this, (IntSequence) other);
    }

    /**
     * Compares this sequence with another sequence lexicographically. The comparison is based on
     * the values of the elements of the sequences.
     *
     * @param other the sequence to compare with
     * @return a negative value if this sequence is less than the other sequence, zero if they are
     *     equal, or a positive value if this sequence is greater
     */
    @Override
    public int compareTo(IntSequence other) {
        return IntSequences.compare(this, Objects.requireNonNull(other));
    }
}
