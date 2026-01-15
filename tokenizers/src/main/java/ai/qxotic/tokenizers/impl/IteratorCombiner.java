package ai.qxotic.tokenizers.impl;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

final class IteratorCombiner<T> implements Iterator<T> {
    private final List<Iterator<T>> iterators;
    private int currentIndex = 0;

    @SafeVarargs
    public static <T> Iterator<T> of(Iterator<T>... iterators) {
        return new IteratorCombiner<>(Arrays.asList(iterators));
    }

    private IteratorCombiner(List<Iterator<T>> iterators) {
        this.iterators = iterators;
    }

    @Override
    public boolean hasNext() {
        while (currentIndex < iterators.size()) {
            if (iterators.get(currentIndex).hasNext()) {
                return true;
            }
            currentIndex++;
        }
        return false;
    }

    @Override
    public T next() {
        return iterators.get(currentIndex).next();
    }
}
