package ai.qxotic.jota;

public final class Util {
    public static int wrapAround(int _index, int size) {
        assert size >= 0;
        int index = _index >= 0 ? _index : _index + size;
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException("wrap-around index out of bounds");
        }
        return index;
    }
}
