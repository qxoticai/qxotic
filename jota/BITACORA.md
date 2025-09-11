# Jota Bitacora

# June 9th, 2025

### Basic properties
Shapes must be immutable, any modification should return a new shape instance. Shouldn't contain negative dimensions.

Let's start with a basic Shape interface:

```java
interface Shape {
    int rank();
    long dimension(int _axis);
    long totalNumberOfElements();
}
```

### Wrap-around indexing

For better usability, I borrowed wrap-around indexing from Python.  
Wrap-around is NOT modulo-indexing e.g. `[0 ... n)` has valid wrap-around indices `[-(n - 1) ... n)`.

To be explicit: index/axis parameters that support wrap-around indexing must have a leading or trailing '_'.  
A `_` prefix => index/axis supports wrap-around wrt. input shape (this).  
A `_` suffix => index/axis supports wrap-around wrt. output shape (returned shape).  

Some methods with wrap-around indexing

```java
interface Shape { 
    
    ...

    // wrap-around wrt. input shape (this)
    Shape swap(int _axis0, int _axis1);
    Shape remove(int... _axes);
    Shape keep(int... _axes);

    // wrap-around wrt. output shape (returned shape)
    Shape insert(int axis_, long dimension);
    Shape unsqueeze(int axis_);

    // no wrap-around
    Shape permute(int... permutation);
}
```

Examples
```java
// _prefix parameters
Shape.of(2, 3, 5).dimension(-1); // 5
Shape.of(2, 3).swap(0, -1); // [3, 2]

// suffix_ parameters
Shape.of(2, 3, 5).insert(-1, 7); // [2, 3, 5, 7]
Shape.of(2, 3).unsqueeze(-1); // [2, 3, 1]
```

### Disambiguating `.squeeze()`
In Python, `squeeze()` removes all singleton (1) dimensions in a shape, but squeeze can also
take specific dimensions to remove e.g. `squeeze(2)`.  
In Java, if the method is declared as `Shape squeeze(int... _axes)` it is ambiguous.  
To disambiguate between `squeeze()` and `squeeze(new int[0])` e.g. remove all vs. remove nothing, two methods are added instead:

```java
Shape squeezeAll(); // explicit
Shape squeeze(int _axis); // most common case
```

### `[]` vs. `[0]` vs. `[1]`
The Shape interface contains a few predicates grouped in two categories: rank-based and element-based.
Rank-based include `.isScalar()`, `.isVector()` and is `.isMatrix()`.
Element-based include `.hasZeroElements()` and `.hasOneElement()`.

There's no `.isEmpty()` method to avoid confusion to which category it applies: rank or elements ... e.g. `[]` or `[0]`.
