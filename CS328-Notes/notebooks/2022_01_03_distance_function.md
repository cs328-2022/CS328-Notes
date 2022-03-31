# Distance Function

After object representation, next we need distance functions for objects.

Distance
: A function $d$, given two objects $x$, $y$; produces a real number.

$$ d(x, y) \rightarrow \mathbb{R} \geq 0$$

### Nice to have metric properties

It would be nice, if my distance function have these below properties. But it is not necessary for the distance function to follow any of the property below.

Nice if, he distance between two different objects is always greater than zero.
```{math}
:label: property_1
d(x, y) &= 0 \rightarrow x = y \\
d(x, y) &\geq 0
```


It would be nice if symmetric.
```{math}
:label: property_2
d(x, y) = d(y, x)
```

Nice, if follows triangle inequality.
```{math}
:label: property_3
d(x, y) + d(y, z) \geq d(x, z)
```

```{caution}
The above three properties are "nice to have" and not necessary for a distance function to follow.
```

Also good, if we are able to calculate "average".

Average
: $\min_{x} \sum^{}_{i} d(p_{i}, x)$. The object $x$ which minimizes the sum of distance from the objects in a given set to $x$.

It would be nice, if average can be calculated easily.



Eucledian Distance
: $d(x, y) = \sqrt{\sum_{i} (x_{i} - y{i})^{2}}$. The length of line segment joint vector $x$ and $y$ in Eucledian space.

Eucledian distance follows all of the nice properties (including "easy average") that we have mentioned above.

<footer>
Author(s): Sachin Yadav
</footer>
