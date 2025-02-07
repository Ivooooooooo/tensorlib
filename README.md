# Tensorlib: Numerical Tensor Library in Modern C++

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/)
[![Header-only](https://img.shields.io/badge/header--only-brightgreen)](https://github.com/tensorlib/tensorlib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tensorlib is a numerical tensor library inspired by NumPy, designed for C++20. It provides a modern and efficient interface for operations with multidimensional tensors, including broadcasting, slicing, and advanced mathematical operations.

---

## Main Features

- **Multidimensional Tensors**: Support for tensors of arbitrary rank.
- **Automatic Broadcasting**: Operations between tensors of different shapes.
- **Slicing and Views**: Efficient access to subregions without copying data.
- **Mathematical Operations**: Addition, multiplication, elemental functions (sqrt, exp, log).
- **STL-compatible**: Standard iterators and algorithms.
- **Type-safe**: Use of C++20 concepts for type safety.
- **Efficiency**: Optimized for intensive numerical operations.

---

## Usage Example

```cpp
#include "tensorlib.hpp"
#include <iostream>

int main() {
    using namespace tensorlib;

    // Create tensors
    Tensor<double, 2> A({2, 2}, 3.0);
    Tensor<double, 1> B({2}, {1.0, 2.0});

    // Broadcasting: A + B
    auto C = A + B.reshape({2, 1});

    // Display results
    std::cout << "A:\n" << A(0, 0) << " " << A(0, 1) << "\n"
              << A(1, 0) << " " << A(1, 1) << "\n";

    std::cout << "B:\n" << B(0) << " " << B(1) << "\n";

    std::cout << "A + B (broadcasting):\n"
              << C(0, 0) << " " << C(0, 1) << "\n"
              << C(1, 0) << " " << C(1, 1) << "\n";

    // Mathematical operations
    auto D = C.sqrt();
    std::cout << "sqrt(A + B):\n"
              << D(0, 0) << " " << D(0, 1) << "\n"
              << D(1, 0) << " " << D(1, 1) << "\n";

    return 0;
}
```

**Expected Output:**
```
A:
3.0 3.0
3.0 3.0

B:
1.0 2.0

A + B (broadcasting):
4.0 5.0
4.0 5.0

sqrt(A + B):
2.0 2.23607
2.0 2.23607
```

---

## Installation

Tensorlib is a header-only library. Just include tensorlib.hpp in your project.

```bash
git clone https://github.com/tensorlib/tensorlib.git
cd tensorlib
```

---

## Requirements

- C++20-compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- CMake (optional, for tests and examples)

---

## Contributions

Contributions are welcome! Open an issue or submit a pull request.
