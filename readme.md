# toy-gemm: 2D matrices in pure C++

## Assumptions:
* C++17 compiler (tested on g++ 8.2 & clang++ 7.0.0)

## Limitations:
* only works with 2D dense matrices
* only tested with numeric types (int, float, double, complex); needs to have * defined

## Features: 
* list initialization (building 2 by 3 matrix from `{{1,2,3},{4,5,6}}` rather than `{1,2,3,4,5,6}`)
* O(n) space compile time transpose and multiplication
* efficient access to a view of a column for copying & modification
* convenience functions identity(), zeros(), ones()
