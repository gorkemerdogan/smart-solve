
# Smart-Solve
**SMART-SOLVE** is a high-precision numerical computing suite for the Ethereum Virtual Machine (EVM). It implements **IEEE-754 quadruple precision** arithmetic within a modular **Diamond Standard (EIP-2535)** architecture, enabling complex scientific, engineering, and deterministic simulations on-chain.

---

## Key Features

* **Quadruple Precision:** Full IEEE-754 binary128 implementation using `bytes16` (~34 decimal digits).
* **Deterministic Execution:** Identical results across all EVM-compatible nodes for verifiable research.
* **Comprehensive Math Suite:** Advanced calculus, linear algebra, and iterative solvers.
* **Stateless Libraries:** Optimized numerical logic decoupled from storage.

---

## Feature Breakdown

### 1. Matrix Operations (`MatrixMaster.sol`)
* **Creation:** Zeros, Ones, Identity, Diagonal, deterministic pseudo-random matrices (seeded).
* **Sparse Support:** CSR matrices including zero, identity, diagonal, and COO-to-CSR conversion.
* **Access & Manipulation:** Element access and mutation (row-major layout).
* **Reshape & Slicing:** Contiguous submatrix extraction and shape-preserving reshape.
* **Transformations:** Matrix transpose.
* **Elementwise Arithmetic:** Addition, subtraction, scalar multiplication, scalar division.
* **Multiplication:** Matrix–matrix, matrix–vector, sparse matrix–vector (CSR).
* **Vector Ops:** Dot product, Euclidean norm, normalization, convergence checks.
* **Determinant:** LU-based determinant with partial pivoting.
* **Inversion:** Gauss–Jordan elimination for square matrices.
* **Eigen Analysis:** Dominant eigenvalue and eigenvector via power iteration.

### 2. Linear Solvers & Optimization (`LinearSolvers.sol`)
* **Direct Solvers:** Gaussian elimination with partial pivoting.
* **LU Decomposition:** Doolittle LU factorization (no pivoting).
* **Iterative Solvers:** Jacobi and Gauss–Seidel methods.
* **Least Squares:** Fixed-step gradient descent.

### 3. Numerical Differentiation (`Differentiation.sol`)
* **Methods:** Forward difference, backward difference, and centered difference schemes.
* **Accuracy:** First-order O(h) for forward/backward differences, second-order O(h²) for centered difference.
* **Step Size Control:** User-defined step size with global configuration fallback and safe default.

### 4. Numerical Integration (`Integration.sol`)
* **Methods:** Composite trapezoidal rule, Simpson’s 1/3 rule, and Simpson’s 3/8 rule.
* **Accuracy:** Second-order accuracy for trapezoidal rule, fourth-order accuracy for Simpson-based rules under smooth integrands.
* **Validation:** Enforces method-specific constraints (even 'n' for Simpson 1/3, 'n % 3 == 0' for Simpson 3/8).

### 5. Ordinary Differential Equation (ODE) Solvers (`ODESolver.sol`)
* **Methods:** Euler’s method, RK2 (Midpoint), RK2 (Heun / Improved Euler), and classical RK4.
* **Accuracy:** First-order accuracy for Euler, second-order accuracy for RK2 methods, and fourth-order accuracy for RK4.
* **Design Scope:** Single-step integrators intended for deterministic on-chain simulation and controlled step advancement.

### 6. Polynomial Utilities (`Polynomial.sol`)
* **Evaluation:** Polynomial evaluation using Horner’s method, including optimized evaluation for monic polynomials.
* **Derivatives:** Exact polynomial derivative computation and combined evaluation.
* **Algebraic Operations:** Coefficient-wise addition, subtraction, scalar multiplication, and polynomial multiplication via convolution.
* **Integration:** Exact computation of the indefinite integral polynomial with configurable constant of integration.
* **Division:** Synthetic division by linear factors of the form \((x - r)\), returning quotient and remainder.
* **Utilities:** Degree computation, trailing-zero trimming, and canonical zero-polynomial handling.

### 7. Root Finding (`RootFinding.sol`)
* **Methods:** Bisection, Newton–Raphson (analytic derivative), and Secant (derivative-free).
  * Bisection enforces sign change on the initial interval.  
  * Newton method guards against zero derivatives.  
  * Secant method guards against zero slope between iterates.
* **Convergence Control:** User-specified tolerance with global configuration fallback and enforced minimum tolerance.

### 8. Trigonometry Stack (`Trigonometry.sol`)
* **Methods:** sin, cos, tan, cot, arcsin, arccos, and arctan, internally routed through dedicated sine/cosine, tangent/cotangent, and inverse-trigonometry modules.
* **Design Scope:** Deterministic, pure-function trigonometric primitives intended for on-chain numerical analysis, optimization, differential equations, and geometric algorithms.

---

## System Architecture

SMART-SOLVE is structured as a layered numerical stack designed to maximize code reuse, numerical precision, and EVM bytecode efficiency while enabling advanced scientific computation on-chain.
The architecture separates core floating-point arithmetic, numerical constants, and high-level numerical algorithms, ensuring that heavy quad-precision logic is compiled once and reused across all modules.

---

### Core Math Layer

This layer provides the numerical foundation for the entire system and is the only place where raw ABDK quad logic is directly referenced.

* **`MathLib.sol`**  
  Thin public-call wrapper around `ABDKMathQuad` exposing IEEE-754 binary128 (`bytes16`) operations such as add, sub, zero checks, absolute value, and integer <-> quad conversions.
  All numerical libraries call `MathLib` instead of inlining ABDK logic, drastically reducing bytecode size and preventing EVM contract size limit violations.

* **`QuadConstants.sol`**  
  Centralized repository of high-precision constants and tolerances including π, π/2, rational helpers (1/2, 1/6), and predefined epsilons (1e-6 → 1e-30).
  Provides a single authoritative source for numerical thresholds and avoids duplicated or inconsistent constants.

---

## Project Structure

```text
contracts/
├── facets/
│   ├── core/              # Diamond management (Cut, Loupe, Ownership)
│   │   ...
│   ├── numeric/
│   │   ├── DifferentiationFacet.sol
│   │   ├── IntegrationFacet.sol
│   │   ├── LinearSolversFacet.sol
│   │   ├── MatrixMasterFacet.sol
│   │   ├── NumericConfigFacet.sol
│   │   ├── ODESolverFacet.sol
│   │   ├── PolynomialFacet.sol
│   │   ├── RootFindingFacet.sol
│   │   └── TrigonometryFacet.sol
│
├── interfaces/
│   │   ...                # EIP-2535 and Custom Interfaces
├── libraries/
│   ├── numeric/
│   │   ├── Differentiation.sol
│   │   ├── Integration.sol
│   │   ├── LinearSolvers.sol
│   │   ├── MatrixMaster.sol
│   │   ├── ODESolver.sol
│   │   ├── Polynomial.sol
│   │   └── RootFinding.sol
│   │
│   ├── trigonometry/
│   │   ├── Trigonometry.sol
│   │   ├── TrigonometryArc.sol
│   │   ├── TrigonometrySinCos.sol
│   │   └── TrigonometryTanCot.sol
│   │
│   ├── LibSmartSolve.sol
│   ├── MathLib.sol        # Math library depends on ABDKMathQuad
│   └── QuadConstants.sol  # Math constants
│   │   ...
```
## Precision & Performance

SMART-SOLVE is engineered for environments where standard `uint256` fixed-point math is insufficient. By implementing the IEEE-754 standard, we provide floating-point capabilities directly on the EVM.

### Technical Specifications

| Parameter | Specification | Details |
| :--- | :--- | :--- |
| **Standard** | IEEE-754 binary128 | Quadruple precision floating-point format |
| **Storage Type** | `bytes16` | Compact representation for stack efficiency |
| **Precision** | ~34 Decimal Digits | High-fidelity scientific significand |

> [!WARNING]
> **Gas Profile**: Computation in quad-precision is intensive. Operations like `Matrix Inversion` or `RK4 Integration` are optimized for **Correctness** and **Reproducibility** rather than low-cost DeFi swaps. Use this suite for high-value simulations, research, and engineering logic where accuracy is non-negotiable.

---

## License

Distributed under the **MIT License**.

```text
Copyright (c) 2026 SMART-SOLVE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
