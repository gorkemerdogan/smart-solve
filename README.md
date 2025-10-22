
# Smart-Solve 🧮✨

Smart-Solve is a modular **Diamond Standard (EIP-2535)** project for numerical methods on Ethereum.  
It combines **core diamond facets** with **numeric libraries** (fixed-point & floating-point) and a clean storage architecture for upgradeability.

---

## Project Structuref
contracts/
├── facets/                 # Diamond “entrypoints” (external APIs)
│   ├── core/               # Core diamond management facets
│   │   ├── DiamondCutFacet.sol      # Upgrade facet (add/replace/remove functions)
│   │   ├── DiamondLoupeFacet.sol    # Query facet (list current facets/selectors)
│   │   └── OwnershipFacet.sol       # Ownership control (owner + transferOwnership)
│   └── numeric/             # Domain-specific facets
│       └── NumericConfigFacet.sol   # Config facet (eps tolerance, max iterations)
│
├── interfaces/              # External-facing interfaces (EIP-2535 + ERCs)
│   ├── IDiamondCut.sol
│   ├── IDiamondLoupe.sol
│   ├── IERC165.sol
│   └── IERC173.sol
│
├── libraries/               # Stateless helper libraries
│   ├── numeric/             # Math backends
│   │   ├── AbdkQuad.sol         # High-precision floating-point (ABDKMathQuad)
│   │   ├── FixedPoint.sol       # Fixed-point math (PRBMath SD59x18 style)
│   │   ├── PolynomialFixed.sol  # Polynomial evaluation in fixed-point
│   │   └── Scalar.sol           # Scalar utilities (abs, clamp, nearlyEqual, etc.)
│   └── LibSmartSolve.sol        # Core diamond helper functions
│
├── storagelibs/             # Libraries that define Diamond Storage layouts
│   └── LibNumericConfig.sol    # Storage for eps + maxIter config
│
├── SmartSolve.sol           # The Diamond contract root (proxy entrypoint)

**How to Run**
*Compile*
npx hardhat compile
*Test*
npx hardhat test
*Run a Local Node*
npx hardhat node
*Deploy Locally*
npx hardhat run scripts/deploy.ts --network localhost

**Core Concepts**
*Diamond (SmartSolve.sol):*
Acts as the central proxy. Delegates calls to facets, while all storage lives in one place.

*Facets:*
Modular “controllers” that can be added/removed/upgraded via diamondCut.

*Libraries:*
Contain pure functions (no state). For example:
- FixedPoint.sol → gas-efficient math with 18-decimal precision.
- AbdkQuad.sol → high-precision math for demonstrations.
- Scalar.sol → common numeric utilities like clamp or nearlyEqual.

*Storage Libraries:*
Define layouts for persistent variables. Example: LibNumericConfig manages global tolerances (eps) and iteration limits (maxIter).

**Use Case**
Smart-Solve enables on-chain experiments with root-finding, polynomial evaluation, numerical integration, ODE solving, and matrix operations, using two math backends:
- Fixed-point → efficient, practical for Ethereum.
- Floating-point (ABDKMathQuad) → high precision, useful for research & comparisons.

**Roadmap**
- Core diamond setup (cut, loupe, ownership)
- Numeric config facet
- Fixed-point & ABDK libraries
- Add polynomial differentiation & integration facets
- Add root-finding methods (Bisection, Newton-Raphson)
- Add ODE solvers (Euler, RK4)
- Add matrix methods (Gaussian elimination, multiplication, power iteration)