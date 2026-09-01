# Numerical benchmark precision methodology

The contracts operate on IEEE-754 binary128 (`bytes16`) values. That fact alone
does not make every reported error a binary128-precision measurement. Accuracy
claims depend on the output conversion and on the precision of the reference
oracle.

All accuracy reports should state:

- precision classification;
- comparison scale or resolution;
- oracle precision and construction;
- conversion path;
- absolute error, relative error, threshold, and pass/fail result.

`toFloat` helpers multiply by a decimal scale and call `MathLib.toInt`, which
truncates toward zero. They are compatibility/reporting helpers, not lossless
binary128 conversions.

| Suite/path | Classification | Effective comparison/oracle | Supported interpretation |
| --- | --- | --- | --- |
| `differentiation.gen.test.ts` | binary128-aware comparison | Direct `bytes16` decode at `1e30`; exact rational polynomial method-error formulas | 30-decimal reporting of differentiation method error |
| `rootfinding.gen.test.ts` | binary128-aware comparison | Direct `bytes16` decode at `1e33`; 33+ digit root constants or exact roots | Root error against a `1e-28` threshold |
| `polynomial.acc.test.ts` | binary128-aware comparison | Direct `bytes16` decode at `1e30`; exact BigInt rational oracle evaluated on the encoded binary128 inputs | Polynomial operation error separated from input-representation error |
| `integration.gen.test.ts` | method-reference and JavaScript-number-limited | `1e12` truncating output; JavaScript `Number` quadrature and analytic references | Lower-precision agreement with the named quadrature method |
| `ode-solver.gen.test.ts` | method-reference and JavaScript-number-limited | `1e12` truncating output; JavaScript `Number` Euler/RK and exact references | Lower-precision agreement with the named ODE discretization |
| `matrixmaster.acc.test.ts` | mixed fixed-point/JavaScript-number-limited | `1e12` truncating output; exact integer references for some paths and JS `Number` for determinant/eigenvalue paths | Accuracy only at each reported path and tolerance |
| `linear-solvers.gen.test.ts` | fixed-point/truncated comparison | `1e12` truncating output with BigInt residual/error arithmetic | `1e12`-scale solution and residual agreement |
| `optimization.acc.test.ts` | fixed-point/truncated comparison | `1e12` truncating output with known optima and BigInt errors | `1e12`-scale optimizer agreement at method-specific tolerances |
| matrix gas/add/sub/multiply suites | fixed-point/truncated comparison | generally `1e12` output with BigInt error arithmetic | Fixed-point correctness accompanying gas measurements |
| polynomial gas suite | reporting-only fixed-point conversion | harness scale; no high-precision accuracy oracle | Gas/reporting results only |
| optimization gas suite | JavaScript-number approximation for reporting | approximate binary128-to-`Number` display | Gas and qualitative convergence reporting only |
| diamond smoke tests | fixed-point/truncated comparison | harness-specific `1e12` or `1e18` output | Interface/integration correctness, not precision evidence |

The JavaScript-number-limited suites must not be cited as evidence of
approximately 34-decimal binary128 accuracy. To support that claim for
transcendental integration or ODE cases, use an independently generated
high-precision oracle (for example MPFR with precision and rounding mode
recorded) and compare it to direct binary128 decoding.
