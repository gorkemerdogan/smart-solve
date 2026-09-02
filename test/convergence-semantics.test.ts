import { expect } from "chai";
import { ethers } from "hardhat";

describe("Convergence semantics regressions", function () {
    let steepest: any;
    let linearObjective: any;
    let sphericalObjective: any;
    let root: any;
    let linear: any;

    before(async function () {
        const mathFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const math = await mathFactory.deploy();
        await math.waitForDeployment();
        const libraries = { MathLib: await math.getAddress() };

        steepest = await (await ethers.getContractFactory("SteepestDescentHarness", { libraries })).deploy();
        linearObjective = await (await ethers.getContractFactory("LinearObjectiveHarness", { libraries })).deploy();
        sphericalObjective = await (await ethers.getContractFactory("SphericalObjectiveHarness", { libraries })).deploy();
        root = await (await ethers.getContractFactory("RootFindingHarness", { libraries })).deploy();
        linear = await (await ethers.getContractFactory("LinearSolversHarness", { libraries })).deploy();

        await Promise.all([
            steepest.waitForDeployment(),
            linearObjective.waitForDeployment(),
            sphericalObjective.waitForDeployment(),
            root.waitForDeployment(),
            linear.waitForDeployment(),
        ]);
    });

    it("does not treat an objective zero crossing as optimizer convergence", async function () {
        const zero = await steepest.qFromInt(0n);
        const tol = await steepest.qFromFrac(1n, 10n ** 12n);
        const [, , iterations, status] = await steepest.solve(
            await linearObjective.getAddress(),
            [zero],
            3n,
            tol
        );

        expect(iterations).to.equal(3n);
        expect(status).to.equal(3n); // STATUS_MAX_ITER_EXCEEDED
    });

    it("returns STATUS_ZERO_GRADIENT for an effectively stationary point", async function () {
        const tiny = await steepest.qFromFrac(1n, 10n ** 20n);
        const tol = await steepest.qFromFrac(1n, 10n ** 18n);
        const [, , iterations, status] = await steepest.solve(
            await sphericalObjective.getAddress(),
            [tiny],
            5n,
            tol
        );

        expect(iterations).to.equal(0n);
        expect(status).to.equal(1n); // STATUS_ZERO_GRADIENT
    });

    it("classifies a tiny Newton step with a large residual as stagnation", async function () {
        const zero = await root.qFromInt(0n);
        const one = await root.qFromInt(1n);
        const tol = await root.qFromFrac(1n, 10n ** 18n);
        const fSelector = root.interface.getFunction("f_constant_one")!.selector;
        const dfSelector = root.interface.getFunction("df_very_large")!.selector;
        const [approximation, iterations, converged, residual] = await root.rootFindingNewton(
            await root.getAddress(),
            fSelector,
            await root.getAddress(),
            dfSelector,
            zero,
            tol,
            10n
        );

        expect(approximation).not.to.equal(zero);
        expect(iterations).to.equal(1n);
        expect(converged).to.equal(false);
        expect(residual).to.equal(one);
    });

    it("rejects small Jacobi and Gauss-Seidel changes when residuals remain large", async function () {
        const huge = await linear.qFromInt(10n ** 20n);
        const one = await linear.qFromInt(1n);
        const minusOne = await linear.qFromInt(-1n);
        const zero = await linear.qFromInt(0n);
        const tol = await linear.qFromFrac(1n, 10n ** 30n);
        const A = [huge, huge, huge, huge];
        const b = [one, minusOne];
        const x0 = [zero, zero];

        const [, jacobiIterations, jacobiConverged] = await linear.jacobiWithStatus(
            2n, A, b, x0, 1n, tol
        );
        const [, gsIterations, gsConverged] = await linear.gaussSeidelWithStatus(
            2n, A, b, x0, 1n, tol
        );

        expect(jacobiIterations).to.equal(1n);
        expect(gsIterations).to.equal(1n);
        expect(jacobiConverged).to.equal(false);
        expect(gsConverged).to.equal(false);
    });

    it("distinguishes final-iteration convergence from maxIter exhaustion", async function () {
        const one = await linear.qFromInt(1n);
        const two = await linear.qFromInt(2n);
        const zero = await linear.qFromInt(0n);
        const huge = await linear.qFromInt(10n ** 20n);
        const minusOne = await linear.qFromInt(-1n);
        const tol = await linear.qFromFrac(1n, 10n ** 30n);

        const [, convergedIterations, converged] = await linear.jacobiWithStatus(
            2n,
            [one, zero, zero, one],
            [one, two],
            [zero, zero],
            2n,
            tol
        );
        const [, exhaustedIterations, exhausted] = await linear.jacobiWithStatus(
            2n,
            [huge, huge, huge, huge],
            [one, minusOne],
            [zero, zero],
            2n,
            tol
        );

        expect(convergedIterations).to.equal(2n);
        expect(exhaustedIterations).to.equal(2n);
        expect(converged).to.equal(true);
        expect(exhausted).to.equal(false);
    });
});
