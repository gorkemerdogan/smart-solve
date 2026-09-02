import { expect } from "chai";
import { ethers } from "hardhat";
import type { Contract } from "ethers";

type MatrixHarness = Contract & {
    qFromInt(value: bigint): Promise<string>;
    qFromFrac(numerator: bigint, denominator: bigint): Promise<string>;
    detHarness(rows: bigint, cols: bigint, data: string[]): Promise<string>;
    inverseHarness(rows: bigint, cols: bigint, data: string[]): Promise<[bigint, bigint, string[]]>;
};

type LinearHarness = Contract & {
    qFromInt(value: bigint): Promise<string>;
    qFromFrac(numerator: bigint, denominator: bigint): Promise<string>;
    gaussianElimination(n: bigint, A: string[], b: string[]): Promise<string[]>;
    luDecomposition(n: bigint, A: string[]): Promise<[string[], string[]]>;
};

describe("Scale-aware elimination pivot policy", function () {
    let matrix: MatrixHarness;
    let linear: LinearHarness;

    before(async function () {
        const mathFactory = await ethers.getContractFactory(
            "contracts/libraries/MathLib.sol:MathLib"
        );
        const math = await mathFactory.deploy();
        await math.waitForDeployment();
        const libraries = { MathLib: await math.getAddress() };

        const matrixFactory = await ethers.getContractFactory("MatrixMasterHarness", { libraries });
        matrix = await matrixFactory.deploy() as unknown as MatrixHarness;
        await matrix.waitForDeployment();

        const linearFactory = await ethers.getContractFactory("LinearSolversHarness", { libraries });
        linear = await linearFactory.deploy() as unknown as LinearHarness;
        await linear.waitForDeployment();
    });

    it("preserves the determinant of a tiny, well-scaled 1x1 matrix", async function () {
        const tiny = await matrix.qFromFrac(1n, 10n ** 31n);
        expect(await matrix.detHarness(1n, 1n, [tiny])).to.equal(tiny);
    });

    it("inverts a tiny, well-scaled 1x1 matrix", async function () {
        const tiny = await matrix.qFromFrac(1n, 10n ** 19n);
        const expected = await matrix.qFromInt(10n ** 19n);
        const [rows, cols, inverse] = await matrix.inverseHarness(1n, 1n, [tiny]);

        expect(rows).to.equal(1n);
        expect(cols).to.equal(1n);
        expect(inverse[0]).to.equal(expected);
    });

    it("accepts uniformly scaled well-conditioned matrices", async function () {
        const baseA = await Promise.all([
            matrix.qFromInt(2n), matrix.qFromInt(1n),
            matrix.qFromInt(1n), matrix.qFromInt(2n),
        ]);
        const scale = 10n ** 31n;
        const scaledA = await Promise.all([
            matrix.qFromFrac(2n, scale), matrix.qFromFrac(1n, scale),
            matrix.qFromFrac(1n, scale), matrix.qFromFrac(2n, scale),
        ]);

        expect(await matrix.detHarness(2n, 2n, baseA)).not.to.equal(
            "0x00000000000000000000000000000000"
        );
        await expect(matrix.inverseHarness(2n, 2n, baseA)).not.to.be.reverted;
        expect(await matrix.detHarness(2n, 2n, scaledA)).not.to.equal(
            "0x00000000000000000000000000000000"
        );
        await expect(matrix.inverseHarness(2n, 2n, scaledA)).not.to.be.reverted;
    });

    it("classifies the same relatively near-singular matrix consistently", async function () {
        const one = await matrix.qFromInt(1n);
        const onePlusTiny = await matrix.qFromFrac(10n ** 31n + 1n, 10n ** 31n);
        const A = [one, one, one, onePlusTiny];

        expect(await matrix.detHarness(2n, 2n, A)).to.equal(
            "0x00000000000000000000000000000000"
        );
        await expect(matrix.inverseHarness(2n, 2n, A)).to.be.revertedWith(
            "MatrixMaster: singular matrix"
        );

        const linearOne = await linear.qFromInt(1n);
        const linearOnePlusTiny = await linear.qFromFrac(10n ** 31n + 1n, 10n ** 31n);
        const linearA = [linearOne, linearOne, linearOne, linearOnePlusTiny];
        await expect(linear.gaussianElimination(2n, linearA, [linearOne, linearOne])).to.be.revertedWith(
            "LinearSolversMM: unsafe pivot"
        );
        await expect(linear.luDecomposition(2n, linearA)).to.be.revertedWith(
            "LinearSolversMM: unsafe pivot LU"
        );

        const scaledA = await Promise.all([
            matrix.qFromFrac(1n, 10n ** 20n),
            matrix.qFromFrac(1n, 10n ** 20n),
            matrix.qFromFrac(1n, 10n ** 20n),
            matrix.qFromFrac(10n ** 31n + 1n, 10n ** 51n),
        ]);
        expect(await matrix.detHarness(2n, 2n, scaledA)).to.equal(
            "0x00000000000000000000000000000000"
        );
        await expect(matrix.inverseHarness(2n, 2n, scaledA)).to.be.revertedWith(
            "MatrixMaster: singular matrix"
        );
    });

    it("lets Gaussian elimination and LU solve a tiny, well-scaled 1x1 system", async function () {
        const tiny = await linear.qFromFrac(1n, 10n ** 31n);
        const twiceTiny = await linear.qFromFrac(2n, 10n ** 31n);
        const two = await linear.qFromInt(2n);
        const one = await linear.qFromInt(1n);

        expect(await linear.gaussianElimination(1n, [tiny], [twiceTiny])).to.deep.equal([two]);
        const [L, U] = await linear.luDecomposition(1n, [tiny]);
        expect(L).to.deep.equal([one]);
        expect(U).to.deep.equal([tiny]);
    });

    it("rejects pivots tiny relative to the matrix scale in both linear solvers", async function () {
        const one = await linear.qFromInt(1n);
        const tiny = await linear.qFromFrac(1n, 10n ** 31n);
        const zero = await linear.qFromInt(0n);
        const A = [one, zero, zero, tiny];

        await expect(linear.gaussianElimination(2n, A, [one, tiny])).to.be.revertedWith(
            "LinearSolversMM: unsafe pivot"
        );
        await expect(linear.luDecomposition(2n, A)).to.be.revertedWith(
            "LinearSolversMM: unsafe pivot LU"
        );

        const matrixOne = await matrix.qFromInt(1n);
        const matrixTiny = await matrix.qFromFrac(1n, 10n ** 31n);
        const matrixZero = await matrix.qFromInt(0n);
        const matrixA = [matrixOne, matrixZero, matrixZero, matrixTiny];
        expect(await matrix.detHarness(2n, 2n, matrixA)).to.equal(
            "0x00000000000000000000000000000000"
        );
        await expect(matrix.inverseHarness(2n, 2n, matrixA)).to.be.revertedWith(
            "MatrixMaster: singular matrix"
        );
    });
});
