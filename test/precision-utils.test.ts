import { expect } from "chai";
import {
    binary128ToRational,
    binary128ToScaledInt,
    decimalStringToScaledInt,
} from "./precision-utils";

describe("precision-utils", function () {
    it("decodes exact normal binary128 values without JavaScript Number", function () {
        const one = "0x3fff0000000000000000000000000000";
        const minusOne = "0xbfff0000000000000000000000000000";
        const half = "0x3ffe0000000000000000000000000000";

        expect(binary128ToScaledInt(one, 33)).to.equal(10n ** 33n);
        expect(binary128ToScaledInt(minusOne, 33)).to.equal(-(10n ** 33n));
        expect(binary128ToScaledInt(half, 33)).to.equal(5n * 10n ** 32n);
        expect(binary128ToRational(half)).to.deep.equal({ num: 1n, den: 2n });
    });

    it("parses high-precision decimal strings without rounding through Number", function () {
        expect(
            decimalStringToScaledInt("1.521379706804567569604080832254438", 33)
        ).to.equal(1_521_379_706_804_567_569_604_080_832_254_438n);
        expect(decimalStringToScaledInt("-1.25e-3", 12)).to.equal(-1_250_000_000n);
    });
});
