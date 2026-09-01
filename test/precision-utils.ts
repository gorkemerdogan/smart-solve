export type PrecisionClassification =
    | "binary128-aware comparison"
    | "fixed-point/truncated comparison"
    | "JavaScript-number-limited comparison"
    | "method-reference comparison";

export interface PrecisionMetadata {
    classification: PrecisionClassification;
    comparisonScale: string;
    oraclePrecision: string;
    conversion: string;
    claim: string;
}

const BINARY128_FRACTION_BITS = 112n;
const BINARY128_EXPONENT_BIAS = 16383n;
const BINARY128_EXPONENT_MASK = 0x7fffn;
const BINARY128_FRACTION_MASK = (1n << BINARY128_FRACTION_BITS) - 1n;

export interface ExactRational {
    num: bigint;
    den: bigint;
}

/** Return the exact rational value represented by a finite binary128 word. */
export function binary128ToRational(value: string): ExactRational {
    if (!/^0x[0-9a-fA-F]{32}$/.test(value)) {
        throw new Error(`Expected a 16-byte binary128 hex value, received: ${value}`);
    }

    const bits = BigInt(value);
    const negative = (bits >> 127n) === 1n;
    const exponentBits = (bits >> BINARY128_FRACTION_BITS) & BINARY128_EXPONENT_MASK;
    const fraction = bits & BINARY128_FRACTION_MASK;
    if (exponentBits === BINARY128_EXPONENT_MASK) {
        throw new Error("NaN and infinity do not have finite rational values");
    }
    if (exponentBits === 0n && fraction === 0n) return { num: 0n, den: 1n };

    let num = exponentBits === 0n ? fraction : (1n << BINARY128_FRACTION_BITS) + fraction;
    const binaryExponent = exponentBits === 0n
        ? 1n - BINARY128_EXPONENT_BIAS - BINARY128_FRACTION_BITS
        : exponentBits - BINARY128_EXPONENT_BIAS - BINARY128_FRACTION_BITS;
    let den = 1n;
    if (binaryExponent >= 0n) num <<= binaryExponent;
    else den <<= -binaryExponent;

    while (den > 1n && (num & 1n) === 0n) {
        num >>= 1n;
        den >>= 1n;
    }
    return { num: negative ? -num : num, den };
}

/**
 * Decode a finite IEEE-754 binary128 value directly from bytes16 and truncate
 * it toward zero at the requested decimal scale. This avoids Solidity
 * harness conversion scales and avoids JavaScript Number entirely.
 */
export function binary128ToScaledInt(value: string, decimalDigits: number): bigint {
    if (!/^0x[0-9a-fA-F]{32}$/.test(value)) {
        throw new Error(`Expected a 16-byte binary128 hex value, received: ${value}`);
    }
    if (!Number.isInteger(decimalDigits) || decimalDigits < 0) {
        throw new Error(`Invalid decimal digit count: ${decimalDigits}`);
    }

    const rational = binary128ToRational(value);
    const decimalScale = 10n ** BigInt(decimalDigits);
    return (rational.num * decimalScale) / rational.den;
}

/** Parse a base-10 string into a scaled integer, truncating toward zero. */
export function decimalStringToScaledInt(value: string, decimalDigits: number): bigint {
    const match = value.trim().match(/^([+-]?)(\d+)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/);
    if (!match) throw new Error(`Invalid decimal value: ${value}`);

    const negative = match[1] === "-";
    const integer = match[2];
    const fraction = match[3] ?? "";
    const exponent = Number(match[4] ?? "0");
    const digits = BigInt(integer + fraction);
    const power = exponent - fraction.length + decimalDigits;
    const magnitude = power >= 0
        ? digits * (10n ** BigInt(power))
        : digits / (10n ** BigInt(-power));
    return negative ? -magnitude : magnitude;
}

export function formatScaledInt(value: bigint, decimalDigits: number): string {
    const scale = 10n ** BigInt(decimalDigits);
    const negative = value < 0n;
    const magnitude = negative ? -value : value;
    const integer = magnitude / scale;
    const fraction = (magnitude % scale).toString().padStart(decimalDigits, "0");
    const rendered = decimalDigits === 0
        ? integer.toString()
        : `${integer}.${fraction}`.replace(/\.?0+$/, "");
    return negative ? `-${rendered}` : rendered;
}

export function printPrecisionMetadata(metadata: PrecisionMetadata): void {
    console.log(`Precision Class      : ${metadata.classification}`);
    console.log(`Comparison Scale     : ${metadata.comparisonScale}`);
    console.log(`Oracle Precision     : ${metadata.oraclePrecision}`);
    console.log(`Conversion Path      : ${metadata.conversion}`);
    console.log(`Supported Claim      : ${metadata.claim}`);
}
