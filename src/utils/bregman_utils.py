import math

MIN_LAMBDA = 1e-10
INIT_LAMBDA = 0.1

BREGMAN_LAMBDA_CONFIGS = {
    "AdaBreg": {
        0.5: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 2.0,
        },
        0.7: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 4.0,
        },
        0.9: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 8.0,
        },
        0.95: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 14.0,
        },
        0.99: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 18.0,
        },
    },
    "LinBreg": {
        0.5: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 0.001,
        },
        0.7: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 0.005,
        },
        0.9: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 0.03,
        },
        0.95: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 0.1,
        },
        0.99: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 0.3,
        },
    },
    "ProxSGD": {
        0.5: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 0.1,
        },
        0.7: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 0.5,
        },
        0.9: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 1.0,
        },
        0.95: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 2.0,
        },
        0.99: {
            "initial_lambda": INIT_LAMBDA,
            "min_lambda": MIN_LAMBDA,
            "fixed_lambda": 5.0,
        },
    },
}
# Aliases: AdaBregW and AdaBregL2 use same lambda config as AdaBreg
BREGMAN_LAMBDA_CONFIGS["AdaBregW"] = BREGMAN_LAMBDA_CONFIGS["AdaBreg"]
BREGMAN_LAMBDA_CONFIGS["AdaBregL2"] = BREGMAN_LAMBDA_CONFIGS["AdaBreg"]


def get_bregman_lambda(
    optimizer_type: str, target_sparsity: float, param_type: str
) -> float:
    """Interpolates initial_lambda or min_lambda based on target_sparsity.

    Uses logarithmic interpolation between defined sparsity anchor points.
    """
    if optimizer_type not in BREGMAN_LAMBDA_CONFIGS:
        raise ValueError(f"Unknown optimizer {optimizer_type}")

    if param_type not in ["initial_lambda", "min_lambda", "fixed_lambda"]:
        raise ValueError(f"Unknown param_type {param_type}")

    configs = BREGMAN_LAMBDA_CONFIGS[optimizer_type]

    # Exact match
    if target_sparsity in configs:
        return configs[target_sparsity][param_type]

    # Sort sparsities
    sparsities = sorted(configs.keys())

    # Extrapolate below min
    if target_sparsity < sparsities[0]:
        # Linear scaling towards 0
        scale = target_sparsity / sparsities[0]
        return configs[sparsities[0]][param_type] * scale

    # Extrapolate above max
    if target_sparsity > sparsities[-1]:
        # Log-linear extrapolation
        s1, s2 = sparsities[-2], sparsities[-1]
        v1, v2 = configs[s1][param_type], configs[s2][param_type]

        m = (math.log(v2) - math.log(v1)) / (s2 - s1)
        return math.exp(math.log(v2) + m * (target_sparsity - s2))

    # Interpolate between points
    for i in range(len(sparsities) - 1):
        s1, s2 = sparsities[i], sparsities[i + 1]
        if s1 < target_sparsity < s2:
            v1, v2 = configs[s1][param_type], configs[s2][param_type]
            # Log-linear interpolation
            weight = (target_sparsity - s1) / (s2 - s1)
            log_v1, log_v2 = math.log(v1), math.log(v2)
            return math.exp(log_v1 + weight * (log_v2 - log_v1))

    return configs[sparsities[-1]][param_type]
