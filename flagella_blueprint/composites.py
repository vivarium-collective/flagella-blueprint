"""Composite document factory for FlagellaProcess simulations."""

from flagella_blueprint.processes import (
    BETA_DEFAULT, BETA_PRIME_DEFAULT,
)


def make_flagella_document(
    beta=None,
    beta_prime=None,
    X_max=1.0,
    X_midpoint=0.055,
    X_steepness=0.004,
    Y_max=1.0,
    Y_midpoint=0.060,
    Y_steepness=0.005,
    interval=0.001,
):
    """Build a composite document running FlagellaProcess with an emitter.

    Args:
        beta: 7 FlhDC activation coefficients. Defaults to wild-type values.
        beta_prime: 7 FliA activation coefficients. Defaults to wild-type values.
        X_max, X_midpoint, X_steepness: FlhDC profile sigmoid parameters.
        Y_max, Y_midpoint, Y_steepness: FliA profile sigmoid parameters.
        interval: dOD per process update step.

    Returns:
        dict: composite document ready for `Composite({'state': doc}, core=core)`.
    """
    return {
        'flagella': {
            '_type': 'process',
            'address': 'local:FlagellaProcess',
            'config': {
                'beta': list(beta if beta is not None else BETA_DEFAULT),
                'beta_prime': list(
                    beta_prime if beta_prime is not None else BETA_PRIME_DEFAULT),
                'X_max': X_max,
                'X_midpoint': X_midpoint,
                'X_steepness': X_steepness,
                'Y_max': Y_max,
                'Y_midpoint': Y_midpoint,
                'Y_steepness': Y_steepness,
            },
            'interval': interval,
            'inputs': {},
            'outputs': {
                'OD': ['stores', 'OD'],
                'X': ['stores', 'X'],
                'Y': ['stores', 'Y'],
                'GFP': ['stores', 'GFP'],
                'promoter_activity': ['stores', 'promoter_activity'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'OD': 'float',
                    'X': 'float',
                    'Y': 'float',
                    'GFP': 'overwrite[list]',
                    'promoter_activity': 'overwrite[list]',
                    'time': 'float',
                },
            },
            'inputs': {
                'OD': ['stores', 'OD'],
                'X': ['stores', 'X'],
                'Y': ['stores', 'Y'],
                'GFP': ['stores', 'GFP'],
                'promoter_activity': ['stores', 'promoter_activity'],
                'time': ['global_time'],
            },
        },
    }
