"""FlagellaProcess: class 2 flagella gene network dynamic model.

Implements the bilinear SUM-gate model from Kalir & Alon (Cell 2004):

    P_i(OD) = beta_i * X(OD) + beta'_i * Y(OD)
    dGFP_i / dOD = P_i(OD)

for 7 class 2 promoters (fliL, fliE, fliF, flgB, flgA, flhB, fliA),
where X(OD) is the effective FlhDC activity profile and Y(OD) the
effective FliA activity profile.

The Process treats OD (cell density) as the time variable. Each
update(interval=dOD) advances OD by dOD and integrates GFP_i with a
midpoint rule on a fine substep.

Parameters X_midpoint, X_steepness, Y_midpoint, Y_steepness shape
analytic sigmoids that mimic the recovered X(OD), Y(OD) profiles in
Figure 4C of the paper. All parameters are configurable, enabling the
reprogramming experiments of Figures 5B (modify beta_i) and 5C (modify
X_max).
"""

import numpy as np
from process_bigraph import Process


PROMOTER_NAMES = ['fliL', 'fliE', 'fliF', 'flgB', 'flgA', 'flhB', 'fliA']

BETA_DEFAULT = [1200.0, 450.0, 350.0, 350.0, 150.0, 100.0, 50.0]
BETA_PRIME_DEFAULT = [250.0, 350.0, 300.0, 450.0, 300.0, 350.0, 300.0]


class FlagellaProcess(Process):
    """Class 2 flagella gene network dynamics (Kalir & Alon 2004).

    Time variable is OD (optical density); intervals are dOD. Outputs
    the cumulative GFP/OD per promoter, the instantaneous promoter
    activity, and the underlying FlhDC and FliA effective activities.

    Config:
        beta: 7 FlhDC activation coefficients (GFP/OD)
        beta_prime: 7 FliA activation coefficients (GFP/OD)
        X_max, X_midpoint, X_steepness: FlhDC profile sigmoid (decreasing)
        Y_max, Y_midpoint, Y_steepness: FliA profile sigmoid (increasing)
        substep: max dOD per integration substep (numerical accuracy)
    """

    config_schema = {
        'beta': {'_type': 'overwrite[list]', '_default': list(BETA_DEFAULT)},
        'beta_prime': {
            '_type': 'overwrite[list]', '_default': list(BETA_PRIME_DEFAULT)},
        'X_max': {'_type': 'float', '_default': 1.0},
        'X_midpoint': {'_type': 'float', '_default': 0.055},
        'X_steepness': {'_type': 'float', '_default': 0.004},
        'Y_max': {'_type': 'float', '_default': 1.0},
        'Y_midpoint': {'_type': 'float', '_default': 0.060},
        'Y_steepness': {'_type': 'float', '_default': 0.005},
        'substep': {'_type': 'float', '_default': 1e-4},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._od = 0.0
        self._beta = np.asarray(self.config['beta'], dtype=float)
        self._beta_prime = np.asarray(self.config['beta_prime'], dtype=float)
        if self._beta.shape != self._beta_prime.shape:
            raise ValueError(
                f'beta and beta_prime must have same length; '
                f'got {self._beta.shape} and {self._beta_prime.shape}')
        self._gfp = np.zeros_like(self._beta)

    def evaluate_X(self, od):
        """FlhDC effective activity profile (decreasing sigmoid)."""
        od = np.asarray(od, dtype=float)
        return self.config['X_max'] / (
            1.0 + np.exp((od - self.config['X_midpoint']) /
                          self.config['X_steepness']))

    def evaluate_Y(self, od):
        """FliA effective activity profile (increasing sigmoid)."""
        od = np.asarray(od, dtype=float)
        return self.config['Y_max'] / (
            1.0 + np.exp(-(od - self.config['Y_midpoint']) /
                           self.config['Y_steepness']))

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'OD': 'overwrite[float]',
            'X': 'overwrite[float]',
            'Y': 'overwrite[float]',
            'GFP': 'overwrite[list]',
            'promoter_activity': 'overwrite[list]',
        }

    def _read_state(self):
        x = float(self.evaluate_X(self._od))
        y = float(self.evaluate_Y(self._od))
        Pi = self._beta * x + self._beta_prime * y
        return {
            'OD': float(self._od),
            'X': x,
            'Y': y,
            'GFP': self._gfp.tolist(),
            'promoter_activity': Pi.tolist(),
        }

    def initial_state(self):
        return self._read_state()

    def update(self, state, interval):
        substep = self.config['substep']
        n_substeps = max(1, int(np.ceil(interval / substep)))
        d_od = interval / n_substeps
        for _ in range(n_substeps):
            mid = self._od + 0.5 * d_od
            Pi_mid = (self._beta * float(self.evaluate_X(mid)) +
                      self._beta_prime * float(self.evaluate_Y(mid)))
            self._gfp += Pi_mid * d_od
            self._od += d_od
        return self._read_state()
