"""Unit tests for FlagellaProcess."""

import numpy as np
import pytest
from process_bigraph import allocate_core
from flagella_blueprint.processes import (
    FlagellaProcess, PROMOTER_NAMES, BETA_DEFAULT, BETA_PRIME_DEFAULT,
)


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('FlagellaProcess', FlagellaProcess)
    return c


def test_instantiation(core):
    proc = FlagellaProcess(config={}, core=core)
    assert len(proc.config['beta']) == 7
    assert len(proc.config['beta_prime']) == 7
    assert proc.config['beta'] == BETA_DEFAULT
    assert proc.config['beta_prime'] == BETA_PRIME_DEFAULT


def test_promoter_names_length():
    assert len(PROMOTER_NAMES) == 7
    assert PROMOTER_NAMES[0] == 'fliL'
    assert PROMOTER_NAMES[-1] == 'fliA'


def test_initial_state(core):
    proc = FlagellaProcess(config={}, core=core)
    state = proc.initial_state()
    assert state['OD'] == 0.0
    assert state['X'] > 0.99  # FlhDC fully active at OD=0
    assert state['Y'] < 0.01  # FliA inactive at OD=0
    assert state['GFP'] == [0.0] * 7
    assert len(state['promoter_activity']) == 7


def test_X_Y_profiles(core):
    proc = FlagellaProcess(config={}, core=core)
    # X drops, Y rises across the OD axis
    od = np.linspace(0, 0.1, 11)
    X = proc.evaluate_X(od)
    Y = proc.evaluate_Y(od)
    assert X[0] > X[-1]   # decreasing
    assert Y[0] < Y[-1]   # increasing
    assert X[0] > 0.99
    assert X[-1] < 0.01
    assert Y[0] < 0.01
    assert Y[-1] > 0.99


def test_single_update_advances_OD(core):
    proc = FlagellaProcess(config={}, core=core)
    proc.initial_state()
    result = proc.update({}, interval=0.01)
    assert result['OD'] == pytest.approx(0.01, rel=1e-9)
    # Early phase: GFP grows roughly linearly with beta * OD
    fliL_gfp = result['GFP'][0]
    assert fliL_gfp == pytest.approx(BETA_DEFAULT[0] * 0.01, rel=0.02)


def test_full_run_cumulative_GFP(core):
    proc = FlagellaProcess(config={}, core=core)
    proc.initial_state()
    od_step = 0.001
    n_steps = 100  # final OD = 0.1
    for _ in range(n_steps):
        result = proc.update({}, interval=od_step)
    assert result['OD'] == pytest.approx(0.1, rel=1e-6)
    gfp = np.array(result['GFP'])
    # All promoters should have positive cumulative GFP
    assert np.all(gfp > 0)
    # fliL (strongest beta) reaches highest cumulative GFP
    assert gfp[0] == max(gfp)
    # fliA (weakest beta) reaches lowest cumulative GFP
    assert gfp[-1] == min(gfp)


def test_promoter_hierarchy_preserved_early(core):
    """In early phase (X dominant), promoter_activity should follow beta order."""
    proc = FlagellaProcess(config={}, core=core)
    state = proc.initial_state()  # OD=0, X≈1, Y≈0
    activity = state['promoter_activity']
    # Should be approximately equal to beta values
    for i in range(7):
        assert activity[i] == pytest.approx(BETA_DEFAULT[i], rel=0.01)


def test_promoter_hierarchy_late(core):
    """In late phase (Y dominant), promoter_activity should follow beta_prime order."""
    proc = FlagellaProcess(config={}, core=core)
    proc.initial_state()
    # Advance to OD=0.1 (X≈0, Y≈1)
    proc.update({}, interval=0.1)
    state = proc._read_state()
    activity = state['promoter_activity']
    # Should be approximately equal to beta_prime values
    for i in range(7):
        assert activity[i] == pytest.approx(BETA_PRIME_DEFAULT[i], rel=0.01)


def test_reprogramming_beta1_lowers_fliL(core):
    """Reducing beta_fliL (Fig 5B) should reduce fliL maximal expression."""
    wt = FlagellaProcess(config={}, core=core)
    wt.initial_state()
    wt.update({}, interval=0.1)

    weak_beta = list(BETA_DEFAULT)
    weak_beta[0] = 200.0  # 6x weaker than wild-type 1200
    mut = FlagellaProcess(config={'beta': weak_beta}, core=core)
    mut.initial_state()
    mut.update({}, interval=0.1)

    assert mut._gfp[0] < wt._gfp[0]


def test_X_max_scaling_collapses_timing(core):
    """High X_max (Fig 5C) should make all promoters reach normalized 50%
    at nearly the same OD (timing differences vanish)."""
    proc_high = FlagellaProcess(config={'X_max': 50.0}, core=core)
    proc_high.initial_state()
    od_step = 0.001
    n_steps = 100
    gfp_history = []
    for _ in range(n_steps):
        r = proc_high.update({}, interval=od_step)
        gfp_history.append(list(r['GFP']))
    gfp_arr = np.array(gfp_history)  # (steps, 7)
    gmax = gfp_arr[-1]
    norm = gfp_arr / gmax
    # OD at which each promoter crosses 0.5
    od_axis = np.linspace(od_step, n_steps * od_step, n_steps)
    half_od = []
    for i in range(7):
        idx = np.argmax(norm[:, i] >= 0.5)
        half_od.append(od_axis[idx])
    # With high X_max, the spread between earliest and latest should be tiny
    assert (max(half_od) - min(half_od)) < 0.01


def test_outputs_schema(core):
    proc = FlagellaProcess(config={}, core=core)
    outputs = proc.outputs()
    for port in ('OD', 'X', 'Y', 'GFP', 'promoter_activity'):
        assert port in outputs
