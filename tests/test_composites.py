"""Integration tests: full Composite assembly and run."""

import pytest
from process_bigraph import (
    Composite, allocate_core, gather_emitter_results,
)
from process_bigraph.emitter import RAMEmitter
from flagella_blueprint.processes import FlagellaProcess
from flagella_blueprint.composites import make_flagella_document


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('FlagellaProcess', FlagellaProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_make_document_structure():
    doc = make_flagella_document()
    assert 'flagella' in doc
    assert 'stores' in doc
    assert 'emitter' in doc
    assert doc['flagella']['address'] == 'local:FlagellaProcess'


def test_composite_runs(core):
    doc = make_flagella_document(interval=0.001)
    sim = Composite({'state': doc}, core=core)
    sim.run(0.05)  # advance through 50 process intervals
    assert sim.state['stores']['OD'] > 0


def test_composite_emits(core):
    doc = make_flagella_document(interval=0.005)
    sim = Composite({'state': doc}, core=core)
    sim.run(0.05)
    results = gather_emitter_results(sim)
    # results is keyed by emitter path tuple
    assert len(results) >= 1
    series = list(results.values())[0]
    assert len(series) > 1
    # Final OD should be ~0.05
    assert series[-1]['OD'] == pytest.approx(0.05, rel=0.05)
    # GFP should be a 7-list
    assert len(series[-1]['GFP']) == 7
