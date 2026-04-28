# flagella-blueprint

Process-bigraph dynamic model of the *E. coli* class 2 flagella gene network,
reproducing Figures 4 and 5 of:

> Kalir S & Alon U. **Using a Quantitative Blueprint to Reprogram the Dynamics
> of the Flagella Gene Network.** *Cell* **117**, 713–720 (2004).

A single `FlagellaProcess` implements the bilinear SUM-gate model

```
P_i(OD) = β_i · X(OD) + β'_i · Y(OD)
dGFP_i / dOD = P_i(OD)
```

for the seven class 2 promoters (`fliL`, `fliE`, `fliF`, `flgB`, `flgA`, `flhB`, `fliA`),
where `X(OD)` is the effective FlhDC activity and `Y(OD)` the effective FliA activity
recovered in Figure 4C of the paper.

## Install

```bash
git clone <this repo>
cd flagella-blueprint
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick start

```python
from process_bigraph import allocate_core
from flagella_blueprint import FlagellaProcess

core = allocate_core()
core.register_link('FlagellaProcess', FlagellaProcess)

proc = FlagellaProcess(config={}, core=core)
state = proc.initial_state()
for _ in range(200):
    state = proc.update({}, interval=5e-4)  # 5e-4 OD per step

print('Final OD:', state['OD'])
print('Final GFP per promoter:', state['GFP'])
```

## API

`FlagellaProcess` (single Process)

| Port | Type | Direction | Description |
| --- | --- | --- | --- |
| `OD` | `overwrite[float]` | output | current OD (time variable) |
| `X` | `overwrite[float]` | output | FlhDC effective activity |
| `Y` | `overwrite[float]` | output | FliA effective activity |
| `GFP` | `overwrite[list]` | output | cumulative GFP/OD per promoter (length 7) |
| `promoter_activity` | `overwrite[list]` | output | instantaneous P_i(OD) per promoter (length 7) |

| Config | Default | Description |
| --- | --- | --- |
| `beta` | `[1200, 450, 350, 350, 150, 100, 50]` | FlhDC activation coefficients (GFP/OD) |
| `beta_prime` | `[250, 350, 300, 450, 300, 350, 300]` | FliA activation coefficients (GFP/OD) |
| `X_max`, `X_midpoint`, `X_steepness` | `1.0`, `0.055`, `0.004` | FlhDC sigmoid (decreasing) |
| `Y_max`, `Y_midpoint`, `Y_steepness` | `1.0`, `0.060`, `0.005` | FliA sigmoid (increasing) |
| `substep` | `1e-4` | max dOD per integration substep |

The `interval` argument to `update()` is a Δ-OD, not Δ-time.

## Reproducing the paper

```bash
python demo/demo_report.py
```

Generates `demo/report.html` with:

* **Fig 4A/B** — model GFP/Gmax dynamics for all 7 promoters
* **Fig 4C** — recovered X(OD) and Y(OD) profiles
* **Fig 4D** — fliL* (FlhDC-only) and class 3 (FliA-only) controls
* **Fig 5A** — Nq vs Gmax with the analytic curve `Nq = q·Nf·Gmax / (Gmax − Ga)`
* **Fig 5B** — `β₁` reprogramming sweep for fliL
* **Fig 5C** — FlhDC induction collapsing the timing hierarchy

The report opens automatically in Safari.

## Tests

```bash
pytest -q
```

## Architecture notes

Everything lives in one `Process`. The X(OD) and Y(OD) profiles are smooth analytic
sigmoids parameterised to mimic the recovered profiles from Fig 4C; they are configurable,
which is how the reprogramming experiments of Fig 5B/5C are run (`beta`, `X_max`, etc.).
GFP integration uses an internal midpoint rule with a configurable substep so accuracy is
independent of the calling step size.
