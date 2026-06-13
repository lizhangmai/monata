# Monata

Monata is a Python toolkit for circuit workflow automation. It provides a
typed front end for organizing circuit projects, authoring SPICE-oriented
netlists, running simulator tasks, and working with measurement and result
data.

Monata is designed as a lightweight Python package, not as a simulator
distribution. It integrates with simulator and model-toolchain programs that
users install in their own environments.

## Features

- Library, cell, and view organization for circuit projects.
- Native circuit/netlist records with ngspice-compatible rendering.
- Backend-neutral simulation tasks, result objects, sweeps, corners, Monte
  Carlo flows, and measurements.
- Built-in ngspice subprocess backend, plus a shared-library ngspice runner
  for environments that provide `libngspice`.
- Plotting and HDF5 result import/export available from the default install.
- Optional technology-library packages, such as `monata-techlib`, for reusable
  model metadata and redistributed model assets.

## Installation

```bash
python -m pip install monata
```

Monata supports Python 3.11 and 3.12.

## Quick Start

Create a Monata library and build a small circuit:

```python
from monata import LibraryRegistry
from monata.netlist import Circuit

registry = LibraryRegistry()
library = registry.create_library(path="work/analog", name="analog")
cell = library.create_cell("rc_filter", description="RC low-pass filter")

circuit = Circuit("rc low-pass")
circuit.voltage("in", "vin", "0", "1")
circuit.resistor("load", "vin", "vout", "1k")
circuit.capacitor("hold", "vout", "0", "1n")

print(cell.name)
print(circuit.to_spice())
```

Run a simulation when a supported backend is installed:

```python
from monata.sim.core import DCSpec, LocalExecutor, SimTask

task = SimTask(
    circuit=circuit,
    analysis_spec=DCSpec(source="Vin", start=0, stop=1, step=0.25),
    output_names=["vout"],
)

result = LocalExecutor(max_workers=1).submit(task).result()
if result.failed:
    raise RuntimeError(result.error_message)

print(result.waveforms["vout"])
```

The default backend is `ngspice-subprocess`, which runs a local `ngspice`
executable from `PATH` or `CONDA_PREFIX/bin`.

## External Tools

The public `monata` package does not ship simulator binaries, shared simulator
libraries, XSPICE code models, OpenVAF, Xyce, foundry PDKs, PTM model cards, or
compiled OSDI binaries.

Monata can use:

- a user-installed `ngspice` executable through `ngspice-subprocess`;
- a user-provided `libngspice` shared library through `ngspice-shared`;
- separately installed technology-library packages such as `monata-techlib`.

Users and downstream packagers remain responsible for installing external tools
and complying with their upstream licenses.

## Documentation

Long-form documentation lives outside this source package:

- Documentation repository: https://github.com/lizhangmai/monata-docs
- Getting started: https://github.com/lizhangmai/monata-docs/tree/main/docs/getting-started
- User guide: https://github.com/lizhangmai/monata-docs/tree/main/docs/user-guide
- API boundaries: https://github.com/lizhangmai/monata-docs/blob/main/docs/reference/api-boundaries.md
- Simulator and tool setup: https://github.com/lizhangmai/monata-docs/blob/main/docs/toolchain/external-tools.md
- Maintainer notes: https://github.com/lizhangmai/monata-docs/tree/main/docs/maintainers

## Security

Monata project views can be Python source files. Loading schematic and testbench
views executes that project code in the current Python process; view loading is
not sandboxed. Open, load, generate, and simulate only trusted libraries and
project workspaces.

## License

Monata's Python framework is licensed under the MIT License. See
[`LICENSE`](LICENSE).
