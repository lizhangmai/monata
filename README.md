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

## Environment Setup

Monata supports Python 3.11 and 3.12. The Python package does not bundle
simulator binaries, so set up the runtime environment before running simulation
examples.

Choose one of these paths. The agent-managed pixi path is recommended for new
projects. The existing-environment path is only for users who have already
built and installed the required circuit packages into the Python environment
they plan to use.

### Recommended: agent-managed pixi environment

This is the recommended path when you use Codex, Claude Code, or another coding
agent that can install skills. Install the `conda-build` skill from
`lizhangmai/skills` together with the user-facing `monata-sim-env` skill, then
ask the agent to create the Monata runtime environment for you. The agent
should build only the native packages required by the requested workflow.

In Claude Code, install the skill through the plugin marketplace:

```text
/plugin marketplace add https://github.com/lizhangmai/skills
/plugin install monata-sim-env@lizhangmai
/plugin install conda-build@lizhangmai
```

In Codex or another agent, install both skills with the open skills installer
or that agent's normal skill-install flow:

```bash
npx skills add lizhangmai/skills --skill monata-sim-env --skill conda-build
```

Then open the agent in your project workspace and ask:

```text
Use the monata-sim-env skill to set up a Monata simulation environment.

Use this final local conda channel for circuit-tool packages:
CONDA_BUILD_OUTPUT_DIR=<absolute-path-you-choose>

Build or reuse the circuit-toolchain ngspice package, create a pixi project
environment that uses that local channel plus conda-forge, install Python 3.12,
ngspice, and the PyPI monata package, then verify that Python can import monata
and find the ngspice executable.

Build only the packages needed for this Monata workflow. For the current Monata
backend, build or reuse ngspice only; do not build the full circuit-toolchain
set unless I explicitly request it.

Do not publish or upload packages to any remote channel.
```

Replace `<absolute-path-you-choose>` with a real absolute path before sending
the prompt. If the prompt does not include `CONDA_BUILD_OUTPUT_DIR=...`, the
agent should ask for it before running build, pixi, or install commands. Add
extra circuit packages only when your workflow needs them, for example
`openvaf-r` for Verilog-A to OSDI preparation. The Xyce recipe stack is not
required for the current Monata backend.

### Existing simulator environment

Use this path if your current Python environment already has the relevant
circuit packages installed, such as an `ngspice` executable on `PATH` or under
`CONDA_PREFIX/bin`. In that case, only the PyPI package needs to be installed.

```bash
python -m pip install monata
python -c "import monata, shutil; print(shutil.which('ngspice'))"
```

If you only use Monata for library organization, netlist generation, or result
post-processing, the simulator check is not required until you run a simulation
backend.

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

## External Tool Boundary

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
