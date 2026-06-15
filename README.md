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
- Local technology-library resource directories for reusable model metadata and
  model-card assets.

## Environment Setup

Monata supports Python 3.11 and 3.12. The Python package does not bundle
simulator binaries, so set up the runtime environment before running simulation
examples.

Choose one of these environment paths. The agent-managed pixi path is
recommended for new projects. The existing-environment path is only for users
who have already built and installed the required circuit packages into the
Python environment they plan to use.

### Recommended: agent-managed pixi environment

This is the recommended environment path when you use Codex, Claude Code, or
another coding agent that can install skills or plugins. Install
`monata-sim-env` from
[`lizhangmai/skills`](https://github.com/lizhangmai/skills), then ask the agent
to create the complete Monata runtime environment for you. The skill builds or
reuses required circuit-tool packages, configures pixi, installs Monata,
bootstraps PTM techlibs, generates `monata_readme_demo.py`, and runs it. These
install methods are equivalent; use whichever one fits your agent.

Open Skills CLI, for generic skill-aware agents:

```bash
npx skills@latest add lizhangmai/skills --skill monata-sim-env
```

Codex plugin marketplace:

```bash
codex plugin marketplace add https://github.com/lizhangmai/skills --ref main
codex plugin list --marketplace lizhangmai --available --json
codex plugin add monata-sim-env@lizhangmai
```

Claude Code plugin marketplace:

```text
/plugin marketplace add https://github.com/lizhangmai/skills
/plugin marketplace update lizhangmai
/plugin install monata-sim-env@lizhangmai
/reload-plugins
```

Then start a fresh agent session in your project workspace and ask:

```text
Use the monata-sim-env skill to set up this complete Monata environment,
bootstrap PTM techlibs, generate monata_readme_demo.py, and run it.
CONDA_BUILD_OUTPUT_DIR=<absolute-path-you-choose>
MONATA_HOME=<optional-absolute-monata-home>
```

Replace `<absolute-path-you-choose>` with a real absolute path before sending
the prompt. If the prompt does not include `CONDA_BUILD_OUTPUT_DIR=...`, the
agent should ask for it before running build, pixi, or install commands. The
skill inspects the Monata workspace before choosing tool packages; the current
Monata baseline is `ngspice` plus `openvaf-r`. The Xyce recipe stack is not
required for the current Monata backend. Omit `MONATA_HOME` to use
`~/.monata`; techlibs are installed under `$MONATA_HOME/techlibs`.

### Existing simulator environment

Use this path if your current Python environment already has the relevant
circuit packages installed, such as an `ngspice` executable on `PATH` or under
`CONDA_PREFIX/bin` and `openvaf-r` for model compilation. In that case, only
the PyPI package needs to be installed.

```bash
python -m pip install monata
python -c "import shutil; print(shutil.which('ngspice')); print(shutil.which('openvaf-r'))"
```

If you only use Monata for library organization, netlist generation, or result
post-processing, the simulator check is not required until you run a simulation
backend.

## Quick Start

Create a Monata library and build a small circuit:

```python
from monata import LibraryRegistry
from monata.netlist import Circuit
from pathlib import Path
import shutil

demo_root = Path("monata_readme_demo_work")
if demo_root.exists():
    shutil.rmtree(demo_root)
demo_root.mkdir(parents=True)

registry = LibraryRegistry()
library = registry.create_library(path=demo_root / "analog", name="analog")
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
if result.status != "ok":
    raise RuntimeError(result.error_message or "simulation failed")

print(result.waveforms["vout"])
```

The default backend is `ngspice-subprocess`, which runs a local `ngspice`
executable from `PATH` or `CONDA_PREFIX/bin`.

For reusable library cells, author the canonical schematic view as structured
data. The Python builder is an authoring helper; the persisted cellview is
`schematic.monata.json`.

```python
from monata.schematic import SchematicBuilder

(
    SchematicBuilder("rc_filter")
    .pin("vin", direction="input")
    .pin("vout", direction="output")
    .pin("0")
    .primitive("load", "resistor", connections={"n1": "vin", "n2": "vout"}, value="1k")
    .primitive("hold", "capacitor", connections={"n1": "vout", "n2": "0"}, value="1n")
    .write(cell.path / "schematic.monata.json")
)

cell.create_view("schematic")
cell.generate_symbol()
cell.generate_netlist()
```

## External Tool Boundary

The public `monata` package does not ship simulator binaries, shared simulator
libraries, XSPICE code models, OpenVAF, Xyce, foundry PDKs, PTM model cards, or
compiled OSDI binaries.

Monata can use:

- a user-installed `ngspice` executable through `ngspice-subprocess`;
- a user-installed `openvaf-r` executable to compile Verilog-A models into
  OSDI artifacts used by ngspice workflows;
- a user-provided `libngspice` shared library through `ngspice-shared`;
- user-managed technology-library resources loaded from explicit paths,
  `MONATA_TECHLIB_PATH`, or the user Monata techlib directory.

Users and downstream packagers remain responsible for installing external tools
and complying with their upstream licenses.

## Technology Libraries

Monata can load reusable technology libraries from local resource directories
that contain `techlib.toml` and referenced model files. This keeps PDK-like
resources outside the Python package while preserving a stable loader API.

```python
from monata.techlib.registry import TechlibRegistry

registry = TechlibRegistry(
    search_paths=["./techlibs"],
    auto_discover=False,
)
print(registry.list_techlibs())
```

For user-level discovery, set `MONATA_TECHLIB_PATH` to one or more techlib
collection directories, separated by the platform path separator. Monata also
checks `$MONATA_HOME/techlibs`. When `MONATA_HOME` is not set, Monata treats it
as `~/.monata`.

For a managed PTM setup, use the `monata-sim-env` skill from
[`lizhangmai/skills`](https://github.com/lizhangmai/skills). It downloads
official PTM resources on the user's machine, generates `PTM_MG` and
`PTM_BULK` techlibs, preserves upstream notices, verifies Monata discovery, and
runs `monata_readme_demo.py`.

## Documentation

Long-form documentation lives outside this source package:

- Documentation repository: https://github.com/lizhangmai/monata-docs
- Getting started: https://github.com/lizhangmai/monata-docs/tree/main/docs/getting-started
- User guide: https://github.com/lizhangmai/monata-docs/tree/main/docs/user-guide
- API boundaries: https://github.com/lizhangmai/monata-docs/blob/main/docs/reference/api-boundaries.md
- Simulator and tool setup: https://github.com/lizhangmai/monata-docs/blob/main/docs/toolchain/external-tools.md
- Maintainer notes: https://github.com/lizhangmai/monata-docs/tree/main/docs/maintainers

## Security

Monata 0.2 treats ordinary cellviews as declarative data by default:
`schematic.monata.json`, `symbol.monata.json`, and `testbench.monata.json` are
parsed and validated without executing project code. For data views, `read()`
returns the structured payload and `load()` remains a safe parse operation.

Canonical schematics are structured data, and default symbol generation,
netlist generation, JSON testbench DUT resolution, and digital truth-table DUT
resolution do not execute a neighboring `schematic.py`. Python remains useful
as an authoring surface that writes data, or as an explicit trusted extension
for executable views such as Python testbenches. Python view metadata without
an explicit format and `trusted = true` is rejected; `run_trusted()` and
`load_trusted()` execute project code in the current Python process and are not
sandboxed. Open, load, generate, and simulate executable views only from
trusted libraries and project workspaces.

## License

Monata's Python framework is licensed under the MIT License. See
[`LICENSE`](LICENSE).
