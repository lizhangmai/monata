# Monata

Monata is a lightweight Python framework for circuit workflow automation. It
organizes projects around libraries, cells, and views, then connects circuit
authoring to simulation tasks, measurements, sweeps, corners, Monte Carlo, and
optimization.

This repository contains the Monata Python package source. Long-form user
guides, toolchain notes, maintainer notes, release process, roadmap material,
and packaging drafts live in the documentation repository:
https://github.com/lizhangmai/monata-docs.

## Package Boundary

The public `monata` package contains the Python framework, public API
contracts, and backend adapters. API ownership and compatibility boundaries are
documented in the Monata documentation repository; release-specific status
belongs in release notes and package metadata rather than this README.

Monata adapts to simulator and model-toolchain programs installed by the user.
It does not ship `ngspice`, `libngspice`, XSPICE code models, OpenVAF, Xyce,
foundry PDKs, PTM model cards, or compiled OSDI binaries.

The default `ngspice-subprocess` backend runs a local `ngspice` executable from
`PATH` or `CONDA_PREFIX/bin`. The optional `ngspice-shared` backend loads a
user-provided `libngspice` shared library at runtime and the
`monata[ngspice-shared]` extra only installs the Python FFI dependency needed
for that runtime loading path. Users and downstream packagers remain
responsible for installing external tools and complying with their upstream
licenses.

## Install

From PyPI, after public release:

```bash
python -m pip install monata
```

For source development:

```bash
git clone https://github.com/lizhangmai/monata.git
cd monata
python -m pip install -e ".[dev]"
```

## Quick Start

```python
from monata import LibraryRegistry

reg = LibraryRegistry()
lib = reg.create_library(
    path="work/cmos_cells",
    name="cmos_cells",
    techlib_attachments=["PTM_BULK"],
    default_corner="ptm65",
)

cell = lib.create_cell("inverter", description="CMOS inverter")
```

The default native simulator backend uses a local `ngspice` executable.
Plotting support requires the optional `monata[plot]` extra.

Optional technology libraries can be installed as separate packages. These keep
PDK/model bundles outside `monata` core while preserving front-end PDK identity:

```python
from monata.netlist import Circuit
from monata.techlib.registry import TechlibRegistry
from monata.techlib.schema import TechlibAttachment

registry = TechlibRegistry()
circuit = Circuit("ptm device")
mn = circuit.pdk_instance(
    "mn",
    lib="PTM_MG",
    cell="nfet",
    view="ngspice",
    pins={"d": "out", "g": "in", "s": "0", "b": "0"},
)
projection = registry.validate_instance(
    mn,
    attachments=[TechlibAttachment("PTM_MG", default_corner="ptm20hp")],
).project()
projection.apply_to(circuit)
```

The optional `monata-techlib` package provides a first-party PTM techlib bundle
with `PTM_MG` and `PTM_BULK` metadata.

### Corner API

Process and simulation corners now use one public operating-point type:
`monata.corner.OperatingCorner`. Concrete simulation and techlib submodules do
not re-export a `Corner` alias; import the canonical type from `monata.corner`.
New result exports and experiment saves write the canonical payload with
`voltages`, `process`, `techlib`, `model_deck`, `section`, `model_file`,
`nominal_vdd`, `process_node`, `flavor`, `device_defaults`, and explicit
`metadata`. Historical aliases such as `voltage` and `node` are rejected on
load.

Consumers should resolve process policy from techlib metadata instead of local
corner tables:

```python
from monata.techlib.registry import TechlibRegistry

techlib = TechlibRegistry()["PTM_BULK"]
corner = techlib.corner("ptm65")
vdd = corner.nominal_vdd
device_defaults = corner.device_defaults
```

Generated Monata netlist artifacts may contain `.monata_model_ref` directives.
Those directives are Monata-internal logical model references for portable
artifacts; they are not raw ngspice directives. Runtime and export decks should
project with concrete references so ngspice receives normal `.lib` statements.

Digital truth-table testbenches are available through the simulation layer:

```python
from monata.sim.digital_table import DigitalTruthTable

truth_table = DigitalTruthTable(
    dut=Inverter,
    inputs=("vin",),
    outputs=("out",),
    expected=lambda bits: (int(not bits[0]),),
    setup=add_cmos_models,
    period=2e-9,
    step=5e-11,
)

result = truth_table.run("transient")
assert not result.failed
assert result.measurements_as_dict()["truth_table"]["status"] == "PASS"
```

Digital truth-table runs are measurement-oriented: `truth_table` and
`max_propagation_delay` are reported as run measurements, while the underlying
transient executions are stored as generic task artifacts when an artifact
directory is requested through `SimTask.artifacts` or session artifact options.
Backend-specific run controls such as ngspice rawfile format use
`SimTask.backend_options`; `metadata` is reserved for descriptive task context.

See the documentation repository for complete guides and simulator setup notes:
https://github.com/lizhangmai/monata-docs.

## Security

Monata project views can be Python source files. Loading schematic and testbench
views executes that project code in the current Python process; view loading is
not sandboxed. Open, load, generate, and simulate only trusted libraries and
project workspaces.

## Development

```bash
pytest -q
ruff check src/monata tests
pyright src/monata tests
python -m build
twine check dist/*
```

## Documentation

- Documentation repository: https://github.com/lizhangmai/monata-docs
- User guide: https://github.com/lizhangmai/monata-docs/tree/main/docs/user-guide
- API boundaries: https://github.com/lizhangmai/monata-docs/blob/main/docs/reference/api-boundaries.md
- Simulator and tool setup: https://github.com/lizhangmai/monata-docs/blob/main/docs/toolchain/external-tools.md
- Maintainer notes: https://github.com/lizhangmai/monata-docs/tree/main/docs/maintainers

## License

Monata's Python framework is licensed under the MIT License. See
[`LICENSE`](LICENSE).
