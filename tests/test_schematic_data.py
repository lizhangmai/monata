import json

import pytest

from monata.netlist import SubCircuit, render_ngspice
from monata.schematic import (
    Instance,
    InstanceRef,
    Net,
    Pin,
    SchematicBuilder,
    SchematicData,
    dump_schematic,
    load_schematic,
    read_schematic,
    schematic_from_source_subcircuit,
    schematic_to_subcircuit,
    write_schematic,
)


def test_schematic_data_rejects_duplicate_pin_names():
    with pytest.raises(ValueError, match="duplicate pin"):
        SchematicData(
            cell="dup",
            pins=(Pin("a"), Pin("a")),
            nets=(Net("a"),),
        )


def test_schematic_data_rejects_unknown_instance_nets():
    with pytest.raises(ValueError, match="missing referenced net"):
        SchematicData(
            cell="bad",
            pins=(Pin("a"),),
            nets=(Net("a"),),
            instances=(
                Instance(
                    "x1",
                    ref=InstanceRef(kind="subckt", subckt="child"),
                    connections={"a": "missing"},
                ),
            ),
        )


def test_pin_rejects_empty_explicit_net():
    with pytest.raises(ValueError, match="pin a.net"):
        Pin("a", net="")


def test_schematic_data_rejects_duplicate_instance_names():
    with pytest.raises(ValueError, match="duplicate instance"):
        SchematicData(
            cell="dup",
            pins=(Pin("a"),),
            instances=(
                Instance("x1", ref=InstanceRef(kind="subckt", subckt="child"), connections={"a": "a"}),
                Instance("x1", ref=InstanceRef(kind="subckt", subckt="child"), connections={"a": "a"}),
            ),
        )


def test_instance_rejects_ambiguous_node_and_pin_connections():
    with pytest.raises(ValueError, match="cannot mix nodes and connections"):
        Instance(
            "x1",
            ref=InstanceRef(kind="subckt", subckt="child"),
            connections={"a": "a"},
            nodes=("a",),
        )


def test_json_loader_rejects_invalid_optional_text_fields():
    payload = {
        "schema_version": 2,
        "view_type": "schematic",
        "cell": {"name": "bad"},
        "interface": {"pins": [{"name": "a", "direction": None}]},
    }

    with pytest.raises(ValueError, match="direction"):
        load_schematic(payload)

    payload["interface"]["pins"] = [{"name": "a", "net": ""}]
    with pytest.raises(ValueError, match="net"):
        load_schematic(payload)

    payload["interface"]["pins"] = [{"name": "a"}]
    payload["nets"] = [{"name": "a", "kind": 7}]
    with pytest.raises(ValueError, match="kind"):
        load_schematic(payload)


def test_data_model_rejects_unprojectable_instance_kind():
    with pytest.raises(ValueError, match="unsupported instance ref kind"):
        InstanceRef(kind="primitive", model="symbol")


def test_json_roundtrip_preserves_order_and_optional_metadata():
    schematic = (
        SchematicBuilder("tg")
        .pin("a", direction="input")
        .pin("z", direction="output")
        .pin("vdd", direction="power")
        .pin("vss", direction="ground")
        .pdk_instance(
            "mn",
            lib="PTM",
            cell="nfet",
            view="ngspice",
            pins={"d": "z", "g": "a", "s": "vss", "b": "vss"},
            parameters={"w": "1u", "l": "45n"},
        )
        .property("purpose", "unit-test")
        .annotation(kind="note", text="kept")
        .build()
    )

    loaded = load_schematic(json.loads(dump_schematic(schematic)))

    assert loaded.pin_names == ("a", "z", "vdd", "vss")
    assert loaded.instances[0].parameters["w"] == "1u"
    assert loaded.properties["purpose"] == "unit-test"
    assert loaded.annotations[0]["text"] == "kept"
    assert dump_schematic(loaded) == dump_schematic(schematic)


def test_builder_projection_to_native_netlist_ir():
    schematic = (
        SchematicBuilder("inverter")
        .pin("vin", direction="input")
        .pin("vout", direction="output")
        .pin("vdd", direction="power")
        .pin("vss", direction="ground")
        .pdk_instance(
            "mn",
            lib="PTM",
            cell="nfet",
            view="ngspice",
            pins={"d": "vout", "g": "vin", "s": "vss", "b": "vss"},
        )
        .primitive(
            "mp",
            "pmos",
            connections={"d": "vout", "g": "vin", "s": "vdd", "b": "vdd"},
            model="pmos",
        )
        .build()
    )

    circuit = schematic_to_subcircuit(schematic)

    assert isinstance(circuit, SubCircuit)
    assert circuit.nodes == ("vin", "vout", "vdd", "vss")
    assert circuit.pdk_instances[0].cell == "nfet"
    assert "Mmp vout vin vdd vdd pmos" in render_ngspice(circuit)


def test_source_subcircuit_import_preserves_provenance(tmp_path):
    source = tmp_path / "and2.scs"
    source.write_text(".subckt and2 a b out\nI1 (a b out) nand2\n.ends and2\n")

    schematic = schematic_from_source_subcircuit(
        source,
        expected_name="and2",
        expected_ports=("a", "b", "out"),
        expected_count=1,
        allowed_kinds=("nand2",),
    )
    path = write_schematic(tmp_path / "schematic.monata.json", schematic)
    loaded = read_schematic(path)
    circuit = schematic_to_subcircuit(loaded)

    assert loaded.pin_names == ("a", "b", "out")
    assert loaded.provenance[0].kind == "source-subcircuit"
    assert loaded.provenance[0].source == str(source)
    assert loaded.provenance[0].sha256
    assert "XI1 a b out nand2" in render_ngspice(circuit)
