import tomllib

import pytest
from monata.models.manifest import DeviceMetadata, ModelManifest
from monata.models.registry import ModelEntry, ModelRegistry
from monata.models.diagnostics import ModelDiagnosticError
from monata.netlist import Circuit
from monata.sim.analysis_spec import TranSpec
from monata.sim.corner import CornerMatrix
from monata.workspace.experiment import Experiment
from monata.workspace.project import Project
from support.executors import CapturingExecutor
from support.workspaces import create_project


class TestProject:
    def test_create_new_project(self, tmp_path):
        proj = create_project(tmp_path, "my_project")
        assert proj.path.exists()
        assert (proj.path / "project.toml").exists()
        assert (proj.path / "experiments").is_dir()
        assert (proj.path / "libraries").is_dir()

    def test_project_constructor_does_not_create_missing_path(self, tmp_path):
        project_path = tmp_path / "typo_project"

        with pytest.raises(FileNotFoundError, match="Project.create"):
            Project(project_path)

        assert not project_path.exists()

    def test_project_create_rejects_existing_path_without_exist_ok(self, tmp_path):
        proj = create_project(tmp_path, "my_project")

        with pytest.raises(FileExistsError, match="already exists"):
            Project.create(proj.path)

    def test_open_existing_project(self, tmp_path):
        proj_path = tmp_path / "existing"
        proj_path.mkdir()
        (proj_path / "project.toml").write_text('[project]\nname = "existing"\n')
        proj = Project(proj_path)
        assert proj.name == "existing"

    def test_project_config_rejects_unknown_root_fields(self, tmp_path):
        proj_path = tmp_path / "project"
        proj_path.mkdir()
        (proj_path / "project.toml").write_text(
            'unexpected = true\n\n[project]\nname = "project"\n'
        )

        with pytest.raises(ValueError, match="project.toml has unknown fields: unexpected"):
            Project(proj_path)

    def test_project_config_rejects_unknown_project_fields(self, tmp_path):
        proj_path = tmp_path / "project"
        proj_path.mkdir()
        (proj_path / "project.toml").write_text(
            '[project]\nname = "project"\nunexpected = true\n'
        )

        with pytest.raises(ValueError, match="project table has unknown fields: unexpected"):
            Project(proj_path)

    def test_project_config_rejects_unknown_library_entry_fields(self, tmp_path):
        proj_path = tmp_path / "project"
        proj_path.mkdir()
        (proj_path / "project.toml").write_text(
            '[project]\nname = "project"\n\n'
            '[[libraries]]\nname = "analog"\npath = "libraries/analog"\n'
            'unexpected = true\n'
        )

        with pytest.raises(
            ValueError,
            match=r"project libraries\[0\] has unknown fields: unexpected",
        ):
            Project(proj_path)

    def test_new_experiment(self, tmp_path):
        proj = create_project(tmp_path)
        exp = proj.new_experiment("folded_cascode_v1", description="First attempt")
        assert exp.name == "folded_cascode_v1"
        assert (proj.path / "experiments" / "folded_cascode_v1").exists()

    def test_experiment_config_rejects_unknown_root_fields(self, tmp_path):
        proj = create_project(tmp_path)
        exp = proj.new_experiment("exp1")
        (exp.path / "experiment.toml").write_text(
            'unexpected = true\n\n[experiment]\nname = "exp1"\n'
        )

        with pytest.raises(ValueError, match="experiment.toml has unknown fields: unexpected"):
            Experiment(exp.path)

    def test_experiment_config_rejects_unknown_experiment_fields(self, tmp_path):
        proj = create_project(tmp_path)
        exp = proj.new_experiment("exp1")
        (exp.path / "experiment.toml").write_text(
            '[experiment]\nname = "exp1"\nunexpected = true\n'
        )

        with pytest.raises(ValueError, match="experiment table has unknown fields: unexpected"):
            Experiment(exp.path)

    def test_experiment_config_requires_experiment_table(self, tmp_path):
        proj = create_project(tmp_path)
        exp = proj.new_experiment("exp1")
        (exp.path / "experiment.toml").write_text("")

        with pytest.raises(ValueError, match=r"experiment.toml is missing \[experiment\]"):
            Experiment(exp.path)

    @pytest.mark.parametrize("name", ["../outside", "quote\"name", "evil\nname", "tab\tname", "space name"])
    def test_new_experiment_rejects_unsafe_names(self, tmp_path, name):
        proj = create_project(tmp_path)

        with pytest.raises(ValueError, match="single safe path segment"):
            proj.new_experiment(name)

    def test_list_experiments_empty(self, tmp_path):
        proj = create_project(tmp_path)
        assert proj.list_experiments() == []

    def test_list_experiments(self, tmp_path):
        proj = create_project(tmp_path)
        proj.new_experiment("exp1")
        proj.new_experiment("exp2")
        (proj.path / "experiments" / "scratch.txt").write_text("not an experiment")
        exps = proj.list_experiments()
        assert sorted(exps) == ["exp1", "exp2"]

    def test_list_experiments_ignores_directories_without_experiment_metadata(self, tmp_path):
        proj = create_project(tmp_path)
        proj.new_experiment("exp1")
        scratch = proj.path / "experiments" / "scratch"
        scratch.mkdir()

        assert proj.list_experiments() == ["exp1"]

    def test_duplicate_experiment_raises(self, tmp_path):
        proj = create_project(tmp_path)
        proj.new_experiment("exp1")
        with pytest.raises(FileExistsError):
            proj.new_experiment("exp1")

    def test_compare_placeholder(self, tmp_path):
        proj = create_project(tmp_path)
        proj.new_experiment("a")
        proj.new_experiment("b")
        df = proj.compare("a", "b")
        assert df is not None

    def test_compare_without_names_uses_all_experiments_in_stable_order(self, tmp_path):
        proj = create_project(tmp_path)
        proj.new_experiment("z_last")
        proj.new_experiment("a_first")

        rows = proj.compare()

        assert rows is not None
        assert [row["experiment"] for row in rows] == ["a_first", "z_last"]

    def test_compare_without_names_ignores_scratch_directories_without_side_effects(self, tmp_path):
        proj = create_project(tmp_path)
        proj.new_experiment("exp1")
        scratch = proj.path / "experiments" / "scratch"
        scratch.mkdir()

        rows = proj.compare()

        assert rows is not None
        assert [row["experiment"] for row in rows] == ["exp1"]
        assert not (scratch / "results").exists()

    def test_compare_ignores_missing_experiments_without_creating_paths(self, tmp_path):
        proj = create_project(tmp_path)

        assert proj.compare("missing") is None
        assert not (proj.path / "experiments" / "missing").exists()

    def test_compare_rejects_unsafe_experiment_names(self, tmp_path):
        proj = create_project(tmp_path)

        with pytest.raises(ValueError, match="single safe path segment"):
            proj.compare("../outside")

    def test_create_project_library_round_trip(self, tmp_path):
        proj = create_project(tmp_path)

        lib = proj.create_library("analog", tech_model_paths=["models/nmos.lib"])

        assert lib.name == "analog"
        assert lib.path == proj.path / "libraries" / "analog"
        assert proj.list_libraries() == ["analog"]
        assert proj.get_library("analog").tech_model_paths == ["models/nmos.lib"]

        reopened = Project(proj.path)
        assert reopened.list_libraries() == ["analog"]
        assert reopened.get_library("analog").path == lib.path
        assert "analog" in reopened.library_registry()

    def test_create_project_library_records_description(self, tmp_path):
        proj = create_project(tmp_path)

        proj.create_library("analog", description="frontend devices")

        with open(proj.path / "libraries" / "analog" / "lib.toml", "rb") as file:
            config = tomllib.load(file)
        assert config["library"]["description"] == "frontend devices"

    def test_create_project_library_records_techlib_attachments(self, tmp_path):
        proj = create_project(tmp_path)

        lib = proj.create_library(
            "analog",
            techlib_attachments=["PTM_BULK"],
            default_corner="ptm65",
        )

        assert lib.attached_techlibs == ["PTM_BULK"]
        assert proj.get_library("analog").techlib_attachments[0].default_corner == "ptm65"

    def test_add_existing_library_round_trip(self, tmp_path):
        lib_dir = tmp_path / "external" / "rf"
        lib_dir.mkdir(parents=True)
        (lib_dir / "lib.toml").write_text(
            '[library]\nname = "rf"\ndescription = ""\n\n'
            '[technology]\nmodel_paths = ["rf.mod"]\n'
        )
        proj = create_project(tmp_path)

        lib = proj.add_library(lib_dir)

        assert lib.name == "rf"
        assert proj.list_libraries() == ["rf"]
        assert Project(proj.path).get_library("rf").path == lib_dir

    def test_create_library_rejects_path_traversal_name(self, tmp_path):
        proj = create_project(tmp_path)

        with pytest.raises(ValueError, match="single safe path segment"):
            proj.create_library("../outside")

    @pytest.mark.parametrize("name", ["quote\"name", "evil\nname", "tab\tname", "space name"])
    def test_create_library_rejects_toml_breaking_names(self, tmp_path, name):
        proj = create_project(tmp_path)

        with pytest.raises(ValueError, match="single safe path segment"):
            proj.create_library(name)

    def test_add_relative_external_library_round_trip(self, tmp_path, monkeypatch):
        caller = tmp_path / "caller"
        lib_dir = caller / "external" / "rf"
        lib_dir.mkdir(parents=True)
        (lib_dir / "lib.toml").write_text(
            '[library]\nname = "rf"\ndescription = ""\n\n'
            '[technology]\nmodel_paths = ["rf.mod"]\n'
        )
        proj = create_project(tmp_path)
        monkeypatch.chdir(caller)

        proj.add_library("external/rf")

        assert Project(proj.path).get_library("rf").path == lib_dir

    def test_project_rewrite_preserves_multiple_library_entries(self, tmp_path):
        proj = create_project(tmp_path)

        proj.create_library("analog", tech_model_paths=["models/analog.lib"])
        proj.create_library("rf", tech_model_paths=["models/rf.lib"])
        reopened = Project(proj.path)

        assert reopened.list_libraries() == ["analog", "rf"]
        assert reopened.get_library("analog").tech_model_paths == ["models/analog.lib"]
        assert reopened.get_library("rf").tech_model_paths == ["models/rf.lib"]

    def test_project_model_manifest_round_trip_uses_sidecar(self, tmp_path):
        proj = create_project(tmp_path)
        osdi = proj.path / "models" / "bsim4.osdi"
        source = proj.path / "va" / "bsim4.va"
        include = proj.path / "va" / "constants.va"
        osdi.parent.mkdir()
        source.parent.mkdir()
        osdi.write_text("compiled")
        source.write_text("module bsim4; endmodule\n")
        include.write_text("parameter real tox = 1e-9;\n")
        manifest = ModelManifest([
            ModelEntry(
                name="nmos",
                family="mos",
                module_name="bsim4",
                osdi_path=osdi,
                source_va=source,
                include_paths=[include],
                provenance={"source": "synthetic"},
            )
        ])

        proj.save_model_manifest(manifest)
        text = proj.model_manifest_path().read_text()
        reopened = Project(proj.path).model_manifest()

        assert 'osdi_path = "models/bsim4.osdi"' in text
        assert 'source_va = "va/bsim4.va"' in text
        assert reopened.entries[0].osdi_path == str(osdi)
        assert reopened.entries[0].source_va == str(source)
        assert reopened.entries[0].include_paths == [str(include)]
        assert reopened.entries[0].provenance == {"source": "synthetic"}
        assert reopened.entries[0].level is None
        assert reopened.entries[0].version is None
        assert "[[libraries]]" not in text
        assert 'level = ""' not in text
        assert 'version = ""' not in text
        registry = ModelRegistry(auto_discover=False)
        registry.load_entries(reopened.entries)
        resolved = registry.resolve("mos")
        assert resolved is not None
        assert resolved.osdi_path == str(osdi)

    def test_project_model_manifest_relative_paths_survive_project_move(self, tmp_path):
        proj = create_project(tmp_path)
        osdi = proj.path / "models" / "bsim4.osdi"
        osdi.parent.mkdir()
        osdi.write_text("compiled")
        proj.save_model_manifest(ModelManifest([
            ModelEntry(name="nmos", family="mos", module_name="bsim4", osdi_path=osdi)
        ]))
        moved = tmp_path / "moved"
        moved.mkdir()
        new_project = moved / "proj"
        proj.path.rename(new_project)

        manifest = Project(new_project).model_manifest()

        assert manifest.entries[0].osdi_path == str(new_project / "models" / "bsim4.osdi")

    def test_project_model_manifest_invalid_shape_raises_diagnostic(self, tmp_path):
        proj = create_project(tmp_path)
        proj.model_manifest_path().write_text("models = \"not an array\"\n")

        with pytest.raises(ModelDiagnosticError) as excinfo:
            proj.model_manifest()

        assert excinfo.value.diagnostic.code == "model_manifest_invalid"

    @pytest.mark.parametrize(
        ("body", "missing"),
        [
            ('family = "mos"\nmodule_name = "bsim4"\n', "name"),
            ('name = "nmos"\nmodule_name = "bsim4"\n', "family"),
        ],
    )
    def test_project_model_manifest_invalid_model_required_fields_raise_diagnostic(
        self,
        tmp_path,
        body,
        missing,
    ):
        proj = create_project(tmp_path)
        proj.model_manifest_path().write_text(f"[[models]]\n{body}")

        with pytest.raises(ModelDiagnosticError) as excinfo:
            proj.model_manifest()

        assert excinfo.value.diagnostic.code == "model_manifest_invalid"
        assert f"models[0] is missing {missing}" == excinfo.value.diagnostic.message

    def test_project_model_manifest_device_metadata_round_trip(self, tmp_path):
        proj = create_project(tmp_path)
        manifest = ModelManifest(
            devices=[
                DeviceMetadata(
                    name="nmos",
                    family="mos",
                    module_name="bsim4",
                    model_name="nch",
                    parameters={"w": "width", "l": "length"},
                    documentation="Synthetic NMOS metadata",
                    provenance={"source": "unit-test"},
                ),
                DeviceMetadata(name="diode", family="d", model_name="dmod"),
            ]
        )

        proj.save_model_manifest(manifest)
        reopened = Project(proj.path).model_manifest()
        text = proj.model_manifest_path().read_text()

        nmos = reopened.device("nmos")
        assert [device.name for device in reopened.list_devices("mos")] == ["nmos"]
        assert nmos.module_name == "bsim4"
        assert nmos.model_name == "nch"
        assert nmos.parameters == {"w": "width", "l": "length"}
        assert nmos.documentation == "Synthetic NMOS metadata"
        assert nmos.provenance == {"source": "unit-test"}
        with pytest.raises(ModelDiagnosticError) as excinfo:
            reopened.device("missing")
        assert excinfo.value.diagnostic.code == "device_metadata_missing"
        assert 'module_name = ""' not in text

    def test_project_model_manifest_invalid_device_shape_raises_diagnostic(self, tmp_path):
        proj = create_project(tmp_path)
        proj.model_manifest_path().write_text("[[devices]]\nfamily = \"mos\"\n")

        with pytest.raises(ModelDiagnosticError) as excinfo:
            proj.model_manifest()

        assert excinfo.value.diagnostic.code == "model_manifest_invalid"

    def test_project_model_manifest_end_to_end_projection_to_corner_task(self, tmp_path):
        proj = create_project(tmp_path)
        model_file = proj.path / "models" / "models.lib"
        osdi = proj.path / "models" / "bsim4.osdi"
        model_file.parent.mkdir()
        model_file.write_text(".lib tt\n.endl\n")
        osdi.write_text("compiled")
        proj.save_model_manifest(ModelManifest([
            ModelEntry(
                name="nmos_tt",
                family="mos",
                module_name="bsim4",
                osdi_path=osdi,
                model_file=model_file,
                lib_section="tt",
            )
        ]))
        manifest = Project(proj.path).model_manifest()
        circuit = Circuit("manifest projection")
        circuit.voltage("DD", "vdd", "0", "1")
        circuit.resistor("1", "vdd", "0", "1k")

        matrix = CornerMatrix(
            circuit=circuit,
            analysis_spec=TranSpec(stop=1e-6),
            output_names=["vdd"],
            model_manifest=manifest,
        )
        matrix.add_temperatures(27)
        matrix.add_model_corners(tt="nmos_tt")
        executor = CapturingExecutor()
        matrix.run(executor)
        task = executor.tasks[0]

        assert circuit.directives == []
        assert task.circuit.directives[0].name == "lib"
        assert task.circuit.directives[0].args == (str(model_file), "tt")
        assert [str(path) for path in task.osdi_paths] == [str(osdi)]
        assert task.corner.model_file == str(model_file)
        assert task.metadata["model_selection"]["models"][0]["name"] == "nmos_tt"

    def test_project_model_manifest_missing_artifact_raises_diagnostic(self, tmp_path):
        proj = create_project(tmp_path)
        missing = proj.path / "models" / "missing.lib"
        proj.save_model_manifest(ModelManifest([
            ModelEntry(
                name="nmos_tt",
                family="mos",
                module_name="bsim4",
                model_file=missing,
                lib_section="tt",
            )
        ]))

        manifest = Project(proj.path).model_manifest()
        with pytest.raises(ModelDiagnosticError) as excinfo:
            manifest.resolve(name="nmos_tt")

        assert excinfo.value.diagnostic.code == "model_artifact_missing"
        assert excinfo.value.diagnostic.context["missing"] == [
            {
                "model": "nmos_tt",
                "field": "model_file",
                "path": str(missing),
            }
        ]
