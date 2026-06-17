"""monata.models — Verilog-A model compilation, caching, and registry.

Public API:
    ModelCompiler  — compile .va sources to .osdi or .so
    ModelRegistry  — discover and resolve compiled model artifacts
    ModelCache     — content-hash-based compilation cache
"""

from monata.models.compiler import ModelCompiler
from monata.models.diagnostics import ModelDiagnostic, ModelDiagnosticError
from monata.models.flow import ModelFlowError, ModelFlowRecipe, ResolvedModelFlow, SimulationModelConfig
from monata.models.manifest import DeviceMetadata, ModelManifest, ModelSelection
from monata.models.resolver import ModelResolver, resolve_model_flow
from monata.models.registry import ModelRegistry
from monata.models.cache import ModelCache, resolve_model_cache_dir

__all__ = [
    "ModelCompiler",
    "DeviceMetadata",
    "ModelFlowRecipe",
    "ModelFlowError",
    "ModelDiagnostic",
    "ModelDiagnosticError",
    "ModelManifest",
    "ModelResolver",
    "ModelSelection",
    "ResolvedModelFlow",
    "SimulationModelConfig",
    "ModelRegistry",
    "ModelCache",
    "resolve_model_cache_dir",
    "resolve_model_flow",
]
