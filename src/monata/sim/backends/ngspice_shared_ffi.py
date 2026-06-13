"""FFI loading helpers for the ngspice shared-library backend."""

from __future__ import annotations

import ctypes.util
from importlib import import_module
from typing import Any


class NgspiceSharedError(RuntimeError):
    """Base class for ngspice shared-library failures."""


class NgspiceSharedLibraryError(NgspiceSharedError):
    """Raised when libngspice or cffi cannot be loaded."""


_CDEF = """
typedef struct ngcomplex
{
  double cx_real;
  double cx_imag;
} ngcomplex_t;

typedef struct vector_info
{
  char *v_name;
  int v_type;
  short v_flags;
  double *v_realdata;
  ngcomplex_t *v_compdata;
  int v_length;
} vector_info, *pvector_info;

typedef struct vecvalues
{
  char *name;
  double creal;
  double cimag;
  _Bool is_scale;
  _Bool is_complex;
} vecvalues, *pvecvalues;

typedef struct vecvaluesall
{
  int veccount;
  int vecindex;
  pvecvalues *vecsa;
} vecvaluesall, *pvecvaluesall;

typedef struct vecinfo
{
  int number;
  char *vecname;
  _Bool is_real;
  void *pdvec;
  void *pdvecscale;
} vecinfo, *pvecinfo;

typedef struct vecinfoall
{
  char *name;
  char *title;
  char *date;
  char *type;
  int veccount;
  pvecinfo *vecs;
} vecinfoall, *pvecinfoall;

typedef int (SendChar) (char *, int, void *);
typedef int (SendStat) (char *, int, void *);
typedef int (ControlledExit) (int, _Bool, _Bool, int, void *);
typedef int (SendData) (pvecvaluesall, int, int, void *);
typedef int (SendInitData) (pvecinfoall, int, void *);
typedef int (BGThreadRunning) (_Bool, int, void *);
typedef int (GetVSRCData) (double *, double, char *, int, void *);
typedef int (GetISRCData) (double *, double, char *, int, void *);
typedef int (GetSyncData) (double, double *, double, int, int, int, void *);

int ngSpice_Init (SendChar *, SendStat *, ControlledExit *, SendData *, SendInitData *, BGThreadRunning *, void *);
int ngSpice_Init_Sync (GetVSRCData *, GetISRCData *, GetSyncData *, int *, void *);
int ngSpice_Command (char *);
pvector_info ngGet_Vec_Info (char *);
int ngSpice_Circ (char **);
char *ngSpice_CurPlot (void);
char **ngSpice_AllPlots (void);
char **ngSpice_AllVecs (char *);
_Bool ngSpice_running (void);
_Bool ngSpice_SetBkpt (double);
"""


def new_ffi() -> Any:
    try:
        cffi: Any = import_module("cffi")
    except ImportError as exc:
        raise NgspiceSharedLibraryError("ngspice shared backend requires the optional 'ngspice-shared' extra") from exc
    ffi = cffi.FFI()
    ffi.cdef(_CDEF)
    return ffi


def load_library(ffi: Any, library: str | None) -> Any:
    candidates = library_candidates(library)
    errors: list[str] = []
    for candidate in candidates:
        try:
            return ffi.dlopen(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors)
    raise NgspiceSharedLibraryError(f"libngspice shared library not found ({detail})")


def library_candidates(library: str | None) -> tuple[str, ...]:
    if library:
        return (library,)
    candidates: list[str] = []
    found = ctypes.util.find_library("ngspice")
    if found:
        candidates.append(found)
    candidates.extend(["ngspice", "libngspice.so", "libngspice.dylib", "ngspice.dll"])
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)
