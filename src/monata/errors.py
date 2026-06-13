class LibraryNotFoundError(KeyError):
    def __init__(self, name: str):
        super().__init__(f"Library not found: '{name}'")


class CellNotFoundError(KeyError):
    def __init__(self, cell_name: str, library_name: str):
        super().__init__(
            f"Cell '{cell_name}' not found in library '{library_name}'"
        )


class ViewNotFoundError(KeyError):
    def __init__(self, view_type: str, cell_name: str):
        super().__init__(
            f"View '{view_type}' not found in cell '{cell_name}'"
        )


class ViewNotGeneratedError(FileNotFoundError):
    def __init__(self, view_type: str, cell_name: str):
        super().__init__(
            f"View '{view_type}' in cell '{cell_name}' has not been generated yet. "
            f"Call cell.generate_{view_type}() first."
        )


class ViewAlreadyModifiedError(RuntimeError):
    def __init__(self, view_type: str, cell_name: str):
        super().__init__(
            f"View '{view_type}' in cell '{cell_name}' has been manually modified "
            f"(generated=false). Use force=True to overwrite."
        )
