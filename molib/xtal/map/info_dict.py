import gemmi

from decologr import Decologr as log


def get_mtz_info_dict(mtz_path: str) -> dict | None:
    """
    Get detailed information about an MTZ file without loading the full map.

    Args:
        mtz_path: Path to MTZ file

    Returns:
        Dictionary with MTZ file information or None if loading fails
    """
    try:
        log.info(f"Getting MTZ file info: {mtz_path}")

        # Load MTZ file using Gemmi
        mtz = gemmi.read_mtz_file(mtz_path)

        # Extract column information
        columns_info = []
        for col in mtz.columns:
            col_info = {
                "label": col.label,
                "type": col.type,
                "dataset": col.dataset,
            }

            # Safely get min/max values if available
            try:
                if hasattr(col, "min_value") and col.min_value is not None:
                    col_info["min_value"] = float(col.min_value)
                if hasattr(col, "max_value") and col.max_value is not None:
                    col_info["max_value"] = float(col.max_value)
            except (ValueError, TypeError):
                pass  # Skip if conversion fails

            columns_info.append(col_info)

        # Get crystallographic information
        unit_cell = mtz.cell
        space_group = mtz.spacegroup

        # Get dataset information
        datasets = []
        for dataset in mtz.datasets:
            dataset_info = {
                "name": dataset.dataset_name,
                "project_name": dataset.project_name,
                "crystal_name": dataset.crystal_name,
            }

            # Safely get wavelength if available
            try:
                if hasattr(dataset, "wavelength") and dataset.wavelength is not None:
                    dataset_info["wavelength"] = float(dataset.wavelength)
            except (ValueError, TypeError):
                pass

            datasets.append(dataset_info)

        # Get resolution information safely
        resolution_info = {}
        try:
            if hasattr(mtz, "resolution_high") and mtz.resolution_high is not None:
                resolution_info["d_min"] = float(mtz.resolution_high)
            if hasattr(mtz, "resolution_low") and mtz.resolution_low is not None:
                resolution_info["d_max"] = float(mtz.resolution_low)
        except (ValueError, TypeError):
            pass

        # Get reflection count safely
        reflection_count = 0
        try:
            if hasattr(mtz, "nreflections"):
                reflection_count = mtz.nreflections
            elif hasattr(mtz, "size"):
                reflection_count = mtz.size
        except (AttributeError, TypeError):
            pass

        mtz_info = {
            "file_path": mtz_path,
            "columns": columns_info,
            "unit_cell": {
                "a": unit_cell.a,
                "b": unit_cell.b,
                "c": unit_cell.c,
                "alpha": unit_cell.alpha,
                "beta": unit_cell.beta,
                "gamma": unit_cell.gamma,
            },
            "space_group": str(space_group),
            "datasets": datasets,
            "resolution": resolution_info,
            "reflection_count": reflection_count,
            "column_count": len(mtz.columns),
            "dataset_count": len(mtz.datasets),
        }

        log.info("✅ MTZ file info extracted successfully")
        log.info(f"📊 Columns: {len(columns_info)}")
        log.info(f"📊 Datasets: {len(datasets)}")
        log.info(f"📊 Reflections: {reflection_count}")

        return mtz_info

    except Exception as e:
        log.error(f"❌ Error getting MTZ file info from {mtz_path}: {e}")
        return None
