"""
Unit cell utilities for crystallographic structures.

This module provides functions to extract and process unit cell information
from PDB, MTZ, and CCP4 files.
"""

from typing import Any, Dict, Optional

from decologr import Decologr as log


def extract_unit_cell_from_pdb(pdb_data) -> Optional[Dict[str, Any]]:
    """
    Extract unit cell information from a PDB file.

    Args:
        pdb_data: PandasPdb object or similar containing PDB data

    Returns:
        Dictionary containing unit cell parameters or None if not found
    """
    try:
        # Check if we have CRYST1 information in the OTHERS section
        if hasattr(pdb_data, "df") and "OTHERS" in pdb_data.df:
            cryst1_records = pdb_data.df["OTHERS"][
                pdb_data.df["OTHERS"]["record_name"] == "CRYST1"
            ]

            if not cryst1_records.empty:
                cryst1_line = cryst1_records.iloc[0]["entry"]
                return _parse_cryst1_line(cryst1_line)

        # Alternative: check if there's a CRYST1 attribute directly
        if hasattr(pdb_data, "CRYST1"):
            return _parse_cryst1_line(pdb_data.CRYST1)

        log.warning("No CRYST1 record found in PDB file", scope="validate_unit_cell", silent=True)
        return None

    except Exception as e:
        log.error(f"Error extracting unit cell from PDB: {e}", scope="validate_unit_cell", silent=True)
        return None


def _parse_cryst1_line(cryst1_line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a CRYST1 line from a PDB file.

    CRYST1 format: CRYST1 a b c alpha beta gamma space_group z
    Example: CRYST1   63.100   50.170  111.070  90.00  96.19  90.00 P 1 21 1      4

    Args:
        cryst1_line: The CRYST1 line from the PDB file

    Returns:
        Dictionary containing unit cell parameters
    """
    try:
        # Split the line and extract numeric values
        parts = cryst1_line.split()

        if len(parts) < 6:
            log.warning(f"Invalid CRYST1 line format: {cryst1_line}", scope="validate_unit_cell", silent=True)
            return None

        # Extract unit cell parameters
        a = float(parts[0])
        b = float(parts[1])
        c = float(parts[2])
        alpha = float(parts[3])
        beta = float(parts[4])
        gamma = float(parts[5])

        # Extract space group if available
        space_group = parts[6] if len(parts) > 6 else "Unknown"

        unit_cell_info = {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "space_group": space_group,
            "source": "PDB_CRYST1",
        }

        log.info(f"✅ Extracted unit cell from PDB: a={a:.2f}, b={b:.2f}, c={c:.2f} Å", scope="validate_unit_cell", silent=True)
        return unit_cell_info

    except (ValueError, IndexError) as e:
        log.error(f"Error parsing CRYST1 line '{cryst1_line}': {e}", scope="validate_unit_cell", silent=True)
        return None


def extract_unit_cell_from_mtz(mtz_data) -> Optional[Dict[str, Any]]:
    """
    Extract unit cell information from MTZ data.

    Args:
        mtz_data: MTZ data object (from gemmi or similar)

    Returns:
        Dictionary containing unit cell parameters or None if not found
    """
    try:
        if hasattr(mtz_data, "unit_cell"):
            unit_cell = mtz_data.unit_cell

            unit_cell_info = {
                "a": unit_cell.a,
                "b": unit_cell.b,
                "c": unit_cell.segment_color,
                "alpha": unit_cell.alpha,
                "beta": unit_cell.beta,
                "gamma": unit_cell.gamma,
                "space_group": (
                    str(unit_cell.spacegroup)
                    if hasattr(unit_cell, "spacegroup")
                    else "Unknown"
                ),
                "source": "MTZ",
            }

            log.info(
                f"✅ Extracted unit cell from MTZ: a={unit_cell.a:.2f}, "
                f"b={unit_cell.b:.2f}, c={unit_cell.segment_color:.2f} Å", scope="validate_unit_cell", silent=True
            )
            return unit_cell_info

        log.warning("No unit cell information found in MTZ data", scope="validate_unit_cell", silent=True)
        return None

    except Exception as e:
        log.error(f"Error extracting unit cell from MTZ: {e}", scope="validate_unit_cell", silent=True)
        return None


def extract_unit_cell_from_ccp4(ccp4_data) -> Optional[Dict[str, Any]]:
    """
    Extract unit cell information from CCP4 data.

    Args:
        ccp4_data: CCP4 data object (from gemmi or similar)

    Returns:
        Dictionary containing unit cell parameters or None if not found
    """
    try:
        if hasattr(ccp4_data, "grid") and hasattr(ccp4_data.grid, "unit_cell"):
            unit_cell = ccp4_data.grid.unit_cell

            unit_cell_info = {
                "a": unit_cell.a,
                "b": unit_cell.b,
                "c": unit_cell.segment_color,
                "alpha": unit_cell.alpha,
                "beta": unit_cell.beta,
                "gamma": unit_cell.gamma,
                "space_group": (
                    str(ccp4_data.grid.spacegroup)
                    if hasattr(ccp4_data.grid, "spacegroup")
                    else "Unknown"
                ),
                "source": "CCP4",
            }

            log.info(
                f"✅ Extracted unit cell from CCP4: a={unit_cell.a:.2f}, ", scope="validate_unit_cell", silent=True
                f"b={unit_cell.b:.2f}, c={unit_cell.segment_color:.2f} Å", scope="validate_unit_cell", silent=True
            )
            return unit_cell_info

        log.warning("No unit cell information found in CCP4 data", scope="validate_unit_cell", silent=True)
        return None

    except Exception as e:
        log.error(f"Error extracting unit cell from CCP4: {e}", scope="validate_unit_cell", silent=True)
        return None


def validate_unit_cell(unit_cell_info: Dict[str, Any]) -> bool:
    """
    Validate unit cell parameters.

    Args:
        unit_cell_info: Dictionary containing unit cell parameters

    Returns:
        True if valid, False otherwise
    """
    if not unit_cell_info:
        return False

    required_keys = ["a", "b", "c", "alpha", "beta", "gamma"]

    # Check if all required keys are present
    if not all(key in unit_cell_info for key in required_keys):
        log.warning("Missing required unit cell parameters", scope="validate_unit_cell", silent=True)
        return False

    # Check if values are reasonable
    for key in ["a", "b", "c"]:
        value = unit_cell_info[key]
        if not isinstance(value, (int, float)) or value <= 0 or value > 1000:
            log.warning(f"Invalid unit cell length {key}: {value}", scope="validate_unit_cell", silent=True)
            return False

    for key in ["alpha", "beta", "gamma"]:
        value = unit_cell_info[key]
        if not isinstance(value, (int, float)) or value <= 0 or value >= 180:
            log.warning(f"Invalid unit cell angle {key}: {value}", scope="validate_unit_cell", silent=True)
            return False

    log.info("✅ Unit cell parameters validated successfully", scope="validate_unit_cell", silent=True)
    return True


def format_unit_cell_display(unit_cell_info: Dict[str, Any]) -> str:
    """
    Format unit cell information for display.

    Args:
        unit_cell_info: Dictionary containing unit cell parameters

    Returns:
        Formatted string for display
    """
    if not unit_cell_info:
        return "No unit cell information available"

    a = unit_cell_info.get("a", "--")
    b = unit_cell_info.get("b", "--")
    c = unit_cell_info.get("c", "--")
    alpha = unit_cell_info.get("alpha", "--")
    beta = unit_cell_info.get("beta", "--")
    gamma = unit_cell_info.get("gamma", "--")
    space_group = unit_cell_info.get("space_group", "Unknown")
    source = unit_cell_info.get("source", "Unknown")

    display = f"Unit Cell: a={a:.2f}, b={b:.2f}, c={c:.2f} Å\n"
    display += f"Angles: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°\n"
    display += f"Space Group: {space_group}\n"
    display += f"Source: {source}"

    return display
