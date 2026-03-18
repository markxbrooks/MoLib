import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import gemmi
import pandas as pd
from decologr import Decologr as log

# Residues that often have no CIF in CCP4 monomer lib (water, common ligands);
# log missing CIF at DEBUG to avoid cluttering normal runs.
_COMMON_RESIDUES_NO_CIF = frozenset({"HOH", "WAT", "H2O", "MPD", "ANP", "MRD"})


def find_ccp4_installation() -> Optional[Path]:
    """
    Find CCP4 installation directory using PATH environment variable and common locations.

    :return: Path to CCP4 installation directory or None if not found.
    """
    try:
        # Look for ccp4i2 in PATH
        ccp4i2_path = shutil.which("ccp4i2")
        if ccp4i2_path:
            # ccp4i2 is typically in /path/to/ccp4/bin/ccp4i2
            # So the CCP4 root would be the parent of the bin directory
            ccp4_root = Path(ccp4i2_path).parent.parent
            log.info(f"Found CCP4 installation at: {ccp4_root}")
            return ccp4_root

        # Alternative: look for other CCP4 executables
        for exe in ["refmac5", "coot", "ccp4-python"]:
            exe_path = shutil.which(exe)
            if exe_path:
                ccp4_root = Path(exe_path).parent.parent
                log.info(f"Found CCP4 installation via {exe} at: {ccp4_root}")
                return ccp4_root

        # Check common installation paths (especially for macOS)
        common_paths = [
            "/Applications/ccp4-9",
            "/Applications/ccp4-8",
            "/Applications/ccp4-7",
            "/opt/apps/ccp4/ccp4-8.0",
            "/opt/apps/ccp4/ccp4-9.0",
            "/usr/local/ccp4",
            "/opt/ccp4",
            # Windows-style user installs
            Path.home() / "CCP4-9",  # e.g. C:\\Users\\<user>\\CCP4-9
            Path.home() / "CCP4-9" / "9.0",  # versioned subdir if present
            Path.home() / "CCP4-8" / "8.0",
        ]

        for path in common_paths:
            ccp4_path = Path(path)
            if ccp4_path.exists():
                # Check if it looks like a CCP4 installation
                # Accept multiple layouts (Linux/macOS and Windows CCP4-9 tree)
                monomer_candidates = [
                    ccp4_path / "lib" / "data" / "monomers",
                    ccp4_path / "Lib" / "data" / "monomers",
                    ccp4_path / "share" / "ccp4" / "lib" / "data" / "monomers",
                    ccp4_path / "CCP4" / "lib" / "data" / "monomers",
                    ccp4_path / "9.0" / "CCP4" / "lib" / "data" / "monomers",
                ]
                if (ccp4_path / "bin" / "ccp4i2").exists() or any(
                    p.exists() for p in monomer_candidates
                ):
                    log.info(f"Found CCP4 installation at: {ccp4_path}")
                    return ccp4_path

    except Exception as e:
        log.warning(f"Error finding CCP4 installation: {e}")

    return None


def get_unique_hetatm_cif_dicts(
    hetatm_df: pd.DataFrame, clibd_mon: Optional[Union[str, Path]] = None
) -> Dict[str, gemmi.cif.Block]:
    """
    Return a dictionary of gemmi CIF blocks for unique HETATM residue names.

    :param hetatm_df: DataFrame with a 'residue_name' column.
    :param clibd_mon: Path to CCP4 monomer CIF directory (optional).
    :return: Dictionary mapping residue names to gemmi.CifBlock objects.
    """
    clibd_mon = get_clibd_monomers_directory(clibd_mon)

    cif_blocks: Dict[str, gemmi.cif.Block] = {}

    for resname in hetatm_df["residue_name"].unique():
        cif_path = clibd_mon / resname[0].upper() / f"{resname.upper()}.cif"
        if cif_path.is_file():
            doc = gemmi.cif.read_file(str(cif_path))
            cif_blocks[resname] = doc[0]
        else:
            msg = f"CIF file not found for residue '{resname}': {cif_path}"
            if resname.strip().upper() in _COMMON_RESIDUES_NO_CIF:
                log.message(msg, level=logging.DEBUG)
            else:
                log.warning(msg)

    return cif_blocks


def get_clibd_monomers_directory(clibd_mon: Optional[Union[str, Path]] = None) -> Path:
    """
    get_clibd_monomers_directory

    :param clibd_mon: Optional path to the monomer directory.
    :return: Resolved Path object.

    Resolve the path to the monomer CIF directory.
    Uses CLIBD_MON environment variable if not explicitly provided.
    Falls back to finding CCP4 installation via PATH, then hardcoded path.
    """
    if clibd_mon is None:
        clibd_mon = os.environ.get("CLIBD_MON")
        if clibd_mon is None:
            # Try to find CCP4 installation using PATH
            ccp4_root = find_ccp4_installation()
            if ccp4_root:
                # Try different possible paths for the monomer library
                possible_paths = [
                    ccp4_root / "Lib" / "data" / "monomers",
                    ccp4_root / "lib" / "data" / "monomers",
                    ccp4_root / "share" / "ccp4" / "lib" / "data" / "monomers",
                    # Versioned layouts
                    ccp4_root / "9.0" / "Lib" / "data" / "monomers",
                    ccp4_root / "9.0" / "CCP4" / "lib" / "data" / "monomers",
                    ccp4_root / "8.0" / "Lib" / "data" / "monomers",
                    # Windows CCP4-9 dir with CCP4 subfolder
                    ccp4_root / "CCP4" / "lib" / "data" / "monomers",
                ]

                for path in possible_paths:
                    if path.exists():
                        clibd_mon = path
                        log.info(f"Found monomer library at: {clibd_mon}")
                        break

                if clibd_mon is None:
                    # Use the first possible path as default
                    clibd_mon = possible_paths[0]
                    log.warning(
                        f"Monomer library not found, using default path: {clibd_mon}"
                    )
            else:
                # Fall back to hardcoded path; prefer CCP4-9 if present
                ccp4_9 = Path.home() / "CCP4-9" / "CCP4" / "lib" / "data" / "monomers"
                ccp4_8 = Path.home() / "CCP4-8" / "8.0" / "Lib" / "data" / "monomers"
                clibd_mon = ccp4_9 if ccp4_9.exists() else ccp4_8
                log.warning(
                    f"CCP4 installation not found in PATH, using hardcoded path: {clibd_mon}"
                )

    return Path(clibd_mon).resolve()


def validate_structure(structure: gemmi.Structure) -> dict | None:
    """
    validate_structure

    :param structure: gemmi.Structure to validate.
    :return: Dictionary with validation results or None if validation fails.
    Validate a gemmi.Structure against the CCP4 monomer library.
    """
    try:
        validation_results = {}
        model = structure[0]  # First model
        residue_names = model.get_all_residue_names()

        monomer_lib_path = get_clibd_monomers_directory()
        log.parameter("monomer_lib_path", monomer_lib_path)

        monomer_lib = gemmi.MonLib()
        monomer_lib.read_monomer_lib(
            str(monomer_lib_path), residue_names, logging=sys.stderr
        )

        topology = gemmi.prepare_topology(structure, monomer_lib, 0)
        log.parameter("topology", topology)

        # Example: check if all residues have expected atoms
        validation_results["missing_atoms"] = topology.find_missing_atoms()

        return validation_results

    except Exception as ex:
        log.error(f"Error {ex} occurred validating results")
        return None


def get_unique_hetatm_cif_blocks(
    hetatm_df: pd.DataFrame, clibd_mon: Optional[Union[str, Path]] = None
) -> dict[str, gemmi.cif.Block]:
    """
    Load CIF blocks for each unique HETATM residue using gemmi.

    Args:
        hetatm_df (pd.DataFrame): DataFrame containing HETATM
        records with a 'residue_name' column.
        clibd_mon (str): Path to CCP4 monomer CIF directory
         (default from CLIBD_MON env variable).

    Returns:
        dict[str, gemmi.cif.Block]: Dictionary mapping residue
        names to their CIF data blocks.
    """
    clibd_mon = get_clibd_monomers_directory(clibd_mon)

    unique_residues = hetatm_df["residue_name"].unique()
    cif_blocks = {}

    for res in unique_residues:
        res = res.strip().upper()
        cif_path = Path(clibd_mon) / res[0] / f"{res}.cif"

        if cif_path.exists():
            doc = gemmi.cif.read(str(cif_path))
            if doc and len(doc) > 0:
                cif_blocks[res] = doc[0]  # Most CCP4 monomer CIFs have one block
            else:
                log.message(f"Empty or invalid CIF for {res} at {cif_path}")
        else:
            msg = f"CIF file not found for {res}: {cif_path}"
            if res in _COMMON_RESIDUES_NO_CIF:
                log.message(msg, level=logging.DEBUG)
            else:
                log.message(msg)

    return cif_blocks


def get_unique_hetatm_cif_paths(
    hetatm_df: pd.DataFrame, clibd_mon: Optional[Union[str, Path]] = None
) -> dict[str, Path]:
    """
    Returns a dictionary mapping each unique residue_name from hetatm_df
    to its corresponding CIF file path in the CCP4 CLIBD_MON directory.

    :param hetatm_df: DataFrame containing a 'residue_name' column
    :param clibd_mon: Path to CCP4's CLIBD_MON directory
    :return: Dictionary {residue_name: cif_path}
    """
    clibd_mon = get_clibd_monomers_directory(clibd_mon)

    if "residue_name" not in hetatm_df.columns:
        raise ValueError("hetatm_df must contain a 'residue_name' column.")

    unique_resnames = hetatm_df["residue_name"].unique()
    cif_paths = {}

    for resname in unique_resnames:
        subdir = resname[0].upper()
        cif_file = f"{resname.upper()}.cif"
        full_path = Path(clibd_mon) / subdir / cif_file

        if full_path.exists():
            cif_paths[resname] = full_path
        else:
            cif_paths[resname] = (
                None  # or log a warning, or raise, depending on context
            )

    return cif_paths
