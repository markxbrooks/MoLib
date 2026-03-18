"""
PDB ligand parser for extracting ligand from PDB files.

This module provides functionality to parse PDB files and extract ligand molecules
(HETATM records) with their 3D coordinates and convert them to SMILES for display.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from molib.ligand.pdb.info import PDBLigandInfo
from molib.ligand.pdb.layouts.hetatm import HETATMLayout
from molib.ligand.pdb.spec import PDBLineSpec
from molib.ligand.rdkit.helpers import create_pdb_ligand_info

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Mol, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    AllChem = None

from decologr import Decologr as log

@dataclass
class PDBLigandData:
    """PDBLigandData"""
    res_name: str
    chain_id: str
    res_seq: int
    insertion_code: str | None
    atoms: list[dict] = field(default_factory=list)


class PDBLigandParser:
    """Parser for extracting ligand from PDB files"""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            log.warning("RDKit is not available - PDB ligand parsing will be limited")

    def parse_pdb_file(
        self, pdb_file_path: str | Path, deduplicate: bool = True
    ) -> List["PDBLigandInfo"]:
        """
        Parse a PDB parser and extract ligand information.

        Args:
            pdb_file_path: Path to the PDB parser
            deduplicate: If True, group identical ligand by chemical structure (SMILES)

        Returns:
            List of PDBLigandInfo objects
        """
        pdb_file_path = str(pdb_file_path)
        log.info("PDBLigandParser: Parsing PDB parser: %s", pdb_file_path)

        if not os.path.exists(pdb_file_path):
            raise FileNotFoundError(f"PDB parser not found: {pdb_file_path}")

        ligand_groups = self._group_hetatm_records(pdb_file_path)
        log.info(
            "PDBLigandParser: Found %d ligand groups",
            len(ligand_groups),
        )

        ligands: list["PDBLigandInfo"] = []

        for ligand_key, group in ligand_groups.items():
            try:
                pdb_ligand_data = PDBLigandData(res_name=group.res_name, chain_id=group.chain_id, res_seq=group.res_seq, insertion_code=group.insertion_code, atoms=group.atoms)
                pdb_ligand = create_pdb_ligand_info(pdb_ligand_data)

                if pdb_ligand is not None:
                    ligands.append(pdb_ligand)
            except Exception:
                log.exception(
                    "PDBLigandParser: Failed to create ligand for %s",
                    ligand_key,
                )

        log.info(
            "PDBLigandParser: Extracted %d ligand before deduplication",
            len(ligands),
        )

        if deduplicate and ligands:
            ligands = self._deduplicate_ligands(ligands)
            log.info(
                "PDBLigandParser: %d ligand after deduplication",
                len(ligands),
            )

        return ligands

    def _group_hetatm_records(self, pdb_file_path: str) -> Dict[str, PDBLigandData]:
        ligand_groups: Dict[str, PDBLigandData] = {}

        with open(pdb_file_path, "r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, 1):
                if not line.startswith("HETATM"):
                    continue

                ligand_info = self._parse_hetatm_line(line, line_num)
                if ligand_info is None:
                    continue

                ligand_key = (
                    f"{ligand_info['res_name']}_"
                    f"{ligand_info['chain_id']}_"
                    f"{ligand_info['res_seq']}"
                )

                if ligand_key not in ligand_groups:
                    ligand_groups[ligand_key] = PDBLigandData(
                        res_name=ligand_info["res_name"],
                        chain_id=ligand_info["chain_id"],
                        res_seq=ligand_info["res_seq"],
                        insertion_code=ligand_info["insertion_code"],
                    )

                ligand_groups[ligand_key].atoms.append(ligand_info)

        return ligand_groups

    def _ligand_dedup_key(self, ligand: "PDBLigandInfo") -> str:
        """
        Return a canonical key representing the chemical identity of a ligand.
        """

        # 1. Primary: SMILES (normalized)
        if ligand.canonical_smiles:
            return ligand.canonical_smiles

        # 2. Primary: SMILES (normalized)
        if ligand.smiles:
            smiles = ligand.smiles.strip()
            if smiles:
                return smiles

        # 2. Secondary: empirical formula + atom count
        if ligand.formula and ligand.atom_count > 0:
            return f"{ligand.formula}:{ligand.atom_count}"

        # 3. Absolute fallback: residue identity (never collapses globally)
        return f"{ligand.ligand_id}:{ligand.chain_id}:{ligand.res_seq}{ligand.insertion_code or ''}"

    def _group_ligands_by_identity(
        self,
        ligands: list[PDBLigandInfo],
    ) -> dict[str, list[PDBLigandInfo]]:
        groups = defaultdict(list)
        for ligand in ligands:
            groups[self._ligand_dedup_key(ligand)].append(ligand)
        return groups

    def _select_best_ligand(
        self,
        ligands: list[PDBLigandInfo],
    ) -> PDBLigandInfo:
        """
        Select the most complete ligand instance.
        """
        return max(
            ligands,
            key=lambda l: (
                l.atom_count,
                bool(l.smiles),
                bool(l.coordinates),
            ),
        )

    def _deduplicate_ligands(
        self,
        ligands: list[PDBLigandInfo],
    ) -> list[PDBLigandInfo]:
        """
        Deduplicate ligand by chemical identity and retain the most complete instance.
        """
        if not ligands:
            return []

        log.info(
            "PDBLigandParser: Deduplicating %d ligand",
            len(ligands),
        )

        groups = self._group_ligands_by_identity(ligands)
        deduplicated: list[PDBLigandInfo] = []

        for key, group in groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
                continue

            best = self._select_best_ligand(group)

            log.info(
                "PDBLigandParser: %d instances of %s → selected %s (%d atoms)",
                len(group),
                key[:40],
                best.ligand_id,
                best.atom_count,
            )

            deduplicated.append(best)

        log.info(
            "PDBLigandParser: Deduplication complete (%d → %d)",
            len(ligands),
            len(deduplicated),
        )

        return deduplicated

    def _parse_hetatm_line(self, line: str, line_num: int) -> Optional[dict]:
        """Parse a HETATM line from a PDB parser using declarative layout."""
        if len(line) < 80:
            log.debug(
                f"PDBLigandParser: Line {line_num} too short ({len(line)} chars), skipping"
            )
            return None

        record_type = HETATMLayout.record_type.parse(line)
        if record_type != "HETATM":
            return None

        try:
            parsed = {
                spec.name: spec.parse(line)
                for spec in HETATMLayout.__dict__.values()
                if isinstance(spec, PDBLineSpec)
            }

            return {
                "atom_serial": parsed["atom_serial"],
                "atom_name": parsed["atom_name"],
                "res_name": parsed["res_name"],
                "chain_id": parsed["chain_id"],
                "res_seq": parsed["res_seq"],
                "insertion_code": parsed["insertion_code"],
                "coordinates": (
                    float(parsed["x"]),
                    float(parsed["y"]),
                    float(parsed["z"]),
                ),
                "occupancy": parsed["occupancy"],
                "temp_factor": parsed["temp_factor"],
                "element": parsed["element"],
            }

        except Exception as exc:
            log.warning(f"PDBLigandParser: Error parsing HETATM line {line_num}: {exc}")
            log.debug(f"PDBLigandParser: Problematic line: {line.rstrip()}")
            return None
