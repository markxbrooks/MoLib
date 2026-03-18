"""
Ligand parser parser for SDF and CSV formats with SMILES strings.

This module provides functionality to parse ligand files and extract molecular information
including SMILES, molecular weight, and other parameters using RDKit.
"""

from pathlib import Path
from typing import List

import pandas as pd
from decologr import Decologr as log
from molib.ligand.file.detect import detect_delimiter, detect_smiles_and_name_columns

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    Crippen = None
    Descriptors = None
    rdMolDescriptors = None


class LigandFileParser:
    """Parser for ligand files in SDF and CSV formats"""

    def __init__(self):
        """Initialize the LigandFileParser"""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ligand parsing but not available")
        self.parsers = {
            ".sdf": self._parse_sdf_file,
            ".csv": self._parse_csv_file,
        }

    def parse_file(self, file_path: str) -> List["LigandInfo"]:
        """
        Parse a ligand parser and return a list of LigandInfo objects.

        :param: file_path: Path to the ligand parser (SDF or CSV)
        :return: List of LigandInfo objects
        :raises:   ValueError: If parser structure_format is not supported
                    FileNotFoundError: If parser doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        suffix = file_path.suffix.lower()

        try:
            parser = self.parsers[suffix]
        except KeyError:
            raise ValueError(
                f"Unsupported format: {suffix}. Supported formats: {list(self.parsers)}"
            )

        return parser(file_path)

    def _parse_sdf_file(self, file_path: Path) -> List["LigandInfo"]:
        """Parse an SDF parser"""
        log.info(f"Parsing SDF parser: {file_path}")

        ligands = []
        supplier = Chem.SDMolSupplier(str(file_path))

        for i, mol in enumerate(supplier):
            if mol is None:
                log.warning(f"Skipping invalid molecule at index {i}")
                continue

            try:
                self.add_ligand_from_mol(i, ligands, mol, name=None)
            except Exception as e:
                log.error(f"Error parsing molecule {i+1}: {e}")
                continue

        log.info(f"Successfully parsed {len(ligands)} ligands from SDF parser")
        return ligands

    def _parse_csv_file(self, file_path: Path) -> List["LigandInfo"]:
        """Parse a CSV parser with SMILES strings"""
        log.info(f"Parsing CSV parser: {file_path}")

        ligands = []

        delimiter = detect_delimiter(file_path)

        # --- Read CSV
        df = pd.read_csv(file_path, delimiter=delimiter)

        name_col, smiles_col = detect_smiles_and_name_columns(df)

        # --- Iterate over rows
        for i, row in df.iterrows():
            smiles = str(row[smiles_col]).strip() if row[smiles_col] else ""
            name = str(row[name_col]).strip() if name_col else f"mol_{i + 1}"

            if not smiles:
                log.warning(f"Skipping row {i + 1}: empty SMILES")
                continue

            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    log.warning(f"Skipping invalid SMILES at row {i + 1}: {smiles}")
                    continue

                self.add_ligand_from_mol(i, ligands, mol, name)

            except Exception as e:
                log.error(f"Error parsing SMILES at row {i + 1}: {e}")
                continue

        log.info(f"Successfully parsed {len(ligands)} ligands from CSV parser")
        return ligands

    def add_ligand_from_mol(
        self, i, ligands: list["LigandInfo"], mol: Chem.Mol, name: str | None = None
    ):
        """Add ligand from RDKit molecule object"""
        ligand_info = self._create_ligand_info_from_mol(mol, name, i)
        ligands.append(ligand_info)
        log.debug(f"Parsed ligand {i + 1}: {ligand_info.name}")

    def _create_ligand_info_from_mol(
        self, mol: Chem.Mol, name: str, ligand_id: int | None = None
    ) -> "LigandInfo":
        """Create LigandInfo from RDKit molecule object"""

        try:
            from molib.ligand.info import LigandInfo

            return LigandInfo(
                name=name,
                smiles=Chem.MolToSmiles(mol, canonical=True),
                mol=mol,
                ligand_id=ligand_id,
                inchikey=Chem.inchi.MolToInchiKey(mol),
                canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
            )

        except Exception as e:
            log.error(f"Error creating LigandInfo for molecule {name}: {e}")
            raise


def check_rdkit_availability() -> bool:
    """Check if RDKit is available and working"""
    return RDKIT_AVAILABLE
