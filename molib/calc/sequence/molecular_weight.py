from Bio.SeqUtils.ProtParam import ProteinAnalysis
from decologr import Decologr as log


def calculate_protein_sequence_molecular_weight(sequence_text: str) -> float | None:
    """
    calculate_protein_sequence_molecular_weight

    :param sequence_text: str
    :return: float | None
    Calculates the molecular weight of a protein sequence.
    """
    try:
        return ProteinAnalysis(sequence_text).molecular_weight()
    except Exception as ex:
        log.error(f"Error calculating molecular weight: {ex}")
        return None
