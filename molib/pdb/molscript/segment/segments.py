"""
Functions for setting up segments
"""

import numpy as np
from decologr import Decologr as log
from molib.core.color import ColorMap
from molib.pdb.molscript.segment.coil import create_coil_segments
from molib.pdb.molscript.segment.helix import create_helix_segments
from molib.pdb.molscript.segment.strand import create_strand_segments
from molib.pdb.molscript.segment.tube import _create_simple_backbone_tube_mesh

# Performance optimization: Cache for secondary structure processing
_secondary_structure_cache = {}


def update_secondary_structure_colors(model: "Molecule3D"):
    """
    Update the colors of existing secondary structure segments when the color map changes.

    Args:
        model: Molecular model with render_buffers.secondary_struct
    """
    if not hasattr(model, "render_buffers") or not hasattr(
        model.render_buffers, "secondary_struct"
    ):
        log.warning("Model does not have secondary structure buffers")
        return

    secondary_struct = model.render_buffers.secondary_struct

    # Get the current color map
    colors = ColorMap.get_ss_colors()

    # Update helix segment colors
    for segment in secondary_struct.helix_segments:
        if hasattr(segment, "c"):
            segment.c = colors["H"]

    # Update strand segment colors
    for segment in secondary_struct.strand_segments:
        if hasattr(segment, "c"):
            segment.c = colors["E"]

    # Update coil segment colors
    for segment in secondary_struct.coil_segments:
        if hasattr(segment, "c"):
            segment.c = colors[" "]

    log.debug("Updated secondary structure segment colors")


def extract_secondary_structure_segments(model: "Molecule3D"):
    """
    Extract secondary structure segments from a molecular model and populate
    the molecule's render_buffers.secondary_struct for OpenGL rendering.

    Args:
        model: Molecular model with get_all_residues() method
    """
    scope = "extract_secondary_structure_segments"
    # Get the secondary structure buffers from the molecule
    if not hasattr(model, "render_buffers") or not hasattr(
        model.render_buffers, "secondary_struct"
    ):
        log.warning("Model does not have secondary structure buffers")
        return

    secondary_struct = model.render_buffers.secondary_struct

    # Calculate model hash for dirty checking
    # Create a stable hash based on actual molecular data, not string representation
    residues = model.get_all_residues()
    if residues:
        # Hash based on residue sequence data (residue number, chain, atom count)
        stable_data = []
        for residue in residues:
            stable_data.append(
                (
                    getattr(residue, "residue_number", 0),
                    getattr(residue, "chain_id", ""),
                    len(getattr(residue, "atoms", [])),
                    getattr(residue, "residue_name", ""),
                )
            )
        model_hash = hash(tuple(stable_data))
    else:
        model_hash = 0

    # Performance optimization: Check cache first
    model_name = (
        getattr(model, "name", None)
        or getattr(getattr(model, "molecule", None), "name", None)
        or "unknown"
    )
    cache_key = f"{model_name}_{model_hash}"
    if cache_key in _secondary_structure_cache:
        log.debug(
            "Using cached secondary structure data",
            scope="extract_secondary_structure_segments",
        )
        cached_data = _secondary_structure_cache[cache_key]
        secondary_struct.helix_segments = cached_data["helix_segments"]
        secondary_struct.strand_segments = cached_data["strand_segments"]
        secondary_struct.coil_segments = cached_data["coil_segments"]
        secondary_struct.helix_segment_count = cached_data["helix_segment_count"]
        secondary_struct.strand_segment_count = cached_data["strand_segment_count"]
        secondary_struct.coil_segment_count = cached_data["coil_segment_count"]
        # Restore the tube from cache if available
        if "tube" in cached_data:
            secondary_struct.tube = cached_data["tube"]
        # Ensure MeshData for helix/strand/coil is (re)built from cached segments
        if hasattr(secondary_struct, "build_meshdata_from_segments"):
            secondary_struct.build_meshdata_from_segments()
        return

    # Check if buffers need regeneration
    if not secondary_struct.needs_regeneration(model_hash):
        # Buffers are up to date; rebuild MeshData from existing segments for modern rendering
        if hasattr(secondary_struct, "build_meshdata_from_segments"):
            secondary_struct.build_meshdata_from_segments()
        return  # Buffers are up to date

    # Clear existing segments
    secondary_struct.helix_segments.clear()
    secondary_struct.strand_segments.clear()
    secondary_struct.coil_segments.clear()
    secondary_struct.helix_segment_count = 0
    secondary_struct.strand_segment_count = 0
    secondary_struct.coil_segment_count = 0

    try:
        # Get all residues from the model
        residues = list(model.get_all_residues())
        if not residues:
            log.message("No residues found in model")
            return

        log.message(f"Processing {len(residues)} residues for secondary structure")

        # Group residues by chain and secondary structure
        current_ss = None
        current_chain = None
        current_segment = []

        helix_count = 0
        strand_count = 0
        coil_count = 0

        for residue in residues:
            # Get secondary structure type
            ss_type = getattr(residue, "secstruc", " ")
            chain_id = getattr(residue, "chain_id", "A")

            # Count secondary structure types
            if ss_type == "H":
                helix_count += 1
            elif ss_type == "E":
                strand_count += 1
            elif ss_type in [" ", "-", "T", "C"]:
                coil_count += 1

            # Check if we need to start a new segment
            if ss_type != current_ss or chain_id != current_chain:
                # Process the previous segment if it exists
                if current_segment:
                    if current_ss in ["H", "E"]:
                        log.message(
                            f"Creating {current_ss} segment with {len(current_segment)} residues"
                        )
                        create_segment_geometry(
                            current_segment, current_ss, secondary_struct
                        )
                    elif (
                        current_ss in [" ", "-", "T", "C"] and len(current_segment) >= 2
                    ):
                        log.message(
                            f"Creating coil segment with {len(current_segment)} residues"
                        )
                        create_coil_segments(current_segment, secondary_struct)

                # Start new segment
                current_ss = ss_type
                current_chain = chain_id
                current_segment = [residue]
            else:
                # Continue current segment
                current_segment.append(residue)

        # Process the final segment
        if current_segment:
            if current_ss in ["H", "E"]:
                log.message(
                    f"Creating final {current_ss} segment with {len(current_segment)} residues"
                )
                create_segment_geometry(current_segment, current_ss, secondary_struct)
            elif current_ss in [" ", "-", "T", "C"] and len(current_segment) >= 2:
                log.message(
                    f"Creating final coil segment with {len(current_segment)} residues"
                )
                create_coil_segments(current_segment, secondary_struct)

        log.message(
            f"Secondary structure counts: H={helix_count}, E={strand_count}, C/coil={coil_count}",
            scope=scope,
        )
        log.message(
            f"Created segments: helix={secondary_struct.helix_segment_count}, strand={secondary_struct.strand_segment_count}, coil={secondary_struct.coil_segment_count}",
            scope=scope,
        )

        # After building ribbon segments, also generate a smoothed backbone tube
        try:
            _create_simple_backbone_tube_mesh(model, secondary_struct)
        except Exception as ex:
            log.error(f"Error creating backbone tube: {ex}", scope=scope)

        # Mark buffers as clean and up-to-date
        secondary_struct.mark_clean(model_hash)

        # Performance optimization: Cache the results
        _secondary_structure_cache[cache_key] = {
            "helix_segments": secondary_struct.helix_segments.copy(),
            "strand_segments": secondary_struct.strand_segments.copy(),
            "coil_segments": secondary_struct.coil_segments.copy(),
            "helix_segment_count": secondary_struct.helix_segment_count,
            "strand_segment_count": secondary_struct.strand_segment_count,
            "coil_segment_count": secondary_struct.coil_segment_count,
            "tube": secondary_struct.tube,  # Cache the tube as well
        }

        # Build MeshData objects for helix/strand/coil so modern renderer can draw them
        if hasattr(secondary_struct, "build_meshdata_from_segments"):
            secondary_struct.build_meshdata_from_segments()

    except Exception as ex:
        log.error(f"Error extracting secondary structure: {ex}", scope=scope)
        import traceback

        traceback.print_exc()


def create_segment_geometry(
    residues, ss_type, secondary_struct: "SecondaryStructureBuffers"
):
    """Create geometry for a secondary structure segment."""

    log.message(f"Creating {ss_type} segment geometry for {len(residues)} residues")

    if len(residues) < 2:
        log.message(f"Not enough residues ({len(residues)}) for segment")
        return

    # Get CA coordinates for the segment
    ca_coords = []
    for i, residue in enumerate(residues):
        if hasattr(residue, "ca") and residue.ca is not None:
            ca_coords.append(residue.ca)
            log.message(f"  Residue {i}: found CA coords {residue.ca}")
        elif hasattr(residue, "coords"):
            ca_coords.append(residue.coords)
            log.message(f"  Residue {i}: found coords {residue.coords}")
        else:
            log.message(
                f"  Residue {i}: no CA or coords found, attributes: {dir(residue)}"
            )

    log.message(f"Found {len(ca_coords)} CA coordinates")

    if len(ca_coords) < 2:
        log.message(f"Not enough CA coordinates ({len(ca_coords)}) for segment")
        return

    # Convert to numpy array
    coords = np.array(ca_coords)
    log.message(f"Created coords array with shape {coords.shape}")

    # Create segment objects based on type
    if ss_type == "H":
        log.message(f"Creating helix segments")
        helix_count = create_helix_segments(coords, residues, secondary_struct)
        log.message(f"  Updated helix count: {helix_count}")
    elif ss_type == "E":
        log.message(f"Creating strand segments")
        strand_count = create_strand_segments(coords, residues, secondary_struct)
        log.message(f"  Updated strand count: {strand_count}")
    elif ss_type in [" ", "-", "T", "C"]:
        log.message(f"Creating coil segments")
        coil_count = create_coil_segments(residues, secondary_struct)
        log.message(f"  Updated coil count: {coil_count}")
