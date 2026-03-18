import numpy as np
from decologr import Decologr as log
from molib.calc.geometry.spline import catmull_rom_chain
from molib.entities.secondary_structure_type import SecondaryStructureType
from picogl.renderer import MeshData


def create_backbone_tube_mesh(
    model,
    secondary_struct,
    radius: float = 0.35,
    ring_segments: int = 12,
    samples_per_segment: int = 8,
):
    """Generate tube segments for coil regions only.

    For each chain, create tubes only for coil regions (intervening spaces between
    secondary structure elements) by following CA atoms and creating smooth tube geometry.
    """
    all_residues = list(model.get_all_residues())
    if not all_residues:
        secondary_struct.tube = MeshData()
        return

    # Group residues by chain while preserving order
    chains = {}
    for res in all_residues:
        chain_id = getattr(res, "chain_id", "A")
        chains.setdefault(chain_id, []).append(res)

    # Helper: build tube mesh arrays for one polyline centerline
    def sweep_tube(centerline: np.ndarray):
        num_points = len(centerline)
        if num_points < 2:
            return None

        # Tangents
        tangents = np.zeros_like(centerline)
        tangents[1:-1] = centerline[2:] - centerline[:-2]
        tangents[0] = centerline[1] - centerline[0]
        tangents[-1] = centerline[-1] - centerline[-2]
        lens = np.linalg.norm(tangents, axis=1)
        lens[lens == 0] = 1.0
        tangents = (tangents.T / lens).T

        def orthogonal(v):
            x, y, z = v
            if abs(x) < 0.9:
                base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                base = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            n = np.cross(v, base)
            n_len = np.linalg.norm(n)
            if n_len == 0:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return n / n_len

        normals = np.zeros_like(centerline)
        binormals = np.zeros_like(centerline)
        normals[0] = orthogonal(tangents[0])
        binormals[0] = np.cross(tangents[0], normals[0])
        b0_len = np.linalg.norm(binormals[0])
        binormals[0] = binormals[0] / (b0_len if b0_len != 0 else 1.0)

        for i in range(1, num_points):
            v = tangents[i - 1]
            w = tangents[i]
            axis = np.cross(v, w)
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-6:
                normals[i] = normals[i - 1]
                binormals[i] = binormals[i - 1]
                continue
            axis /= axis_len
            cos_theta = np.clip(np.dot(v, w), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            n_prev = normals[i - 1]
            k = axis
            n_rot = (
                n_prev * np.cos(theta)
                + np.cross(k, n_prev) * np.sin(theta)
                + k * np.dot(k, n_prev) * (1 - np.cos(theta))
            )
            normals[i] = n_rot / (np.linalg.norm(n_rot) + 1e-12)
            b = np.cross(tangents[i], normals[i])
            binormals[i] = b / (np.linalg.norm(b) + 1e-12)

        ring_angles = np.linspace(0.0, 2.0 * np.pi, ring_segments, endpoint=False)
        cos_a = np.cos(ring_angles)
        sin_a = np.sin(ring_angles)

        vertices = np.zeros((num_points * ring_segments, 3), dtype=np.float32)
        vnormals = np.zeros_like(vertices)

        for i in range(num_points):
            n = normals[i]
            b = binormals[i]
            center = centerline[i]
            for j in range(ring_segments):
                offset = n * (radius * cos_a[j]) + b * (radius * sin_a[j])
                idx = i * ring_segments + j
                vertices[idx] = center + offset
                vnormals[idx] = offset / (np.linalg.norm(offset) + 1e-12)

        indices = []
        for i in range(num_points - 1):
            ring0 = i * ring_segments
            ring1 = (i + 1) * ring_segments
            for j in range(ring_segments):
                j_next = (j + 1) % ring_segments
                a = ring0 + j
                b0 = ring0 + j_next
                c = ring1 + j
                d = ring1 + j_next
                indices.append([a, c, b0])
                indices.append([b0, c, d])

        if len(indices) == 0:
            return None
        return vertices, vnormals, np.asarray(indices, dtype=np.uint32).reshape(-1)

    # Create tubes for all secondary structure elements
    all_vertices = []
    all_normals = []
    all_indices = []
    vertex_offset = 0
    added_ranges = set()  # track (start_idx, end_idx) to avoid duplicates
    added_segments_log = []  # (chain, kind, start_resno, end_resno, num_pts)

    # First, create simple tubes for all secondary structure runs
    for chain_id, res_list in chains.items():
        if not res_list:
            continue

        # Build runs of same secondary structure type
        runs = []  # list of (ss_type, start_idx, end_idx)
        current_ss = getattr(res_list[0], "secstruc", " ")
        # Convert to SecondaryStructureType enum
        if isinstance(current_ss, str):
            current_ss = SecondaryStructureType.from_string(current_ss)
        elif not isinstance(current_ss, SecondaryStructureType):
            current_ss = SecondaryStructureType.COIL
        run_start = 0
        for i in range(1, len(res_list)):
            ss = getattr(res_list[i], "secstruc", " ")
            # Convert to SecondaryStructureType enum
            if isinstance(ss, str):
                ss = SecondaryStructureType.from_string(ss)
            elif not isinstance(ss, SecondaryStructureType):
                ss = SecondaryStructureType.COIL
            if ss != current_ss:
                runs.append((current_ss, run_start, i - 1))
                current_ss = ss
                run_start = i
        runs.append((current_ss, run_start, len(res_list) - 1))

        # Create tubes only for coil regions (intervening spaces between secondary structures)
        for i, (ss_type, start_idx, end_idx) in enumerate(runs):
            # Skip very short runs
            if end_idx - start_idx < 1:
                continue
            # Only generate tubes for coil regions (intervening spaces)
            if ss_type not in [SecondaryStructureType.COIL, " "]:
                continue  # Skip helices and sheets, only process coils

            # Trim the segment to limit extension beyond secondary structure
            trimmed_indices = _trim_tube_segment(runs, i, max_extension=3)
            if trimmed_indices is None:
                continue  # Skip this segment

            actual_start, actual_end = trimmed_indices

            # Extract residues for this run (using trimmed indices)
            seg_res = res_list[actual_start : actual_end + 1]

            # Extract CA coords
            ca_pts = []
            for res in seg_res:
                if hasattr(res, "ca") and res.ca is not None:
                    ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                elif hasattr(res, "coords") and res.coords is not None:
                    ca_pts.append(np.asarray(res.coords, dtype=np.float32))

            if len(ca_pts) < 2:
                continue

            control = np.vstack(ca_pts).astype(np.float32)

            # Spline if enough points, else use raw
            if len(control) >= 4:
                try:
                    centerline = catmull_rom_chain(
                        control, samples_per_segment=samples_per_segment
                    ).astype(np.float32)
                except Exception:
                    centerline = control
            else:
                centerline = control

            swept = sweep_tube(centerline)
            if swept is None:
                continue

            vtx, nrm, idx = swept
            all_vertices.append(vtx)
            all_normals.append(nrm)
            # offset indices
            all_indices.append(idx + vertex_offset)
            vertex_offset += vtx.shape[0]

            # Log meta
            start_resno = getattr(seg_res[0], "residue_number", -1)
            end_resno = getattr(seg_res[-1], "residue_number", -1)
            # Convert enum to string for logging
            ss_type_str = (
                ss_type.to_string()
                if isinstance(ss_type, SecondaryStructureType)
                else str(ss_type)
            )
            added_segments_log.append(
                (chain_id, ss_type_str, start_resno, end_resno, len(seg_res))
            )

    # If we already have tubes from the simple approach, skip the complex logic
    if len(all_vertices) == 0:
        # Fall back to the original complex logic for specific transitions
        # Find H->E transitions and build tubes per segment
        for chain_id, res_list in chains.items():
            # Build runs of same ss
            runs = []  # list of (ss_type, start_idx, end_idx)
            if not res_list:
                continue
            current_ss = getattr(res_list[0], "secstruc", " ")
        # Convert to SecondaryStructureType enum
        if isinstance(current_ss, str):
            current_ss = SecondaryStructureType.from_string(current_ss)
        elif not isinstance(current_ss, SecondaryStructureType):
            current_ss = SecondaryStructureType.COIL
        run_start = 0
        for i in range(1, len(res_list)):
            ss = getattr(res_list[i], "secstruc", " ")
            # Convert to SecondaryStructureType enum
            if isinstance(ss, str):
                ss = SecondaryStructureType.from_string(ss)
            elif not isinstance(ss, SecondaryStructureType):
                ss = SecondaryStructureType.COIL
            if ss != current_ss:
                runs.append((current_ss, run_start, i - 1))
                current_ss = ss
                run_start = i
        runs.append((current_ss, run_start, len(res_list) - 1))

        # Create segments from H run end to next E run start (skipping any coils between)
        for r_idx in range(len(runs) - 1):
            ss_curr, s0, e0 = runs[r_idx]
            if ss_curr != "H":
                continue
            # Look ahead for the next 'E'
            e_run_idx = None
            for r_j in range(r_idx + 1, len(runs)):
                if runs[r_j][0] == "E":
                    e_run_idx = r_j
                    break
                # stop if another helix appears before a sheet
                if runs[r_j][0] == "H":
                    break
            if e_run_idx is None:
                continue
            # Collect residues from end of helix to start of sheet inclusive,
            # but extend by one residue on each side if available to bridge gaps
            _, _, helix_end = runs[r_idx]
            _, sheet_start, _ = runs[e_run_idx]
            if sheet_start <= helix_end:
                continue
            start_idx = max(helix_end - 2, 0)
            end_idx = min(sheet_start + 2, len(res_list) - 1)
            seg_res = res_list[start_idx : end_idx + 1]
            range_key = (start_idx, end_idx)
            if range_key in added_ranges:
                continue
            # Extract CA coords
            ca_pts = []
            for res in seg_res:
                if hasattr(res, "ca") and res.ca is not None:
                    ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                elif hasattr(res, "coords") and res.coords is not None:
                    ca_pts.append(np.asarray(res.coords, dtype=np.float32))
            if len(ca_pts) < 2:
                continue
            control = np.vstack(ca_pts).astype(np.float32)
            # Spline if enough points, else use raw
            if len(control) >= 4:
                try:
                    centerline = catmull_rom_chain(
                        control, samples_per_segment=samples_per_segment
                    ).astype(np.float32)
                except Exception:
                    centerline = control
            else:
                centerline = control

            swept = sweep_tube(centerline)
            if swept is None:
                continue
            vtx, nrm, idx = swept
            all_vertices.append(vtx)
            all_normals.append(nrm)
            # offset indices
            all_indices.append(idx + vertex_offset)
            vertex_offset += vtx.shape[0]
            added_ranges.add(range_key)
            # Log meta
            start_resno = getattr(seg_res[0], "residue_number", -1)
            end_resno = getattr(seg_res[-1], "residue_number", -1)
            added_segments_log.append(
                (chain_id, "H->E", start_resno, end_resno, len(seg_res))
            )

        # Additionally, create segments from H run end to next H run start (only coils between)
        for r_idx in range(len(runs) - 1):
            ss_curr, s0, e0 = runs[r_idx]
            if ss_curr != "H":
                continue
            # Find next helix; abort if a sheet appears before it
            h2_run_idx = None
            blocked_by_sheet = False
            for r_j in range(r_idx + 1, len(runs)):
                if runs[r_j][0] == "E":
                    blocked_by_sheet = True
                    break
                if runs[r_j][0] == "H":
                    h2_run_idx = r_j
                    break
            if h2_run_idx is None or blocked_by_sheet:
                continue
            # Ensure only coils between r_idx and h2_run_idx
            intervening_ok = all(
                runs[k][0] in [" ", "-", "T", "C"] for k in range(r_idx + 1, h2_run_idx)
            )
            if not intervening_ok:
                continue
            # Build segment from end of first helix to start of second helix, extended by ±2
            _, _, helix_end = runs[r_idx]
            _, helix2_start, _ = runs[h2_run_idx]
            if helix2_start <= helix_end:
                continue
            start_idx = max(helix_end - 2, 0)
            end_idx = min(helix2_start + 2, len(res_list) - 1)
            seg_res = res_list[start_idx : end_idx + 1]
            range_key = (start_idx, end_idx)
            if range_key in added_ranges:
                continue
            ca_pts = []
            for res in seg_res:
                if hasattr(res, "ca") and res.ca is not None:
                    ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                elif hasattr(res, "coords") and res.coords is not None:
                    ca_pts.append(np.asarray(res.coords, dtype=np.float32))
            if len(ca_pts) < 2:
                continue
            control = np.vstack(ca_pts).astype(np.float32)
            if len(control) >= 4:
                try:
                    centerline = catmull_rom_chain(
                        control, samples_per_segment=samples_per_segment
                    ).astype(np.float32)
                except Exception:
                    centerline = control
            else:
                centerline = control
            swept = sweep_tube(centerline)
            if swept is None:
                continue
            vtx, nrm, idx = swept
            all_vertices.append(vtx)
            all_normals.append(nrm)
            all_indices.append(idx + vertex_offset)
            vertex_offset += vtx.shape[0]
            added_ranges.add(range_key)
            start_resno = getattr(seg_res[0], "residue_number", -1)
            end_resno = getattr(seg_res[-1], "residue_number", -1)
            added_segments_log.append(
                (chain_id, "H->H", start_resno, end_resno, len(seg_res))
            )

        # Additionally, create segments from E run end to next E run start (only coils between)
        for r_idx in range(len(runs) - 1):
            ss_curr, s0, e0 = runs[r_idx]
            if ss_curr != "E":
                continue
            # Find next sheet; abort if a helix appears before it
            e2_run_idx = None
            blocked_by_helix = False
            for r_j in range(r_idx + 1, len(runs)):
                if runs[r_j][0] == "H":
                    blocked_by_helix = True
                    break
                if runs[r_j][0] == "E":
                    e2_run_idx = r_j
                    break
            if e2_run_idx is None or blocked_by_helix:
                continue
            # Ensure only coils between r_idx and e2_run_idx
            intervening_ok = all(
                runs[k][0] in [" ", "-", "T", "C"] for k in range(r_idx + 1, e2_run_idx)
            )
            if not intervening_ok:
                continue
            # Build segment from end of first sheet to start of second sheet, extended by ±2
            _, _, sheet_end = runs[r_idx]
            _, sheet2_start, _ = runs[e2_run_idx]
            if sheet2_start <= sheet_end:
                continue
            start_idx = max(sheet_end - 2, 0)
            end_idx = min(sheet2_start + 2, len(res_list) - 1)
            seg_res = res_list[start_idx : end_idx + 1]
            range_key = (start_idx, end_idx)
            if range_key in added_ranges:
                continue
            ca_pts = []
            for res in seg_res:
                if hasattr(res, "ca") and res.ca is not None:
                    ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                elif hasattr(res, "coords") and res.coords is not None:
                    ca_pts.append(np.asarray(res.coords, dtype=np.float32))
            if len(ca_pts) < 2:
                continue
            control = np.vstack(ca_pts).astype(np.float32)
            if len(control) >= 4:
                try:
                    centerline = catmull_rom_chain(
                        control, samples_per_segment=samples_per_segment
                    ).astype(np.float32)
                except Exception:
                    centerline = control
            else:
                centerline = control
            swept = sweep_tube(centerline)
            if swept is None:
                continue
            vtx, nrm, idx = swept
            all_vertices.append(vtx)
            all_normals.append(nrm)
            all_indices.append(idx + vertex_offset)
            vertex_offset += vtx.shape[0]
            added_ranges.add(range_key)
            start_resno = getattr(seg_res[0], "residue_number", -1)
            end_resno = getattr(seg_res[-1], "residue_number", -1)
            added_segments_log.append(
                (chain_id, "E->E", start_resno, end_resno, len(seg_res))
            )

        # Generic bridging: any short coil gap (<= 10 residues) between two non-coil runs (H/E)
        max_coil_gap = 10
        for r_idx in range(len(runs) - 2):
            ss_a, s_a, e_a = runs[r_idx]
            ss_gap, s_g, e_g = runs[r_idx + 1]
            ss_b, s_b, e_b = runs[r_idx + 2]
            if (
                ss_a in ["H", "E"]
                and ss_b in ["H", "E"]
                and ss_gap in [" ", "-", "T", "C"]
            ):
                gap_len = e_g - s_g + 1
                if gap_len <= max_coil_gap and s_b > e_a:
                    start_idx = max(e_a - 2, 0)
                    end_idx = min(s_b + 2, len(res_list) - 1)
                    range_key = (start_idx, end_idx)
                    if range_key in added_ranges:
                        continue
                    seg_res = res_list[start_idx : end_idx + 1]
                    ca_pts = []
                    for res in seg_res:
                        if hasattr(res, "ca") and res.ca is not None:
                            ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                        elif hasattr(res, "coords") and res.coords is not None:
                            ca_pts.append(np.asarray(res.coords, dtype=np.float32))
                    if len(ca_pts) < 2:
                        continue
                    control = np.vstack(ca_pts).astype(np.float32)
                    if len(control) >= 4:
                        try:
                            centerline = catmull_rom_chain(
                                control, samples_per_segment=samples_per_segment
                            ).astype(np.float32)
                        except Exception:
                            centerline = control
                    else:
                        centerline = control
                    swept = sweep_tube(centerline)
                    if swept is None:
                        continue
                    vtx, nrm, idx = swept
                    all_vertices.append(vtx)
                    all_normals.append(nrm)
                    all_indices.append(idx + vertex_offset)
                    vertex_offset += vtx.shape[0]
                    added_ranges.add(range_key)
                    start_resno = getattr(seg_res[0], "residue_number", -1)
                    end_resno = getattr(seg_res[-1], "residue_number", -1)
                    added_segments_log.append(
                        (chain_id, "GAP", start_resno, end_resno, len(seg_res))
                    )

        # Explicit bridging ranges by residue number (for 2VUG)
        explicit_ranges = [(312, 317), (351, 355)]
        if res_list:
            # Map index to residue_number for this chain
            residue_numbers = [getattr(r, "residue_number", None) for r in res_list]
            # Precompute coil runs (by index) with their residue_number spans
            coil_runs = []  # list of (start_idx, end_idx, start_resno, end_resno)
            i = 0
            while i < len(res_list):
                ss = getattr(res_list[i], "secstruc", " ")
                if ss in [" ", "-", "T", "C"]:
                    j = i
                    while j + 1 < len(res_list) and getattr(
                        res_list[j + 1], "secstruc", " "
                    ) in [" ", "-", "T", "C"]:
                        j += 1
                    rn_s = getattr(res_list[i], "residue_number", None)
                    rn_e = getattr(res_list[j], "residue_number", None)
                    coil_runs.append((i, j, rn_s, rn_e))
                    i = j + 1
                else:
                    i += 1

            for rstart, rend in explicit_ranges:
                # Locate indices by residue_number inclusion
                idxs = [
                    k
                    for k, rn in enumerate(residue_numbers)
                    if rn is not None and rstart <= rn <= rend
                ]
                start_idx = None
                end_idx = None

                if idxs:
                    start_idx = max(min(idxs) - 10, 0)  # use wider ±10 window
                    end_idx = min(max(idxs) + 10, len(res_list) - 1)

                # Also look for any coil run whose residue_number span overlaps [rstart, rend]
                for cs, ce, rn_s, rn_e in coil_runs:
                    if rn_s is None or rn_e is None:
                        continue
                    if not (rn_e < rstart or rn_s > rend):
                        cs_ext = max(cs - 10, 0)
                        ce_ext = min(ce + 10, len(res_list) - 1)
                        if start_idx is None or cs_ext < start_idx:
                            start_idx = cs_ext
                        if end_idx is None or ce_ext > end_idx:
                            end_idx = ce_ext

                if start_idx is None or end_idx is None or end_idx <= start_idx:
                    continue

                # Anchor to nearest non-coil residues just outside the span for better alignment
                def is_coil_idx(ii: int) -> bool:
                    return getattr(res_list[ii], "secstruc", " ") in [
                        " ",
                        "-",
                        "T",
                        "C",
                    ]

                # Expand start_idx backward to include up to 2 preceding non-coil anchors
                back = start_idx
                anchors = 0
                while back > 0 and anchors < 2:
                    if not is_coil_idx(back - 1):
                        anchors += 1
                    back -= 1
                start_idx = min(start_idx, back)

                # Expand end_idx forward to include up to 2 following non-coil anchors
                fwd = end_idx
                anchors = 0
                while fwd + 1 < len(res_list) and anchors < 2:
                    if not is_coil_idx(fwd + 1):
                        anchors += 1
                    fwd += 1
                end_idx = max(end_idx, fwd)

                range_key = (start_idx, end_idx)
                if range_key in added_ranges:
                    continue

                seg_res = res_list[start_idx : end_idx + 1]
                ca_pts = []
                for res in seg_res:
                    if hasattr(res, "ca") and res.ca is not None:
                        ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                    elif hasattr(res, "coords") and res.coords is not None:
                        ca_pts.append(np.asarray(res.coords, dtype=np.float32))
                if len(ca_pts) < 2:
                    continue
                control = np.vstack(ca_pts).astype(np.float32)
                if len(control) >= 4:
                    try:
                        # Slightly denser sampling for explicit windows
                        centerline = catmull_rom_chain(
                            control, samples_per_segment=max(10, samples_per_segment)
                        ).astype(np.float32)
                    except Exception:
                        centerline = control
                else:
                    centerline = control
                swept = sweep_tube(centerline)
                if swept is None:
                    continue
                vtx, nrm, idx = swept
                all_vertices.append(vtx)
                all_normals.append(nrm)
                all_indices.append(idx + vertex_offset)
                vertex_offset += vtx.shape[0]
                added_ranges.add(range_key)
                start_resno = getattr(seg_res[0], "residue_number", -1)
                end_resno = getattr(seg_res[-1], "residue_number", -1)
                added_segments_log.append(
                    (chain_id, "EXPL", start_resno, end_resno, len(seg_res))
                )

                # Also add a tighter, local segment centered on each overlapping coil run to ensure proximity
                for cs, ce, rn_s, rn_e in coil_runs:
                    if rn_s is None or rn_e is None:
                        continue
                    if not (rn_e < rstart or rn_s > rend):
                        cs2 = max(cs - 2, 0)
                        ce2 = min(ce + 2, len(res_list) - 1)
                        if ce2 <= cs2:
                            continue
                        seg_res2 = res_list[cs2 : ce2 + 1]
                        ca_pts2 = []
                        for res in seg_res2:
                            if hasattr(res, "ca") and res.ca is not None:
                                ca_pts2.append(np.asarray(res.ca, dtype=np.float32))
                            elif hasattr(res, "coords") and res.coords is not None:
                                ca_pts2.append(np.asarray(res.coords, dtype=np.float32))
                        if len(ca_pts2) < 2:
                            continue
                        control2 = np.vstack(ca_pts2).astype(np.float32)
                        if len(control2) >= 4:
                            try:
                                centerline2 = catmull_rom_chain(
                                    control2,
                                    samples_per_segment=max(10, samples_per_segment),
                                ).astype(np.float32)
                            except Exception:
                                centerline2 = control2
                        else:
                            centerline2 = control2
                        swept2 = sweep_tube(centerline2)
                        if swept2 is None:
                            continue
                        vtx2, nrm2, idx2 = swept2
                        all_vertices.append(vtx2)
                        all_normals.append(nrm2)
                        all_indices.append(idx2 + vertex_offset)
                        vertex_offset += vtx2.shape[0]
                        start_resno2 = getattr(seg_res2[0], "residue_number", -1)
                        end_resno2 = getattr(seg_res2[-1], "residue_number", -1)
                        added_segments_log.append(
                            (
                                chain_id,
                                "EXPL_LOCAL",
                                start_resno2,
                                end_resno2,
                                len(seg_res2),
                            )
                        )

        # Terminal short-coil bridging: start-of-chain coils to first non-coil
        if runs:
            first_ss, first_s, first_e = runs[0]
            if first_ss in [" ", "-", "T", "C"] and (first_e - first_s + 1) <= 6:
                # Find next non-coil run
                next_idx = None
                for r_j in range(1, len(runs)):
                    if runs[r_j][0] in ["H", "E"]:
                        next_idx = r_j
                        break
                if next_idx is not None:
                    _, next_start, _ = runs[next_idx]
                    start_idx = 0
                    end_idx = min(next_start + 2, len(res_list) - 1)
                    range_key = (start_idx, end_idx)
                    if range_key not in added_ranges and end_idx > start_idx:
                        seg_res = res_list[start_idx : end_idx + 1]
                        ca_pts = []
                        for res in seg_res:
                            if hasattr(res, "ca") and res.ca is not None:
                                ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                            elif hasattr(res, "coords") and res.coords is not None:
                                ca_pts.append(np.asarray(res.coords, dtype=np.float32))
                        if len(ca_pts) >= 2:
                            control = np.vstack(ca_pts).astype(np.float32)
                            if len(control) >= 4:
                                try:
                                    centerline = catmull_rom_chain(
                                        control, samples_per_segment=samples_per_segment
                                    ).astype(np.float32)
                                except Exception:
                                    centerline = control
                            else:
                                centerline = control
                            swept = sweep_tube(centerline)
                            if swept is not None:
                                vtx, nrm, idx = swept
                                all_vertices.append(vtx)
                                all_normals.append(nrm)
                                all_indices.append(idx + vertex_offset)
                                vertex_offset += vtx.shape[0]
                                added_ranges.add(range_key)
                                start_resno = getattr(seg_res[0], "residue_number", -1)
                                end_resno = getattr(seg_res[-1], "residue_number", -1)
                                added_segments_log.append(
                                    (
                                        chain_id,
                                        "TERM_START",
                                        start_resno,
                                        end_resno,
                                        len(seg_res),
                                    )
                                )

        # Terminal short-coil bridging: last non-coil to end-of-chain coils
        if runs:
            last_ss, last_s, last_e = runs[-1]
            if last_ss in [" ", "-", "T", "C"] and (last_e - last_s + 1) <= 6:
                # Find previous non-coil run
                prev_idx = None
                for r_j in range(len(runs) - 2, -1, -1):
                    if runs[r_j][0] in ["H", "E"]:
                        prev_idx = r_j
                        break
                if prev_idx is not None:
                    _, _, prev_end = runs[prev_idx]
                    start_idx = max(prev_end - 2, 0)
                    end_idx = len(res_list) - 1
                    range_key = (start_idx, end_idx)
                    if range_key not in added_ranges and end_idx > start_idx:
                        seg_res = res_list[start_idx : end_idx + 1]
                        ca_pts = []
                        for res in seg_res:
                            if hasattr(res, "ca") and res.ca is not None:
                                ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                            elif hasattr(res, "coords") and res.coords is not None:
                                ca_pts.append(np.asarray(res.coords, dtype=np.float32))
                        if len(ca_pts) >= 2:
                            control = np.vstack(ca_pts).astype(np.float32)
                            if len(control) >= 4:
                                try:
                                    centerline = catmull_rom_chain(
                                        control, samples_per_segment=samples_per_segment
                                    ).astype(np.float32)
                                except Exception:
                                    centerline = control
                            else:
                                centerline = control
                            swept = sweep_tube(centerline)
                            if swept is not None:
                                vtx, nrm, idx = swept
                                all_vertices.append(vtx)
                                all_normals.append(nrm)
                                all_indices.append(idx + vertex_offset)
                                vertex_offset += vtx.shape[0]
                                added_ranges.add(range_key)
                                start_resno = getattr(seg_res[0], "residue_number", -1)
                                end_resno = getattr(seg_res[-1], "residue_number", -1)
                                added_segments_log.append(
                                    (
                                        chain_id,
                                        "TERM_END",
                                        start_resno,
                                        end_resno,
                                        len(seg_res),
                                    )
                                )

        # Generate tubes for ALL coil regions (replacing CoilRegions)
        for r_idx in range(len(runs)):
            ss_curr, s0, e0 = runs[r_idx]
            if ss_curr not in [" ", "-", "T", "C"]:
                continue
            # Skip if already covered by bridging logic
            range_key = (s0, e0)
            if range_key in added_ranges:
                continue
            # Create tube for this coil run
            seg_res = res_list[s0 : e0 + 1]
            ca_pts = []
            for res in seg_res:
                if hasattr(res, "ca") and res.ca is not None:
                    ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                elif hasattr(res, "coords") and res.coords is not None:
                    ca_pts.append(np.asarray(res.coords, dtype=np.float32))
            if len(ca_pts) < 2:
                continue
            control = np.vstack(ca_pts).astype(np.float32)
            if len(control) >= 4:
                try:
                    centerline = catmull_rom_chain(
                        control, samples_per_segment=samples_per_segment
                    ).astype(np.float32)
                except Exception:
                    centerline = control
            else:
                centerline = control
            swept = sweep_tube(centerline)
            if swept is None:
                continue
            vtx, nrm, idx = swept
            all_vertices.append(vtx)
            all_normals.append(nrm)
            all_indices.append(idx + vertex_offset)
            vertex_offset += vtx.shape[0]
            added_ranges.add(range_key)
            start_resno = getattr(seg_res[0], "residue_number", -1)
            end_resno = getattr(seg_res[-1], "residue_number", -1)
            added_segments_log.append(
                (chain_id, "COIL", start_resno, end_resno, len(seg_res))
            )

    if len(all_vertices) == 0:
        secondary_struct.tube = MeshData()
        # Quiet coverage log
        try:
            log.message("Tube coverage: no segments generated", silent=False)
        except Exception:
            pass
        return

    vertices = np.vstack(all_vertices)
    normals = np.vstack(all_normals)
    indices = np.concatenate(all_indices)

    colors = np.tile(
        np.array([0.7, 0.7, 0.7], dtype=np.float32), (vertices.shape[0], 1)
    )

    secondary_struct.tube = MeshData.from_raw(
        vertices=vertices,
        normals=normals,
        colors=colors,
        indices=indices,
    )

    # Quiet summary logging per chain and kind
    try:
        if added_segments_log:
            # Aggregate counts
            summary = {}
            for chain_id, kind, s, e, n in added_segments_log:
                summary.setdefault(chain_id, {})
                summary[chain_id][kind] = summary[chain_id].get(kind, 0) + 1
            # Build compact message
            parts = []
            for cid, kinds in summary.items():
                kinds_str = ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items()))
                parts.append(f"{cid}[{kinds_str}]")
            msg = "Tube coverage: segments " + "; ".join(parts)
            log.message(msg, silent=False)
    except Exception:
        pass


def _trim_tube_segment(runs, i, max_extension=5):
    """Extend coil segments by 5 residues in either direction.

    Args:
        runs: List of (ss_type, start_idx, end_idx) tuples
        i: Index of current run
        max_extension: Number of residues to extend beyond coil boundaries

    Returns:
        (actual_start, actual_end) or None if segment should be skipped
    """
    ss_type, start_idx, end_idx = runs[i]

    # For actual secondary structure elements (helices/sheets), use the full range
    if ss_type not in [SecondaryStructureType.COIL, " "]:
        return start_idx, end_idx

    # For coils, extend by max_extension residues in either direction
    actual_start = start_idx
    actual_end = end_idx

    # Look for neighboring secondary structure elements
    prev_ss_type = None
    next_ss_type = None

    if i > 0:
        prev_ss_type = runs[i - 1][0]
    if i < len(runs) - 1:
        next_ss_type = runs[i + 1][0]

    # If this coil is between two secondary structure elements,
    # limit it to max_extension residues beyond each boundary
    is_prev_secondary = prev_ss_type in [
        SecondaryStructureType.ALPHA_HELIX,
        SecondaryStructureType.BETA_STRAND,
    ] or prev_ss_type in ["H", "E"]
    is_next_secondary = next_ss_type in [
        SecondaryStructureType.ALPHA_HELIX,
        SecondaryStructureType.BETA_STRAND,
    ] or next_ss_type in ["H", "E"]

    # Extend the coil by max_extension residues in either direction
    # Get the total number of residues in the chain
    total_residues = runs[-1][2] + 1 if runs else 0

    # Extend backwards (towards start of chain)
    actual_start = max(0, start_idx - max_extension)

    # Extend forwards (towards end of chain)
    actual_end = min(total_residues - 1, end_idx + max_extension)

    # Skip if the extended segment is too short
    if actual_end - actual_start < 1:
        return None

    return actual_start, actual_end


def _create_simple_backbone_tube_mesh(
    model,
    secondary_struct,
    radius: float = 0.35,
    ring_segments: int = 12,
    samples_per_segment: int = 8,
):
    """Generate simple tube segments for all secondary structure elements.

    This is a simplified version that creates tubes for all secondary structure
    elements (helices, sheets, coils) without complex transition logic.
    Tubes are limited to extend no more than 2 residues beyond actual secondary structure elements.
    """
    all_residues = list(model.get_all_residues())
    if not all_residues:
        secondary_struct.tube = MeshData()
        return

    # Group residues by chain while preserving order
    chains = {}
    for res in all_residues:
        chain_id = getattr(res, "chain_id", "A")
        chains.setdefault(chain_id, []).append(res)

    # Helper: build tube mesh arrays for one polyline centerline
    def sweep_tube(centerline: np.ndarray):
        num_points = len(centerline)
        if num_points < 2:
            return None

        # Tangents
        tangents = np.zeros_like(centerline)
        tangents[1:-1] = centerline[2:] - centerline[:-2]
        tangents[0] = centerline[1] - centerline[0]
        tangents[-1] = centerline[-1] - centerline[-2]
        lens = np.linalg.norm(tangents, axis=1)
        lens[lens == 0] = 1.0
        tangents = (tangents.T / lens).T

        def orthogonal(v):
            x, y, z = v
            if abs(x) < 0.9:
                base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                base = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            n = np.cross(v, base)
            n_len = np.linalg.norm(n)
            if n_len == 0:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
            return n / n_len

        normals = np.zeros_like(centerline)
        binormals = np.zeros_like(centerline)
        normals[0] = orthogonal(tangents[0])
        binormals[0] = np.cross(tangents[0], normals[0])
        b0_len = np.linalg.norm(binormals[0])
        binormals[0] = binormals[0] / (b0_len if b0_len != 0 else 1.0)

        for i in range(1, num_points):
            v = tangents[i - 1]
            w = tangents[i]
            axis = np.cross(v, w)
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-6:
                normals[i] = normals[i - 1]
                binormals[i] = binormals[i - 1]
                continue
            axis /= axis_len
            cos_theta = np.clip(np.dot(v, w), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            n_prev = normals[i - 1]
            k = axis
            n_rot = (
                n_prev * np.cos(theta)
                + np.cross(k, n_prev) * np.sin(theta)
                + k * np.dot(k, n_prev) * (1 - np.cos(theta))
            )
            normals[i] = n_rot / (np.linalg.norm(n_rot) + 1e-12)
            b = np.cross(tangents[i], normals[i])
            binormals[i] = b / (np.linalg.norm(b) + 1e-12)

        ring_angles = np.linspace(0.0, 2.0 * np.pi, ring_segments, endpoint=False)
        cos_a = np.cos(ring_angles)
        sin_a = np.sin(ring_angles)

        vertices = np.zeros((num_points * ring_segments, 3), dtype=np.float32)
        vnormals = np.zeros_like(vertices)

        for i in range(num_points):
            n = normals[i]
            b = binormals[i]
            center = centerline[i]
            for j in range(ring_segments):
                offset = n * (radius * cos_a[j]) + b * (radius * sin_a[j])
                idx = i * ring_segments + j
                vertices[idx] = center + offset
                vnormals[idx] = offset / (np.linalg.norm(offset) + 1e-12)

        indices = []
        for i in range(num_points - 1):
            ring0 = i * ring_segments
            ring1 = (i + 1) * ring_segments
            for j in range(ring_segments):
                j_next = (j + 1) % ring_segments
                a = ring0 + j
                b0 = ring0 + j_next
                c = ring1 + j
                d = ring1 + j_next
                indices.append([a, c, b0])
                indices.append([b0, c, d])

        if len(indices) == 0:
            return None
        return vertices, vnormals, np.asarray(indices, dtype=np.uint32).reshape(-1)

    # Create tubes for all secondary structure elements
    all_vertices = []
    all_normals = []
    all_indices = []
    vertex_offset = 0
    added_segments_log = []

    for chain_id, res_list in chains.items():
        if not res_list:
            continue

        # Build runs of same secondary structure type
        runs = []  # list of (ss_type, start_idx, end_idx)
        current_ss = getattr(res_list[0], "secstruc", " ")
        # Convert to SecondaryStructureType enum
        if isinstance(current_ss, str):
            current_ss = SecondaryStructureType.from_string(current_ss)
        elif not isinstance(current_ss, SecondaryStructureType):
            current_ss = SecondaryStructureType.COIL
        run_start = 0
        for i in range(1, len(res_list)):
            ss = getattr(res_list[i], "secstruc", " ")
            # Convert to SecondaryStructureType enum
            if isinstance(ss, str):
                ss = SecondaryStructureType.from_string(ss)
            elif not isinstance(ss, SecondaryStructureType):
                ss = SecondaryStructureType.COIL
            if ss != current_ss:
                runs.append((current_ss, run_start, i - 1))
                current_ss = ss
                run_start = i
        runs.append((current_ss, run_start, len(res_list) - 1))

        # Create tubes only for coil regions (intervening spaces between secondary structures)
        for i, (ss_type, start_idx, end_idx) in enumerate(runs):
            # Only generate tubes for coil regions (intervening spaces)
            if ss_type not in [SecondaryStructureType.COIL, " "]:
                continue  # Skip helices and sheets, only process coils
            # Skip very short runs
            if end_idx - start_idx < 1:
                continue

            # Trim the segment to limit extension beyond secondary structure
            trimmed_indices = _trim_tube_segment(runs, i, max_extension=3)
            if trimmed_indices is None:
                continue  # Skip this segment

            actual_start, actual_end = trimmed_indices

            # Extract residues for this run (using trimmed indices)
            seg_res = res_list[actual_start : actual_end + 1]

            # Extract CA coords
            ca_pts = []
            for res in seg_res:
                if hasattr(res, "ca") and res.ca is not None:
                    ca_pts.append(np.asarray(res.ca, dtype=np.float32))
                elif hasattr(res, "coords") and res.coords is not None:
                    ca_pts.append(np.asarray(res.coords, dtype=np.float32))

            if len(ca_pts) < 2:
                continue

            control = np.vstack(ca_pts).astype(np.float32)

            # Spline if enough points, else use raw
            if len(control) >= 4:
                try:
                    centerline = catmull_rom_chain(
                        control, samples_per_segment=samples_per_segment
                    ).astype(np.float32)
                except Exception:
                    centerline = control
            else:
                centerline = control

            swept = sweep_tube(centerline)
            if swept is None:
                continue

            vtx, nrm, idx = swept
            all_vertices.append(vtx)
            all_normals.append(nrm)
            # offset indices
            all_indices.append(idx + vertex_offset)
            vertex_offset += vtx.shape[0]

            # Log meta
            start_resno = getattr(seg_res[0], "residue_number", -1)
            end_resno = getattr(seg_res[-1], "residue_number", -1)
            # Convert enum to string for logging
            ss_type_str = (
                ss_type.to_string()
                if isinstance(ss_type, SecondaryStructureType)
                else str(ss_type)
            )
            added_segments_log.append(
                (chain_id, ss_type_str, start_resno, end_resno, len(seg_res))
            )

    if len(all_vertices) == 0:
        secondary_struct.tube = MeshData()
        return

    vertices = np.vstack(all_vertices)
    normals = np.vstack(all_normals)
    indices = np.concatenate(all_indices)

    colors = np.tile(
        np.array([0.7, 0.7, 0.7], dtype=np.float32), (vertices.shape[0], 1)
    )

    secondary_struct.tube = MeshData.from_raw(
        vertices=vertices,
        normals=normals,
        colors=colors,
        indices=indices,
    )

    # Log summary
    try:
        if added_segments_log:
            summary = {}
            for chain_id, kind, s, e, n in added_segments_log:
                summary.setdefault(chain_id, {})
                summary[chain_id][kind] = summary[chain_id].get(kind, 0) + 1
            parts = []
            for cid, kinds in summary.items():
                kinds_str = ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items()))
                parts.append(f"{cid}[{kinds_str}]")
            msg = "Simple tube coverage: segments " + "; ".join(parts)
            log.message(msg, silent=False)
    except Exception:
        pass
