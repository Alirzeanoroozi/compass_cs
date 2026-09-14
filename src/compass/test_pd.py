from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np


N_FRAMES = 20
RANDOM_SEED = 42
BACKBONE_SELECTION = 'name CA or name GC or name C5X or name "C5\'"'


def _pair_distances_to_matrices(
    pair_distances, row, column, n_backbone_sites
):
    distance_matrices = np.zeros(
        (len(pair_distances), n_backbone_sites, n_backbone_sites),
        dtype=pair_distances.dtype,
    )
    distance_matrices[:, row, column] = pair_distances
    distance_matrices[:, column, row] = pair_distances
    return distance_matrices


def calculate_random_backbone_distances(
    pdb_file, topology, n_frames=N_FRAMES, random_seed=RANDOM_SEED
):
    """Sample frames uniformly and return their backbone distances."""
    rng = np.random.default_rng(random_seed)
    sampled_pair_distances = None
    sampled_frame_indices = np.empty(n_frames, dtype=int)
    first_frame_pair_distances = None
    row = column = None
    n_backbone_sites = 0
    n_seen = 0

    for trajectory in md.iterload(str(pdb_file), top=topology, chunk=20):
        if sampled_pair_distances is None:
            backbone_indices = trajectory.topology.select(
                BACKBONE_SELECTION
            )
            n_backbone_sites = len(backbone_indices)
            if n_backbone_sites < 2:
                raise ValueError(
                    f"{pdb_file} must contain at least two representative "
                    "backbone sites (CA, GC, C5X, or C5')."
                )

            row, column = np.triu_indices(n_backbone_sites, k=1)
            atom_pairs = np.column_stack(
                (backbone_indices[row], backbone_indices[column])
            )
            sampled_pair_distances = np.empty(
                (n_frames, len(atom_pairs)), dtype=np.float32
            )

        chunk_pair_distances = (
            md.compute_distances(trajectory, atom_pairs) * 10.0
        )
        if first_frame_pair_distances is None:
            first_frame_pair_distances = chunk_pair_distances[0].copy()

        for pair_distances in chunk_pair_distances:
            if n_seen < n_frames:
                sample_slot = n_seen
            else:
                sample_slot = rng.integers(0, n_seen + 1)

            if sample_slot < n_frames:
                sampled_pair_distances[sample_slot] = pair_distances
                sampled_frame_indices[sample_slot] = n_seen
            n_seen += 1

    if n_seen < n_frames:
        raise ValueError(
            f"{pdb_file} contains {n_seen} frame(s); "
            f"at least {n_frames} are required."
        )

    order = np.argsort(sampled_frame_indices)
    sampled_pair_distances = sampled_pair_distances[order]
    sampled_frame_numbers = sampled_frame_indices[order] + 1

    distance_matrices = _pair_distances_to_matrices(
        sampled_pair_distances, row, column, n_backbone_sites
    )
    first_frame_matrix = _pair_distances_to_matrices(
        first_frame_pair_distances[None, :],
        row,
        column,
        n_backbone_sites,
    )[0]
    return distance_matrices, sampled_frame_numbers, first_frame_matrix


def plot_distance_matrices(
    distance_matrices,
    frame_numbers,
    output_file,
    reference_matrix=None,
):
    """Plot exact distances or signed differences from a reference."""
    if reference_matrix is None:
        plotted_matrices = distance_matrices
        cmap = "viridis"
        vmin = 0
        vmax = np.max(plotted_matrices)
        colorbar_label = r"Distance ($\AA$)"
    else:
        plotted_matrices = distance_matrices - reference_matrix
        cmap = "coolwarm"
        color_limit = np.max(np.abs(plotted_matrices))
        if color_limit == 0:
            color_limit = 1.0
        vmin = -color_limit
        vmax = color_limit
        colorbar_label = r"Distance difference from frame 1 ($\AA$)"

    n_frames = len(plotted_matrices)
    n_columns = min(5, n_frames)
    n_rows = int(np.ceil(n_frames / n_columns))

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4 * n_columns, 4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    image = None
    for frame_index, axis in enumerate(axes.flat):
        if frame_index >= n_frames:
            axis.set_visible(False)
            continue
        image = axis.imshow(
            plotted_matrices[frame_index],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        frame_number = frame_numbers[frame_index]
        if reference_matrix is None:
            axis.set_title(f"Frame {frame_number}")
        else:
            axis.set_title(
                f"Frame {frame_number} \N{MINUS SIGN} Frame 1"
            )
        axis.set_xlabel("Backbone-site index")
        axis.set_ylabel("Backbone-site index")

    figure.colorbar(
        image,
        ax=[axis for axis in axes.flat if axis.get_visible()],
        label=colorbar_label,
        shrink=0.9,
    )
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def main():
    parser = ArgumentParser(
        description=(
            "Plot exact and frame-1-relative backbone distances for 20 "
            "random frames."
        )
    )
    parser.add_argument(
        "pdb_file",
        nargs="?",
        type=Path,
        default=Path("test.pdb"),
    )
    parser.add_argument(
        "topology",
        nargs="?",
        type=Path,
        default=Path("test.pdb"),
        help="Topology file (default: test.pdb).",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--distances-output",
        dest="distances_output",
        type=Path,
        default=Path(
            "backbone_pairwise_distances_20_random_frames.png"
        ),
    )
    parser.add_argument(
        "--differences-output",
        type=Path,
        default=Path(
            "backbone_pairwise_distance_differences_from_frame_1.png"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used to select frames (default: 42).",
    )
    args = parser.parse_args()

    distance_matrices, frame_numbers, first_frame_matrix = (
        calculate_random_backbone_distances(
            args.pdb_file, args.topology, random_seed=args.seed
        )
    )
    plot_distance_matrices(
        distance_matrices, frame_numbers, args.distances_output
    )
    plot_distance_matrices(
        distance_matrices,
        frame_numbers,
        args.differences_output,
        reference_matrix=first_frame_matrix,
    )
    print(
        f"Selected frames: {', '.join(map(str, frame_numbers))}\n"
        f"Saved exact distances to {args.distances_output}\n"
        f"Saved frame-1 differences to {args.differences_output}"
    )


if __name__ == "__main__":
    main()

