# Created by gonzalezroy at 6/6/24
"""Manage common operations as well as those related to topo and traj files"""
import time
from collections import defaultdict
from os.path import basename, join
import mdtraj as md
import numpy as np

# Atomistic + SIRAH CG backbone beads (one representative site per residue)
BACKBONE_SELECTION = 'name CA or name GC or name C5X or name "C5\'"'

# Salt-bridge residue / bead names (atomistic + SIRAH)
ACIDIC_RESNAMES = {"ASP", "GLU", "sD", "sE"}
BASIC_RESNAMES = {"ARG", "HIS", "LYS", "HSP", "HSD", "HSE", "sR", "sK", "sHe"}
ACIDIC_BEAD_NAMES = {"BOE1", "BOE2", "BOD", "OD1", "OD2", "OE1", "OE2"}
BASIC_BEAD_NAMES = {"BNN1", "BNN2", "BND", "BNE", "BCE", "NZ", "NH1", "NH2", "NE", "ND1", "NE2"}

def select_backbone_atoms(topology):
    """
    Select one backbone atom/bead per biopolymer residue.

    Supports atomistic proteins/nucleic acids (CA, C5') and SIRAH CG (GC, C5X).
    """
    selected = topology.select(BACKBONE_SELECTION)
    return np.asarray(sorted(selected.astype(int)), dtype=np.int32)


def prepare_datastructures(arg, first_timer):
    """
    Prepare datastructures for the calculation of descriptors

    Args:
        arg: namespace with the arguments
        first_timer: first timer to measure the time

    Returns:
        mini_traj: the first chunk of the trajectory
        trajs: list of trajectories
        resids_to_atoms: numba Dict of each residue's indices
        resids_to_noh: numba Dict of each residue's indices without hydrogens
        calphas: numba Dict of each residue's C-alpha indices
        oxy: numba Dict of each residue's oxygen indices
        nitro: numba Dict of each residue's nitrogen indices
        donors: numba Dict of each residue's donor indices
        hydros: numba Dict of each residue's hydrogen indices
        acceptors: numba Dict of each residue's acceptor indices
        corr_indices: list of indices of atoms to be considered for correlation
    """
    # Load trajectory
    trajs = arg.traj.split()
    mini_traj = next(md.iterload(trajs[0], top=arg.topo, chunk=1)) # first ensemble
    full_topo = mini_traj.topology.to_dataframe()[0]
    map_file = join(arg.out_dir, 'mapping_file.txt')

    # Indices of residues in the load trajectory and equivalence
    resids_to_atoms, resids_to_noh = get_resids_indices(mini_traj)
    # print(resids_to_atoms, "resids_to_atoms in topo_traj")
    atoms_to_resids = {y: x for x in resids_to_atoms for y in resids_to_atoms[x]}

    # Atom selections indices for descriptors calculation
    calphas = get_calpha_p_indices(mini_traj, atoms_to_resids, map_file=map_file)
    oxy, nitro = get_sb_indices(full_topo, atoms_to_resids) # salt bridges
    donors, hydros, acceptors = get_dha_indices(mini_traj, arg.heavies, atoms_to_resids) # hydrogen bonds
    # Correlation uses backbone atom coordinates in internal residue order
    corr_indices = [int(calphas[i]) for i in range(len(calphas))]

    prep_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of trajectories are {len(trajs)}")
    print(f" 📋 System details: number of residues are {len(calphas)}")
    print(f" ⏱️  Until datastructures prepared: {prep_time} s")

    return mini_traj, trajs, resids_to_atoms, resids_to_noh, calphas, oxy, nitro, donors, hydros, acceptors, corr_indices



def get_resids_indices(trajectory):
    """
    Get indices of residues in the loaded trajectory, properly handling protein and nucleic acid residues
    while excluding ions and other non-residue atoms.

    Args:
        trajectory: trajectory loaded in mdtraj format
    Returns:
        res_ind_numba: numba Dict of each residue's all atoms indices
        res_ind_noh_numba: numba Dict of each residue's non-hydrogen atom indices
        babel_dict: the equivalence between the original resid numbering and
                   the 0-based numbering used internally
    """
    backbone = select_backbone_atoms(trajectory.topology)
    if len(backbone) == 0:
        raise ValueError(
            "No backbone atoms/beads found. Expected CA/C5' (atomistic) or "
            "GC/C5X (SIRAH CG)."
        )

    # Identify biopolymer residues via backbone sites. Use MDTraj residue.index
    # (not chainID/resSeq): DNA/protein systems often reuse residue numbers
    # across chains, which collapses residues in a PDB-field groupby.
    backbone_res_indices = []
    seen = set()
    for atom_idx in backbone:
        resid = trajectory.topology.atom(int(atom_idx)).residue.index
        if resid in seen:
            raise ValueError(
                f"Residue {resid} has more than one backbone atom/bead; "
                "cannot build a 1:1 residue mapping."
            )
        seen.add(resid)
        backbone_res_indices.append(resid)

    res_ind_zero = {}
    res_ind_noh = {}
    for i, resid in enumerate(backbone_res_indices):
        residue = trajectory.topology.residue(resid)
        atom_ids = np.asarray([atom.index for atom in residue.atoms], dtype=np.int32)
        noh_ids = np.asarray([atom.index for atom in residue.atoms if atom.element is None or atom.element.symbol != "H"], dtype=np.int32)
        res_ind_zero[i] = atom_ids
        res_ind_noh[i] = noh_ids

    return res_ind_zero, res_ind_noh


def get_calpha_p_indices(trajectory, atoms_to_resids, map_file):
    """
    Get atomic indices for backbone representative atoms/beads.

    Maps each internal residue index to its CA/GC (protein) or C5'/C5X (nucleic)
    atom index for distance and correlation descriptors.

    Args:
        trajectory: trajectory loaded in mdtraj format
        atoms_to_resids: dict mapping atoms indices to residues indices
        numba: whether to return a numba dict or a regular dict

    Returns:
        alphas: dict residue_index -> backbone atom index
    """
    all_atoms = select_backbone_atoms(trajectory.topology)

    # Write atom details to the specified map_file
    with open(map_file, 'w') as file:
        for idx in all_atoms:
            atom = trajectory.topology.atom(int(idx))

            # Writing atom details to file
            file.write(f"Atom Index: {idx}, Atom Name: {atom.name}, "
                       f"Residue Name: {atom.residue.name}, Residue Index: {atom.residue.index}, "
                       f"Residue Number: {atom.residue}, chain id:{atom.residue.chain.chain_id}\n")

    # residue index -> backbone atom index (needed by frame_coords[calphas[i]])
    calphas_p = {int(atoms_to_resids[idx]): int(idx) for idx in all_atoms}

    return calphas_p

def get_sb_indices(topo_df, atoms_to_resids):
    """
    Get atomic indices for salt bridges calculation

    Args:
        topo_df: topology dataframe as returned by MDTraj
        atoms_to_resids: dict mapping atoms indices to residues indices

    Returns:
        o_indices: indices of selected oxygen atoms (see VMD definitions)
        n_indices: indices of selected nitrogen atoms (see VMD definitions)

    """
    # Atomistic: charged residues + element O/N
    # SIRAH CG: sD/sE + BOE*, sR/sHe + BN*, sK + BCE (NZ bead)
    sel_acidic = topo_df.resName.isin(ACIDIC_RESNAMES)
    sel_basic = topo_df.resName.isin(BASIC_RESNAMES)
    sel_O = (topo_df.element == "O") | topo_df.name.isin(ACIDIC_BEAD_NAMES)
    sel_N = (topo_df.element == "N") | topo_df.name.isin(BASIC_BEAD_NAMES)

    o_indices = np.array(topo_df[sel_acidic & sel_O].index)
    n_indices = np.array(topo_df[sel_basic & sel_N].index)

    # Process the oxygen indices to a numba dict
    oxy_raw1 = defaultdict(list)
    for x in o_indices:
        x = int(x)
        if x in atoms_to_resids:
            oxy_raw1[atoms_to_resids[x]].append(x)
    oxy = {x: np.asarray(oxy_raw1[x], dtype=np.int32) for x in oxy_raw1}

    # Process the nitrogen indices to a numba dict
    nitro_raw1 = defaultdict(list)
    for x in n_indices:
        x = int(x)
        if x in atoms_to_resids:
            nitro_raw1[atoms_to_resids[x]].append(x)
    nitro = {x: np.asarray(nitro_raw1[x], dtype=np.int32) for x in nitro_raw1}
    return oxy, nitro

def get_dha_indices(trajectory, heavies_elements, atoms_to_resids):
    """
    Get 0-based indices of donors, hydrogens, and acceptors in an MDTraj traj

    Args:
        trajectory: MDTraj trajectory object
        heavies_elements: name of elements considered as heavies
        atoms_to_resids: dict mapping atoms indices to residues indices

    Returns:
        donors: indices of donor atoms (N or O bonded to H)
        hydros: indices of hydrogen atoms (H bonded to N or O)
        heavies: indices of heavy atoms (N or O)

    Note:
        SIRAH CG trajectories have no explicit hydrogens and usually no bonds.
        In that case empty donor/hydrogen maps are returned and HB occupancy
        stays zero unless an atomistic topology is used.
    """
    # Get heavies and hydrogen indices
    df, bonds = trajectory.topology.to_dataframe()
    all_hydrogens = set(df[df.element == "H"].index)

    a_raw1 = set(np.where(df.element.isin(heavies_elements))[0])

    # Find D-H indices (requires bonded H; skipped for typical CG topologies)
    h_raw1 = []
    d_raw1 = []
    for values in bonds:
        at1 = int(values[0])
        at2 = int(values[1])
        if (at1 in all_hydrogens) and (at2 in a_raw1):
            h_raw1.append(at1)
            d_raw1.append(at2)
        elif (at2 in all_hydrogens) and (at1 in a_raw1):
            h_raw1.append(at2)
            d_raw1.append(at1)
        else:
            continue

    # Process the indices of donors to a numba dict
    d_raw = defaultdict(list)
    for x in d_raw1:
        x = int(x)
        if x in atoms_to_resids:
            d_raw[atoms_to_resids[x]].append(x)
    donors = {x: np.asarray(d_raw[x], dtype=np.int32) for x in d_raw}

    # Process the indices of hydrogens to a numba dict
    h_raw = defaultdict(list)
    for x in h_raw1:
        x = int(x)
        if x in atoms_to_resids:
            h_raw[atoms_to_resids[x]].append(x)
    hydros = {x: np.asarray(h_raw[x], dtype=np.int32) for x in h_raw}

    # Process the indices of acceptors to a numba dict
    a_raw = defaultdict(list)
    for x in a_raw1:
        x = int(x)
        if x in atoms_to_resids:
            a_raw[atoms_to_resids[x]].append(x)
    acceptors = {x: np.asarray(a_raw[x], dtype=np.int32) for x in a_raw}
    return donors, hydros, acceptors
