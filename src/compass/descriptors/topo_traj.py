# Created by gonzalezroy at 6/6/24
"""Manage common operations as well as those related to topo and traj files"""
import time
from collections import defaultdict
from os.path import basename, join
import mdtraj as md
import numpy as np
from numba import njit, types
from numba.typed.typeddict import Dict


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
    # Produce mapping files
    # mapping = Mapping(arg.out_dir)
    # map_file, renumbered_pdb = mapping.pre_processing(arg.topo)
    # mapping.post_processing(map_file, renumbered_pdb)

    # Load trajectory
    trajs = arg.traj.split()
    mini_traj = next(md.iterload(trajs[0], top=arg.topo, chunk=1))
    full_topo = mini_traj.topology.to_dataframe()[0]
    map_file = join(arg.out_dir, 'mapping_file.txt')
    # Produce mapping files
    # remap_toppology(arg.topo, mini_traj, arg.out_dir)

    # Indices of residues in the load trajectory and equivalence
    resids_to_atoms, resids_to_noh, internal_equiv = get_resids_indices(mini_traj)
    # print(resids_to_atoms, "resids_to_atoms in topo_traj")
    raw = {y: x for x in resids_to_atoms for y in resids_to_atoms[x]}
    atoms_to_resids = pydict_to_numbadict(raw)

    # Atom selections indices for descriptors calculation
    calphas = get_calpha_p_indices(mini_traj, atoms_to_resids, map_file=map_file)
    oxy, nitro = get_sb_indices(full_topo, atoms_to_resids)
    donors, hydros, acceptors = get_dha_indices(mini_traj, arg.heavies, atoms_to_resids)
    # Correlation uses backbone atom coordinates in internal residue order
    corr_indices = [int(calphas[i]) for i in range(len(calphas))]

    prep_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of trajectories are {len(trajs)}")
    print(f" 📋 System details: number of residues are {len(calphas)}")
    print(f" ⏱️  Until datastructures prepared: {prep_time} s")

    return mini_traj, trajs, resids_to_atoms, resids_to_noh, calphas, oxy, nitro, donors, hydros, acceptors, corr_indices

def get_xyz_chunks(trajs, topo, chunk_size=500):
    """
    Load chunks of xyz coordinates from a list of trajectories

    Args:
        trajs: list of trajectories
        topo: system topology
        chunk_size: size of the chunk to load

    Returns:
        chunk.xyz: chunk of xyz coordinates
    """
    for traj in trajs:
        chunks = md.iterload(traj, top=topo, chunk=chunk_size)
        for chunk in chunks:
            yield chunk.xyz


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
    babel_dict = {}
    for i, resid in enumerate(backbone_res_indices):
        residue = trajectory.topology.residue(resid)
        atom_ids = np.asarray([atom.index for atom in residue.atoms], dtype=np.int32)
        noh_ids = np.asarray(
            [
                atom.index
                for atom in residue.atoms
                if atom.element is None or atom.element.symbol != "H"
            ],
            dtype=np.int32,
        )
        res_ind_zero[i] = atom_ids
        res_ind_noh[i] = noh_ids
        babel_dict[i] = (
            residue.chain.chain_id if residue.chain.chain_id is not None else "",
            residue.resSeq,
            residue.segment_id if residue.segment_id is not None else "",
        )

    res_ind_numba = pydict_to_numbadict(res_ind_zero)
    res_ind_noh_numba = pydict_to_numbadict(res_ind_noh)
    return res_ind_numba, res_ind_noh_numba, babel_dict

def get_corr_indices(trajectory, map_file):
    """
    Get atomic indices for correlation calculation

    Args:
        trajectory: trajectory loaded in mdtraj format

    Returns:
        all_atoms: indices of all atoms to be considered for correlation
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
    return all_atoms


def get_calpha_p_indices(trajectory, atoms_to_resids, map_file, numba=True):
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
    calphas_p_raw = get_corr_indices(trajectory, map_file)
    # residue index -> backbone atom index (needed by frame_coords[calphas[i]])
    calphas_p = {int(atoms_to_resids[idx]): int(idx) for idx in calphas_p_raw}

    if len(calphas_p) != len(calphas_p_raw):
        raise ValueError("\nThe number of calphas + P atoms is different from the number of residues")

    if numba:
        alphas = pydict_to_numbadict(calphas_p)
    else:
        alphas = calphas_p
    return alphas


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
    oxy_raw3 = {x: np.asarray(oxy_raw1[x], dtype=np.int32) for x in oxy_raw1}
    oxy = pydict_to_numbadict(oxy_raw3)

    # Process the nitrogen indices to a numba dict
    nitro_raw1 = defaultdict(list)
    for x in n_indices:
        x = int(x)
        if x in atoms_to_resids:
            nitro_raw1[atoms_to_resids[x]].append(x)
    nitro_raw3 = {x: np.asarray(nitro_raw1[x], dtype=np.int32) for x in nitro_raw1}
    nitro = pydict_to_numbadict(nitro_raw3)
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
    d_raw3 = {x: np.asarray(d_raw[x], dtype=np.int32) for x in d_raw}
    donors = pydict_to_numbadict(d_raw3)

    # Process the indices of hydrogens to a numba dict
    h_raw = defaultdict(list)
    for x in h_raw1:
        x = int(x)
        if x in atoms_to_resids:
            h_raw[atoms_to_resids[x]].append(x)
    h_raw3 = {x: np.asarray(h_raw[x], dtype=np.int32) for x in h_raw}
    hydros = pydict_to_numbadict(h_raw3)

    # Process the indices of acceptors to a numba dict
    a_raw = defaultdict(list)
    for x in a_raw1:
        x = int(x)
        if x in atoms_to_resids:
            a_raw[atoms_to_resids[x]].append(x)
    a_raw3 = {x: np.asarray(a_raw[x], dtype=np.int32) for x in a_raw}
    acceptors = pydict_to_numbadict(a_raw3)
    return donors, hydros, acceptors

@njit(parallel=False)
def dict_get(dico, key):
    """
    Get the value of a key in a dictionary or return None if the key is not in
    the dictionary

    Args:
        dico: dictionary to search
        key: key to search

    Returns:
        value: value of the key in the dictionary or None if the key is not in
               the dictionary
    """
    try:
        value = dico[key]
        return value
    except:
        return None

def pydict_to_numbadict(py_dict):
    """
    Converts from Python dict to Numba dict.

    Empty dicts are created with an explicit type so Numba can compile
    njitted callers (e.g. CG runs with no donor/hydrogen sites).
    Value type is inferred from the first entry when the dict is non-empty.
    """
    if not py_dict:
        # Default used by oxy/nitro/donors/hydros/acceptors/resids maps
        return Dict.empty(key_type=types.int64, value_type=types.int32[:])

    first_val = next(iter(py_dict.values()))
    if isinstance(first_val, np.ndarray):
        numba_dict = Dict.empty(key_type=types.int64, value_type=types.int32[:])
    else:
        numba_dict = Dict.empty(key_type=types.int64, value_type=types.int64)

    for key, value in py_dict.items():
        numba_dict[int(key)] = value
    return numba_dict

def to_matrix(one_dim_array, n, nested=False):
    """
    Converts a one-dimensional array of size = N * (N-1) / 2, into the
    equivalent N * N matrix

    Args:
        one_dim_array: one-dimensional array
        n: number of col / row in the square matrix

    Returns:
        matrix: N * N symmetrycal matrix
    """
    matrix = np.zeros((n, n))
    k = 0

    if nested:
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = one_dim_array[k][0]
                k += 1
    else:
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = one_dim_array[k]
                k += 1

    matrix += matrix.T
    return matrix

class Mapping:
    # todo: simiplify with prody or another pdb parser as explicit line
    #  handling in PDB can be tricky even if standard format exists
    """
    @Sneha's class for residue renumbering and mapping operations on PDB files.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir

    def pre_processing(self, input_pdb):
        # todo: handle topology formats others than PDB
        """
        Renumbers the residues in a PDB file sequentially from 1, changing all chain identifiers to 'A'.
        Outputs a renumbered PDB file and a map file that records the original and new residue numbers and chains.

        Parameters:
        input_pdb (str): Path to the input PDB file.
        """
        # Define file paths for the renumbered PDB file and the map file
        renumbered_pdb_raw = input_pdb.replace(".pdb", "_renumbered.pdb")
        renumbered_pdb = join(self.out_dir, basename(renumbered_pdb_raw))
        map_file_raw = input_pdb.replace(".pdb", "_map.txt")
        map_file = join(self.out_dir, basename(map_file_raw))

        # Open input PDB file for reading, renumbered PDB file and map file for writing
        with open(input_pdb, "r") as infile, open(renumbered_pdb, "w") as outfile, open(map_file, "w") as mapfile:
            # Initialize variables for residue renumbering and mapping
            current_residue_number = 0
            residue_map = {}
            last_residue_id = None

            # Loop through each line in the input PDB file
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    # Extract chain ID, old residue number, and residue name
                    chain_id = line[21]
                    old_residue_number = line[22:26].strip()
                    res_name = line[17:20].strip()
                    residue_id = (chain_id, old_residue_number, res_name)

                    # Check if it's a new residue
                    if residue_id != last_residue_id:
                        current_residue_number += 1
                        last_residue_id = residue_id
                        residue_map[(chain_id, old_residue_number)] = (
                            current_residue_number,
                            "A",
                        )

                    # Write renumbered line with new chain ID 'A'
                    new_line = (
                            line[:21]
                            + "A"
                            + str(current_residue_number).rjust(4)
                            + line[26:]
                    )
                    outfile.write(new_line)
                else:
                    # Write non-ATOM/HETATM lines as they are
                    outfile.write(line)

            # Write the residue map to the map file
            for (chain_id, old_number), (
                    new_number, new_chain) in residue_map.items():
                mapfile.write(
                    f"{chain_id} {old_number} {new_chain} {new_number}\n")

        # Print confirmation messages
        # print(f"Renumbered PDB file saved as: {renumbered_pdb}")
        # print(f"Residue mapping file saved as: {map_file}")
        return map_file, renumbered_pdb

    def post_processing(self, map_file, renumbered_pdb):
        """
        Restores the original residue numbering and chain identifiers in a renumbered PDB file using the map file.

        Parameters:
        map_file (str): Path to the map file containing the original and new residue numbers and chains.
        renumbered_pdb (str): Path to the renumbered PDB file.
        """
        # Define file path for the restored PDB file
        original_pdb = renumbered_pdb.replace("_renumbered.pdb",
                                              "_restored.pdb")

        # Initialize a dictionary to store the residue mapping information
        residue_map = {}

        # Read the map file and populate the residue mapping dictionary
        with open(map_file, "r") as mapfile:
            for line in mapfile:
                original_chain, old_number, new_chain, new_number = line.split()
                residue_map[(new_chain, new_number)] = (
                    original_chain, old_number)

        # Open renumbered PDB file for reading and restored PDB file for writing
        with open(renumbered_pdb, "r") as infile, open(original_pdb,
                                                       "w") as outfile:
            # Loop through each line in the renumbered PDB file
            for line in infile:
                if line.startswith(("ATOM", "HETATM")):
                    # Extract new chain ID and new residue number
                    new_chain = line[21]
                    new_number = line[22:26].strip()
                    original_chain, old_number = residue_map[
                        (new_chain, new_number)]

                    # Write restored line with original chain ID and residue number
                    new_line = (
                            line[:21] + original_chain + old_number.rjust(
                        4) + line[26:]
                    )
                    outfile.write(new_line)
                else:
                    # Write non-ATOM/HETATM lines as they are
                    outfile.write(line)

        # Print confirmation message
        # print(f"Restored PDB file saved as: {original_pdb}")

# =============================================================================
#
# =============================================================================
# import mdtraj as md
# import prody as prd
#
# load topology and trajectory
# topo = '/home/rglez/RoyHub/compass/data/MDs/nucleosome_full_2c/1kx5_dry.pdb'
# traj = '/home/rglez/RoyHub/compass/data/MDs/nucleosome_full_2c/nuc-prot-trim.dcd'
# out_dir = '/home/rglez/RoyHub/compass/data/outputs/nucleosome_full_2c'
