# Created by gonzalezroy at 6/6/24
import time
from collections import defaultdict
from os.path import join
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
    selected = topology.select(BACKBONE_SELECTION)
    return sorted(selected.astype(int))

def prepare_datastructures(trajectories, topo, out_dir, heavies, first_timer):
    # Load trajectory
    trajs = trajectories.split()
    mini_traj = next(md.iterload(trajs[0], top=topo, chunk=1)) # first ensemble
    full_topo, bonds = mini_traj.topology.to_dataframe()
    full_topo_file = join(out_dir, 'full_topo_file.txt')
    with open(full_topo_file, 'w') as file:
        for index, row in full_topo.iterrows():
            file.write(
                f"serial {index + 1}, "
                f"name {row['name']}, "
                f"element {row['element']}, "
                f"resSeq {row['resSeq']}, "
                f"resName {row['resName']}\n"
            )
                  
    bonds_file = join(out_dir, 'bonds_file.txt')
    with open(bonds_file, 'w') as file:
        for bond in bonds:
            file.write(f"Bond: {bond[0]}, {bond[1]}\n")

    resids_to_atoms, resids_to_noh = get_resids_indices(mini_traj)
    atoms_to_resids = {y: x for x in resids_to_atoms for y in resids_to_atoms[x]}

    # Atom selections indices for descriptors calculation
    map_file = join(out_dir, 'mapping_file.txt')
    calphas = get_calpha_p_indices(mini_traj, atoms_to_resids, map_file)
    oxy, nitro = get_sb_indices(full_topo, atoms_to_resids) # salt bridges
    donors, hydros, acceptors = get_dha_indices(full_topo, bonds, heavies, atoms_to_resids) # hydrogen bonds

    _save_index_maps(out_dir, resids_to_atoms, resids_to_noh, atoms_to_resids, calphas, oxy, nitro, donors, hydros, acceptors)

    prep_time = round(time.time() - first_timer, 2)
    print(f" 📋 System details: number of trajectories are {len(trajs)}")
    print(f" 📋 System details: number of residues are {len(calphas)}")
    print(f" 📋 System details: number of atoms are {len(atoms_to_resids)}")
    print(f" ⏱️  Until datastructures prepared: {prep_time} s")

    return resids_to_atoms, resids_to_noh, calphas, oxy, nitro, donors, hydros, acceptors

def _save_index_maps(out_dir, resids_to_atoms, resids_to_noh, atoms_to_resids, calphas, oxy, nitro, donors, hydros, acceptors):
    """Write all residue/atom index maps to human-readable text files."""

    def _write_dict(path, header, data):
        with open(path, 'w') as f:
            f.write(header + "\n")
            for key in sorted(data.keys()):
                f.write(f"{key}: {data[key]}\n")

    _write_dict(join(out_dir, "resids_to_atoms.txt"),
                "# residue_index: [atom_indices]", resids_to_atoms)

    _write_dict(join(out_dir, "resids_to_noh.txt"),
                "# residue_index: [non-hydrogen atom_indices]", resids_to_noh)

    _write_dict(join(out_dir, "atoms_to_resids.txt"),
                "# atom_index: residue_index", atoms_to_resids)

    _write_dict(join(out_dir, "calphas.txt"),
                "# residue_index: backbone_atom_index", calphas)

    _write_dict(join(out_dir, "oxy_indices.txt"),
                "# residue_index: [oxygen atom_indices (salt bridges)]", dict(oxy))

    _write_dict(join(out_dir, "nitro_indices.txt"),
                "# residue_index: [nitrogen atom_indices (salt bridges)]", dict(nitro))

    _write_dict(join(out_dir, "donors_indices.txt"),
                "# residue_index: [donor atom_indices (H-bonds)]", dict(donors))

    _write_dict(join(out_dir, "hydros_indices.txt"),
                "# residue_index: [hydrogen atom_indices (H-bonds)]", dict(hydros))

    _write_dict(join(out_dir, "acceptors_indices.txt"),
                "# residue_index: [acceptor atom_indices (H-bonds)]", dict(acceptors))


def get_resids_indices(trajectory):
    backbone_atom_indices = select_backbone_atoms(trajectory.topology)
    backbone_res_indices = [trajectory.topology.atom(i).residue.index for i in backbone_atom_indices]

    res_ind_zero = {}
    res_ind_noh = {}
    for i, resid in enumerate(backbone_res_indices):
        residue = trajectory.topology.residue(resid)
        res_ind_zero[i] = [atom.index for atom in residue.atoms]
        res_ind_noh[i] = [atom.index for atom in residue.atoms if atom.element is None or atom.element.symbol != "H"]

    return res_ind_zero, res_ind_noh

def get_calpha_p_indices(trajectory, atoms_to_resids, map_file):
    # Maps each internal residue index to its CA/GC (protein) or C5'/C5X (nucleic)

    backbone_atom_indices = select_backbone_atoms(trajectory.topology)

    # Write atom details to the specified map_file
    with open(map_file, 'w') as file:
        for idx in backbone_atom_indices:
            atom = trajectory.topology.atom(int(idx))

            # Writing atom details to file
            file.write(f"Atom Index: {idx}, Atom Name: {atom.name}, "
                       f"Residue Name: {atom.residue.name}, Residue Index: {atom.residue.index}, "
                       f"Residue Number: {atom.residue}, chain id:{atom.residue.chain.chain_id}\n")

    # residue index -> backbone atom index (needed by frame_coords[calphas[i]])
    calphas_p = {int(atoms_to_resids[idx]): int(idx) for idx in backbone_atom_indices}

    return calphas_p

def get_sb_indices(topo_df, atoms_to_resids):
    # {residue index: [oxygen indices]}
    # {residue index: [nitrogen indices]}

    sel_acidic = topo_df.resName.isin(ACIDIC_RESNAMES)
    sel_basic = topo_df.resName.isin(BASIC_RESNAMES)
    sel_O = (topo_df.element == "O") | topo_df.name.isin(ACIDIC_BEAD_NAMES)
    sel_N = (topo_df.element == "N") | topo_df.name.isin(BASIC_BEAD_NAMES)

    o_indices = topo_df[sel_acidic & sel_O].index
    n_indices = topo_df[sel_basic & sel_N].index

    oxy = defaultdict(list)
    for x in o_indices:
        oxy[atoms_to_resids[int(x)]].append(int(x))

    nitro = defaultdict(list)
    for x in n_indices:
        nitro[atoms_to_resids[int(x)]].append(int(x))

    return oxy, nitro

def get_dha_indices(topo_df, bonds, heavies_elements, atoms_to_resids):
    # Get heavies and hydrogen indices

    all_hydrogens = set(topo_df[topo_df.element == "H"].index)
    a_indices = set(np.where(topo_df.element.isin(heavies_elements))[0])

    # Find D-H indices (requires bonded H; skipped for typical CG topologies)
    h_indices = []
    d_indices = []
    for values in bonds:
        at1 = int(values[0])
        at2 = int(values[1])
        if (at1 in all_hydrogens) and (at2 in a_indices):
            h_indices.append(at1)
            d_indices.append(at2)
        elif (at2 in all_hydrogens) and (at1 in a_indices):
            h_indices.append(at2)
            d_indices.append(at1)
        else:
            continue

    donors = defaultdict(list)
    for x in d_indices:
        donors[atoms_to_resids[int(x)]].append(int(x))

    hydros = defaultdict(list)
    for x in h_indices:
        hydros[atoms_to_resids[int(x)]].append(int(x))

    acceptors = defaultdict(list)
    for x in a_indices:
        acceptors[atoms_to_resids[int(x)]].append(int(x))

    return donors, hydros, acceptors
