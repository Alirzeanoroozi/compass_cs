import configparser
import os
from argparse import Namespace
from os.path import dirname, join, normpath

allowed_params = {
    "generals": {"topology", "trajectory", "output_dir", "job_name"},
    "non_bond": {"non_bond_cut"},
    "salt_bridges": {"NO_cut"},
    "hbonds": {"DA_cut", "HA_cut", "DHA_cut", "heavy"},
    "distance cutoffs": {"Graph", "Cliques"},
    # "paths": {"find_path", "sources", "targets"}
}

# Optional keys may be omitted. n_frames: first N frames across all trajectories (0 = all).
optional_params = {
    "generals": {"n_frames"},
}

allowed_heavies = {"S", "N", "O"}

def _parse_n_frames(raw):
    if raw is None:
        return None
    value = str(raw).strip()
    if value in ("", "0", "all", "None", "none"):
        return None
    try:
        n_frames = int(value)
    except ValueError as exc:
        raise ValueError(
            f"n_frames must be a non-negative integer or 0/all (use all frames), got {raw!r}"
        ) from exc
    if n_frames < 0:
        raise ValueError("n_frames must be >= 0 (0 means use all frames)")
    return n_frames

def read_config_file(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config_obj = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes="#")
    config_obj.optionxform = str
    config_obj.read(config_path)
    return config_obj

def check_config(config_obj):
    read_sections = set(config_obj.sections())
    allowed_sections = set(allowed_params.keys())
    sections_equals = read_sections == allowed_sections

    # Check sections
    if not sections_equals:
        print(sections_equals)
        raise ValueError(
            f"\nIncongruence in the number or naming of declared"
            f" sections. Only the following are supported: {allowed_sections}"
        )

    # Check keys
    config_dict = {}
    for section in allowed_params:
        allowed_keys = allowed_params[section]
        optional_keys = optional_params.get(section, set())
        read_keys = set(config_obj[section])
        missing = allowed_keys - read_keys
        unknown = read_keys - allowed_keys - optional_keys
        if missing or unknown:
            extra = f" Optional keys: {optional_keys}." if optional_keys else ""
            raise ValueError(
                f"\nIncongruence in the number or naming of"
                f" declared keys. Only the following are "
                f"supported for section [{section}]: {allowed_keys}.{extra}"
            )

        # Update config
        section_items = dict(config_obj[section].items())
        config_dict.update({section: section_items})

    return config_dict

def parse_params(config_path):
    config_obj = read_config_file(config_path)
    param_dict = check_config(config_obj)
    param_space = Namespace()
    root_dir = dirname(config_path)

    # General params
    param_space.out_dir = normpath(join(root_dir, param_dict["generals"]["output_dir"]))
    param_space.topo = normpath(join(root_dir, param_dict["generals"]["topology"]))
    param_space.title = param_dict["generals"]["job_name"]
    param_space.n_frames = _parse_n_frames(param_dict["generals"].get("n_frames"))

    # Descriptor params
    param_space.nb_cut = float(param_dict["non_bond"]["non_bond_cut"])
    param_space.sb_cut = float(param_dict["salt_bridges"]["NO_cut"])
    param_space.da_cut = float(param_dict["hbonds"]["DA_cut"])
    param_space.ha_cut = float(param_dict["hbonds"]["HA_cut"])
    param_space.dha_cut = float(param_dict["hbonds"]["DHA_cut"])
    param_space.heavies = set(param_dict["hbonds"]["heavy"].split())

    # Distance cutoffs
    param_space.dist_graph = float(param_dict["distance cutoffs"]["Graph"])
    param_space.dist_clique = float(param_dict["distance cutoffs"]["Cliques"])

    # # alternative paths between residues
    # param_space.find_path = str(param_dict["paths"]["find_path"])
    # param_space.source_residue = str(param_dict["paths"]["sources"])
    # param_space.target_residue = str(param_dict["paths"]["targets"])

    if not allowed_heavies.issuperset(param_space.heavies):
        raise ValueError(
            f"The computing of hbonds consider only "
            f"{allowed_heavies} as heavy atoms (donor or acceptor)."
        )

    # Check path existence
    if not os.path.exists(param_space.topo):
        raise FileNotFoundError(f"Topology file not found: {param_space.topo}")

    traj = param_dict["generals"]["trajectory"]
    trajs = [normpath(join(root_dir, x)) for x in traj.split()]
    param_space.traj = " ".join(trajs)
    if not os.path.exists(param_space.out_dir):
        os.makedirs(param_space.out_dir, exist_ok=True)

    return param_space
