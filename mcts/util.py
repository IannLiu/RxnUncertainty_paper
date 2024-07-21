from rdkit import Chem
from typing import Dict, List, Tuple, Set


RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()

bond_types = [Chem.BondType.SINGLE,
              Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE,
              Chem.BondType.AROMATIC]


def str_to_mol(string: str, explicit_hydrogens: bool = True) -> Chem.Mol:
    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        # Set params here so we don't remove hydrogens with atom mapping
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

    if explicit_hydrogens:
        return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)


def drop_map_num(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol)


def drop_hs_map_num(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol)


def add_h_atom(mol: Chem.Mol, idx: int, atom_map_num: int = None):

    new_mol = Chem.RWMol(mol)
    atom = Chem.Atom(1)
    atom.SetAtomMapNum(atom_map_num)
    new_mol.AddAtom(atom)
    new_mol.AddBond(new_mol.GetNumAtoms() - 1, idx, Chem.BondType.SINGLE)

    return new_mol.GetMol()



def mapping_back(origin_smi: str, smiles: str, add_num: Dict[int, List[int]] = None):
    """
    Mapping map number back

    :param origin_smi: All atoms mapped molecule smiles
    :param smiles: Heavy atoms mapped molecule smiles
    :param add_num: add additional map number
    :return:
    """
    orig_mol = str_to_mol(origin_smi)
    mol = str_to_mol(smiles)
    num_idx_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    num_rad_map = {atom.GetAtomMapNum(): atom.GetNumRadicalElectrons() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H'}
    orig_num_rad_map = {atom.GetAtomMapNum(): atom.GetNumRadicalElectrons() for atom in orig_mol.GetAtoms() if atom.GetSymbol() != 'H'}
    add_hs_map = {}
    max_map_num = max([atom.GetAtomMapNum() for atom in orig_mol.GetAtoms()]) + 1
    for num, rad in num_rad_map.items():
        if orig_num_rad_map[num] < rad:
            add_hs_map[num] = rad - orig_num_rad_map[num]

    mol = Chem.RWMol(mol)
    if add_num is None:
        add_num = {}
    for num, hs in add_hs_map.items():
        if num not in add_num.keys():
            assig_nums = list(range(max_map_num, max_map_num + hs))
            max_map_num += hs
        else:
            assig_nums = add_num[num]
        for assig_num in assig_nums:
            atom = Chem.Atom(1)
            atom.SetAtomMapNum(assig_num)
            mol.AddAtom(atom)
            mol.AddBond(mol.GetNumAtoms()-1, num_idx_map[num], Chem.BondType.SINGLE)

    # print(Chem.MolToSmiles(mol))
    # Chem.SanitizeMol(mol)

    return Chem.MolToSmiles(mol), add_num


def find_clusters(mol: Chem.Mol, idx_num_map: dict = None) -> Tuple[List[Tuple[int, ...]], Dict[int, List[int]]]:
    """
    Finds clusters within the molecule.

    :param idx_num_map:
    :param mol: An RDKit molecule.
    :return: A tuple containing a list of atom tuples representing the clusters
             and a list of lists of atoms in each cluster.
    """
    n_atoms = mol.GetNumAtoms()
    atom_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H']
    if idx_num_map is None:
        idx_num_map = {atom.GetIdx(): atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H'}
    if n_atoms == 1:  # special case
        atom_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        return [(atom_nums[0],)], {atom_nums[0]: [atom_nums[0]]}

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetSymbol() == 'H' or a2.GetSymbol() == 'H':
            continue
        if not bond.IsInRing():
            clusters.append((a1.GetAtomMapNum(), a2.GetAtomMapNum()))

    for sssr in Chem.GetSymmSSSR(mol):
        r = []
        for idx in tuple(sssr):
            r.append(idx_num_map[idx])
        clusters.append(tuple(r))

    atom_cls = {num: [] for num in atom_nums}
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def __extract_subgraph(mol: Chem.Mol, selected_atoms: set, add_num: dict = None, check_radicals: bool = True):
    """
    Extracts a subgraph from an RDKit molecule given a set of atom indices.

    :param mol: An RDKit molecule from which to extract a subgraph.
    :param selected_atoms: The atoms which form the subgraph to be extracted.
    :return: A tuple containing an RDKit molecule representing the subgraph
             and a list of root atom indices from the selected indices.
    """
    num_idx_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    selected_atom_idx = set([num_idx_map[num] for num in selected_atoms])
    delete_atom_idx = set([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H' and atom.GetAtomMapNum() not in selected_atoms])
    num_rad_map = {atom.GetAtomMapNum(): atom.GetNumRadicalElectrons() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H'}

    roots = []
    selected_bonds = []
    for idx in selected_atom_idx:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y.GetIdx() for y in atom.GetNeighbors() if y.GetIdx() not in selected_atom_idx and y.GetSymbol() != 'H']
        if len(bad_neis) > 0:
            roots.append(idx)
            for bad_nei in bad_neis:
                selected_bonds.append(tuple([idx, bad_nei]))

    print("selected atoms: ", selected_atoms)
    print('add num: ', add_num)
    # if add_num is None, count Hs should be added to selected atoms
    if add_num is None:
        add_num = {}
        max_map_num = max(list(num_idx_map.keys())) + 1
        for a1, a2 in selected_bonds:
            bond = mol.GetBondBetweenAtoms(a1, a2).GetBondType()
            bond_type_idx = bond_types.index(bond)
            hs_num = bond_type_idx + 1 if bond_type_idx <= 2 else 1
            a1_num = mol.GetAtomWithIdx(a1).GetAtomMapNum()
            if a1_num not in add_num.keys():
                add_num[a1_num] = set(range(max_map_num, max_map_num + hs_num))
            else:
                add_num[a1_num].update(range(max_map_num, max_map_num + hs_num))
            max_map_num += hs_num

    # extend selected H atom index
    extend_idxs = []
    for idx in selected_atom_idx:
        atom = mol.GetAtomWithIdx(idx)
        extend_idx = [y.GetIdx() for y in atom.GetNeighbors() if y.GetIdx() not in selected_atom_idx and y.GetSymbol() == 'H']
        extend_idxs.extend(extend_idx)
    selected_atom_idx.update(extend_idxs)

    delete_idxs = []
    for idx in delete_atom_idx:
        atom = mol.GetAtomWithIdx(idx)
        delete_idx = [y.GetIdx() for y in atom.GetNeighbors() if y.GetIdx() not in delete_atom_idx and y.GetSymbol() == 'H']
        delete_idxs.extend(delete_idx)
    delete_atom_idx.update(delete_idxs)


    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        # atom.SetAtomMapNum(1000)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if
                       bond.GetBeginAtom().GetIdx() in selected_atom_idx and bond.GetEndAtom().GetIdx() in selected_atom_idx]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = sorted(delete_atom_idx, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    # add Hs
    cur_num_idx_dict = {a.GetAtomMapNum(): a.GetIdx() for a in new_mol.GetAtoms()}
    for num, num_list in add_num.items():
        for assig_num in num_list:
            h = Chem.Atom(1)
            h.SetAtomMapNum(assig_num)
            new_mol.AddAtom(h)
            new_mol.AddBond(new_mol.GetNumAtoms()-1, cur_num_idx_dict[num], Chem.BondType.SINGLE)

    print(Chem.MolToSmiles(new_mol))
    Chem.SanitizeMol(new_mol)
    # Check radicals
    if check_radicals:
        max_map_num = max([a.GetAtomMapNum() for a in new_mol.GetAtoms()])
        cur_num_idx_map = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetSymbol() != 'H'}
        cur_num_rad_map = {atom.GetAtomMapNum(): atom.GetNumRadicalElectrons() for atom in new_mol.GetAtoms() if atom.GetSymbol() != 'H'}
        for num, rad in cur_num_rad_map.items():
            if rad - num_rad_map[num] > 0:
                assig_nums = set(range(max_map_num, max_map_num + rad - num_rad_map[num]))
                for assig_num in assig_nums:
                    h = Chem.Atom(1)
                    h.SetAtomMapNum(assig_num)
                    new_mol.AddAtom(h)
                    new_mol.AddBond(new_mol.GetNumAtoms()-1, cur_num_idx_map[num], Chem.BondType.SINGLE)
                max_map_num += rad - num_rad_map[num]
                if num in add_num.keys():
                    add_num[num].update(assig_nums)
                else:
                    add_num[num] = assig_nums

    return new_mol.GetMol(), add_num


def extract_subgraph(smiles: str,
                     selected_atoms: Set[int],
                     add_num: dict = None,
                     check_radicals: bool = False):
    """
    Extracts a subgraph from a SMILES given a set of atom indices.

    :param smiles: A SMILES from which to extract a subgraph.
    :param selected_atoms: The atoms which form the subgraph to be extracted.
    :return: A tuple containing a SMILES representing the subgraph
             and a list of root atom indices from the selected indices.
    """
    # try with kekulization
    mol = str_to_mol(smiles)
    Chem.Kekulize(mol)
    subgraph, add_num = __extract_subgraph(mol, selected_atoms, add_num, check_radicals)
    try:
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = str_to_mol(subgraph)
    except Exception:
        subgraph = None

    mol = str_to_mol(smiles)  # de-kekulize
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), add_num

    # If fails, try without kekulization
    subgraph, roots = __extract_subgraph(mol, selected_atoms)
    subgraph = Chem.MolToSmiles(subgraph)
    print(subgraph)
    subgraph = str_to_mol(subgraph)

    if subgraph is not None:
        return Chem.MolToSmiles(subgraph), add_num
    else:
        return None, None
