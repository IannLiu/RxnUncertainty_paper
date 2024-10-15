import re
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from util import str_to_mol

VERBOSE = False


def clear_mapnum(mol):
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
    return mol
    

def mols_from_smiles_list(all_smiles):
    """Given a list of smiles strings, this function creates rdkit
    molecules"""
    mols = []
    for smiles in all_smiles:
        if not smiles: continue
        mols.append(str_to_mol(smiles))
    return mols


def get_tagged_atoms_from_mols(mols):
    """Takes a list of RDKit molecules and returns total list of
    atoms and their tags"""
    atoms = []
    atom_tags = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms
        atom_tags += new_atom_tags
    return atoms, atom_tags


def get_tagged_atoms_from_mol(mol):
    """Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers"""
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags


def nb_H_Num(atom):
    nb_Hs = 0
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() == 'H':
            nb_Hs = nb_Hs + 1
    return nb_Hs

    
def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])
    

def atoms_are_different(atom1, atom2):
    """Compares two RDKit atoms based on basic properties"""

    if atom1.GetAtomicNum() != atom2.GetAtomicNum():
        return True  # must be true for atom mapping
    # Because of explicit Hs, we must count Hs manually
    atom1_Hs = nb_H_Num(atom1)
    atom2_Hs = nb_H_Num(atom2)
    if atom1_Hs != atom2_Hs:
        return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge():
        return True
    # if Hs number is same, the atom degrees are independent on whether or not Hs are explicit in the graph
    if atom1.GetDegree() != atom2.GetDegree():
        return True
    # if atom1.IsInRing() != atom2.IsInRing(): return True # do not want to check this!
    # e.g., in macrocycle formation, don't want the template to include the entire ring structure
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons():
        return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic():
        return True

    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()])
    if bonds1 != bonds2:
        return True

    return False


def get_changed_atoms(reactants, products):
    """Looks at mapped atoms in a reaction and determines which ones changed"""

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)

    if VERBOSE:
        print('Products contain {} tagged atoms'.format(len(prod_atoms)))
    if VERBOSE:
        print('Products contain {} unique atom numbers'.format(len(set(prod_atom_tags))))

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)
    if len(set(prod_atom_tags)) != len(set(reac_atom_tags)):
        if VERBOSE:
            print('warning: different atom tags appear in reactants and products')
        # err = 1 # okay for Reaxys, since Reaxys creates mass
    if len(prod_atoms) != len(reac_atoms):
        if VERBOSE:
            print('warning: total number of tagged atoms differ, stoichometry != 1?')
        # err = 1

    # Find differences 
    changed_atoms = []  # actual reactant atom species
    changed_atom_tags = []  # atom map numbers of those atoms

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):
        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag:  # Find same tags and compare them.
                continue
            if reac_tag not in changed_atom_tags:  # don't bother comparing if we know this atom changes
                # If atom changed, add
                if atoms_are_different(prod_atoms[i], reac_atoms[j]):
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)# why?
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atom_tags:
            if reac_tag not in prod_atom_tags:
                changed_atoms.append(reac_atoms[j])
                changed_atom_tags.append(reac_tag)

    if VERBOSE:
        print('{} tagged atoms in reactants change 1-atom properties'.format(len(changed_atom_tags)))
        for smarts in [atom.GetSmarts() for atom in changed_atoms]:
            print('  {}'.format(smarts))
    return changed_atoms, changed_atom_tags, err


def bond_index(bond):
    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
        return 4
    else:
        return 5


def Extract_changed_bonds(smilles: list, map_num_info: bool = False):
    """
    The function is to extract the changed bond types for a reactiion

    :param smiles: A list [rsmi, psmi]
    """
    rmol = str_to_mol(smilles[0])
    pmol = str_to_mol(smilles[1])
    pmol_map_to_idx = {}
    rmol_map_to_idx = {}
    changed_bond_types = []
    broken = []
    formation = []
    for atom in pmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        idx = atom.GetIdx()
        if map_num not in pmol_map_to_idx.keys():
            pmol_map_to_idx[map_num] = idx
    for atom in rmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        idx = atom.GetIdx()
        if map_num not in rmol_map_to_idx.keys():
            rmol_map_to_idx[map_num] = idx
    for rbond in rmol.GetBonds():
        if rbond.GetSmarts() in ['', None]:
            rbond_smarts = '-'
        else:
            rbond_smarts = rbond.GetSmarts()
        atom1 = rbond.GetBeginAtom()
        atom1_map_num = atom1.GetAtomMapNum()
        atom2 = rbond.GetEndAtom()
        atom2_map_num = atom2.GetAtomMapNum()
        pbond = pmol.GetBondBetweenAtoms(pmol_map_to_idx[atom1_map_num], pmol_map_to_idx[atom2_map_num])
        changed_ratoms = sorted([atom1.GetSymbol(), atom2.GetSymbol()])
        if pbond is None or pbond.GetSmarts() != rbond.GetSmarts():
            changed_bond_types.append('-' + changed_ratoms[0] + rbond_smarts + changed_ratoms[1])
            if map_num_info:
                broken.append([atom1_map_num, atom2_map_num, bond_index(rbond)])

    for pbond in pmol.GetBonds():
        if pbond.GetSmarts() in ['', None]:
            pbond_smarts = '-'
        else:
            pbond_smarts = pbond.GetSmarts()
        atom1 = pbond.GetBeginAtom()
        atom1_map_num = atom1.GetAtomMapNum()
        atom2 = pbond.GetEndAtom()
        atom2_map_num = atom2.GetAtomMapNum()
        rbond = rmol.GetBondBetweenAtoms(rmol_map_to_idx[atom1_map_num], rmol_map_to_idx[atom2_map_num])
        changed_ratoms = sorted([atom1.GetSymbol(), atom2.GetSymbol()])
        if rbond is None or rbond.GetSmarts() != pbond.GetSmarts():
            changed_bond_types.append('+' + changed_ratoms[0] + pbond_smarts + changed_ratoms[1])
            if map_num_info:
                # [atom_idx1, atom_idx2, bond_change_type]
                formation.append([atom1_map_num, atom2_map_num, bond_index(pbond)])

    changed_bond_types = sorted(changed_bond_types)

    if map_num_info:
        return '.'.join(changed_bond_types), {'broken': broken, 'formation': formation}

    return '.'.join(changed_bond_types)

