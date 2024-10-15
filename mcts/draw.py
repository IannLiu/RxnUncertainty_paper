from rdkit import Geometry
import rdkit.Chem.Draw as Draw
from extract_templates import get_changed_atoms
from io import BytesIO
from PIL import Image, ImageOps
from rdkit.Chem import rdDepictor
from typing import Union
from rdkit import Chem

import numpy as np


'''
Many of these functions are taken from RDKit.
'''


def drop_hs_map_num(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol)


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


def GetIdxToMapNum(mol: Union[str, Chem.Mol], explicit_hydrogens: bool = True) -> dict:
    """
    Return a dictionary of which keys are atom index values are atom map number
    """
    idx_to_map = {}
    if type(mol).__name__ == 'str':
        mol = str_to_mol(mol, explicit_hydrogens=explicit_hydrogens)
    for atom in mol.GetAtoms():
        idx_to_map[atom.GetIdx()] = atom.GetAtomMapNum()

    return idx_to_map


def DrawReaction(rxn: str,
                 filename: str = None,
                 label: str = None,
                 explicitHydrogens: bool = False,
                 highlightChangedAtomMap: list = None,
                 highlightChangedAtoms: bool = True,
                 padding_factor: float = 1.0,
                 acsForm: bool = False):
    """
    Return a reaction picture with svg context
    To show svg in jupyter notebook:

    from IPython.display import SVG
    SVG(svg)

    """
    arrowLength = 60
    arrowPadding = 10
    addFactor = 1.5
    addPadding = 10
    labelPadding = 10
    labelFactor = 1
    rsmis, psmis = rxn.split('>>')[0].split('.'), rxn.split('>>')[-1].split('.')
    if not explicitHydrogens:
        rsmis = [drop_hs_map_num(smi) for smi in rsmis]
        psmis = [drop_hs_map_num(smi) for smi in psmis]
        rmols, pmols = [Chem.MolFromSmiles(rsmi) for rsmi in rsmis], [Chem.MolFromSmiles(psmi) for psmi in psmis]
    else:
        rmols, pmols = [str_to_mol(rsmi, explicit_hydrogens=explicitHydrogens) for rsmi in rsmis], [
            str_to_mol(psmi, explicit_hydrogens=explicitHydrogens) for psmi in psmis]

    if not highlightChangedAtomMap:
        changed_atoms, changed_atom_tags, err = get_changed_atoms(rmols, pmols)
        changed_atom_map = [atom.GetAtomMapNum() for atom in changed_atoms]
    else:
        changed_atom_map = highlightChangedAtomMap

    rs_highlightAtomMap = []
    ps_highlightAtomMap = []
    for mol in rmols:
        highlight_atoms = []
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num in changed_atom_map:
                highlight_atoms.append(atom.GetIdx())
            if atom.GetSymbol() != 'H':
                atom.SetProp("atomNote", str(map_num))
                atom.SetAtomMapNum(0)
        if not highlightChangedAtoms:
            highlight_atoms = None
        rs_highlightAtomMap.append(highlight_atoms)
    for mol in pmols:
        highlight_atoms = []
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num in changed_atom_map:
                highlight_atoms.append(atom.GetIdx())
            if atom.GetSymbol() != 'H':
                atom.SetProp("atomNote", str(map_num))
                atom.SetAtomMapNum(0)
        if not highlightChangedAtoms:
            highlight_atoms = None
        ps_highlightAtomMap.append(highlight_atoms)

    sizer = Draw.MolDraw2DSVG(-1, -1)
    rs_size = [sizer.GetMolSize(mol) for mol in rmols]
    ps_size = [sizer.GetMolSize(mol) for mol in pmols]
    addWidth = sizer.FontSize() * addFactor
    defaultFontSize = sizer.FontSize()
    labelHeight = sizer.FontSize() * labelFactor

    r_addNum = len(rmols) - 1
    p_addNum = len(pmols) - 1

    height = max(max([size[1] for size in rs_size]), max([size[1] for size in ps_size]))
    if label is not None:
        secondHeight = max(max([size[1] for size in rs_size]), max([size[1] for size in ps_size])) \
                       + labelHeight + labelPadding
    else:
        secondHeight = height * padding_factor
    width = arrowLength + arrowPadding + r_addNum * (addPadding + addWidth) + p_addNum * (addPadding + addWidth)
    for size in rs_size:
        width += size[0]
    for size in ps_size:
        width += size[0]

    d2d = Draw.MolDraw2DSVG(int(width), int(secondHeight))
    d2d.SetFlexiMode(True)
    d2d.ClearDrawing()
    dopts = d2d.drawOptions()
    dopts.includeAtomTags = True
    dopts.annotationFontScale = 0.7
    if acsForm:
        dopts.useBWAtomPalette()
    # dopts.clearBackground = False
    dopts.includeAtomTags = True

    width_pos = 0
    for mol, size, highlight_atoms in zip(rmols, rs_size, rs_highlightAtomMap):
        d2d.SetOffset(int(width_pos), (height - size[1]) // 2)
        d2d.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        width_pos += size[0]
        if r_addNum > 0:
            addPos = Geometry.Point2D(width_pos + (addPadding + addWidth) // 2, height // 2)
            d2d.SetFontSize(d2d.FontSize() * addFactor)
            d2d.DrawString("+", addPos, 0, rawCoords=True)
            d2d.SetFontSize(defaultFontSize)
            r_addNum -= 1
            width_pos += (addPadding + addWidth)

    # Draw arrow
    arrowStart = Geometry.Point2D(width_pos + arrowPadding // 2, height // 2)
    arrowEnd1 = Geometry.Point2D(width_pos + arrowPadding // 2 + arrowLength, height // 2)
    width_pos += arrowPadding + arrowLength
    d2d.SetOffset(0, 0)
    # d2d.SetLineWidth(4)
    d2d.DrawArrow(arrowStart, arrowEnd1, asPolygon=True, rawCoords=True)

    for mol, size, highlight_atoms in zip(pmols, ps_size, ps_highlightAtomMap):
        d2d.SetOffset(int(width_pos), (height - size[1]) // 2)
        d2d.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        width_pos += size[0]
        if p_addNum > 0:
            addPos = Geometry.Point2D(width_pos + (addPadding + addWidth) // 2, height // 2)
            d2d.SetFontSize(d2d.FontSize() * addFactor)
            d2d.DrawString("+", addPos, 0, rawCoords=True)
            d2d.SetFontSize(defaultFontSize)
            p_addNum -= 1
            width_pos += (addPadding + addWidth)

    # add label
    if label is not None:
        addPos = Geometry.Point2D(width // 2, height + ((labelPadding + labelHeight) // 2))
        d2d.SetFontSize(d2d.FontSize() * labelFactor)
        d2d.DrawString(label, addPos, 0, rawCoords=True)

    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    if filename:
        with open(filename, 'w') as outfile:
            outfile.write(svg)

    return svg


def draw_mol(smiles: str,
             filename: str = None,
             highlight_atom_map: dict = None,
             explicitHydrogens: bool = False):
    """
    Return a molecule picture with svg context
    To show svg in jupyter notebook:

    from IPython.display import SVG
    SVG(svg)

    """
    mol = str_to_mol(smiles, explicitHydrogens)
    d2d = Draw.MolDraw2DSVG(-1, -1)
    highlight_atoms = []
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num in highlight_atom_map:
            highlight_atoms.append(atom.GetIdx())
        atom.SetProp("atomNote", str(map_num))
        atom.SetAtomMapNum(0)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    dopts = d2d.drawOptions()
    dopts.includeAtomTags = True
    dopts.annotationFontScale = 0.7
    # Draw.DrawMoleculeACS1996(d2d,mol,legend="II")
    d2d.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    if filename:
        with open(filename, 'w') as outfile:
            outfile.write(svg)

    return svg

