import math
import sys
sys.path.append('/home/ly/Desktop/MLPaper')
from typing import Callable, Dict, List, Set, Tuple
from extract_templates import get_changed_atoms
from rdkit import Chem
from rdkit.Chem import Draw
from estimator import predict
from typing import Literal
from util import drop_map_num, drop_hs_map_num, str_to_mol, find_clusters, extract_subgraph

C_PUCT = 100


class ChempropModel:
    def __init__(self, cp_dir: str, num_models: int = 5,
                 value_type: Literal['pred', 'aleatoric', 'epistemic', 'total'] = 'pred'):
        self.base_pred = None
        self.base_model_unc = None
        self.base_data_unc = None
        self.base_total_unc = None
        self.cp_dir = cp_dir
        self.num_models = num_models
        self.value_type = value_type
        
    def set_base_line(self, rxn: str):
        smis = [[rxn]]
        pred = predict(smiles_list=smis, ensemble_dir=self.cp_dir, num_models=self.num_models)
        self.base_pred, self.base_model_unc, self.base_data_unc, self.base_total_unc \
            = pred[0, 0], pred[0, 1], pred[0, 2], pred[0, 3]

    def __call__(self, rxns: List[str]) -> List[List[float]]:
        smis = [[smi] for smi in rxns]
        pred = predict(smiles_list=smis, ensemble_dir=self.cp_dir, num_models=self.num_models)

        value, model_unc, data_unc, total_unc = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        if self.value_type == 'aleatoric':
            score = 1 / (abs(data_unc - self.base_data_unc) / self.base_data_unc + 0.1)  # to constrain score from 0-10
        elif self.value_type == 'epistemic':
            score = 1 / (abs(model_unc - self.base_model_unc) / self.base_model_unc + 0.1)
        elif self.value_type == 'total':
            score = 1 / (abs(total_unc - self.base_total_unc) / self.base_total_unc + 0.1)
        else:
            score = 1 / (abs(value - self.base_pred) / self.base_pred + 0.1)

        results = [[a, b, c, d, e] for a, b, c, d, e in zip(score, value, model_unc, data_unc, total_unc)]

        return results


class MCTSNode:
    """A :class:`MCTSNode` represents a node in a Monte Carlo Tree Search."""

    def __init__(self, rsmi: str, psmi: str,  atoms: Set[int], W: float = 0, N: int = 0, P: float = 0) -> None:
        """
        :param smiles: The SMILES for the substructure at this node.
        :param atoms: A list of atom indices represented by this node.
        :param W: The W value of this node.
        :param N: The N value of this node.
        :param P: The P value of this node.
        """
        self.rsmi = rsmi
        self.psmi = psmi
        self.atoms = atoms
        self.children = []
        self.W = W
        self.N = N
        self.P = P
        self.value = None
        self.model_unc = None
        self.data_unc = None
        self.total_unc = None

    def __repr__(self):
        return f'{self.rsmi}>>{self.psmi} (W={self.W}, N={self.N}, P={self.P})'

    def set_score(self, score, value, model_unc, data_unc, total_unc):
        self.P = score
        self.value = value
        self.model_unc = model_unc
        self.data_unc = data_unc
        self.total_unc = total_unc

    @property
    def smarts(self):
        return '>>'.join([self.rsmi, self.psmi])

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else self.P  # if self.N <=0 , return 0 in chemprop initial version

    def U(self, n: int) -> float:
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def abbr_smarts(self):
        return f'{drop_map_num(self.rsmi)}>>{drop_map_num(self.psmi)}'


def draw_mcts(rationals: List[MCTSNode]):
    """
    Draw mcts
    :param rationals:
    :return: None
    """
    rsmis = [rational.rsmi for rational in rationals]
    rmols = [str_to_mol(rsmi) for rsmi in rsmis]
    rsmis_labels = [f'W={rational.W:.2f}, N={rational.N:.2f}, P={rational.P:.2f}' for rational in rationals]
    psmis = [rational.psmi for rational in rationals]
    pmols = [str_to_mol(psmi) for psmi in psmis]
    psmis_labels = [f'W={rational.W:.2f}, N={rational.N:.2f}, P={rational.P:.2f}' for rational in rationals]

    rimg = Draw.MolsToGridImage(rmols, molsPerRow=3, subImgSize=(200, 200), legends=rsmis_labels, returnPNG=False)
    pimg = Draw.MolsToGridImage(pmols, molsPerRow=3, subImgSize=(200, 200), legends=psmis_labels, returnPNG=False)

    rimg.save('rmol.png')
    pimg.save('pmol.png')


def mcts_rollout(node: MCTSNode,
                 state_map: Dict[str, MCTSNode],
                 orig_rsmi: str,
                 orig_psmi: str,
                 clusters: List[Set[int]],
                 atom_cls: List[Set[int]],
                 nei_cls: List[Set[int]],
                 scoring_function: Callable[[List[str]], List[List[float]]],
                 forbidden_num: Set[int]) -> float:
    """
    A Monte Carlo Tree Search rollout from a given :class:`MCTSNode`.

    :param forbidden_num: A list of numbers belong to reaction center
    :param node: The :class:`MCTSNode` from which to begin the rollout.
    :param state_map: A mapping from SMILES to :class:`MCTSNode`.
    :param orig_smiles: The original SMILES of the molecule.
    :param clusters: Clusters of atoms.
    :param atom_cls: Atom indices in the clusters.
    :param nei_cls: Neighboring clusters.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :return: The score of this MCTS rollout.
    """
    cur_atoms = node.atoms
    """if len(cur_atoms) <= len(forbidden_num):
        return node.P"""

    # Expand if this node has never been visited
    if len(node.children) == 0:
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])  # get clusters belong to current atoms
        print('clusters: ', clusters)
        print('current cluster index: ', cur_cls)
        print('current atoms: ', cur_atoms)
        print('atom_cls: ', atom_cls)
        for i in cur_cls:
            # if an atom belongs to two clusters, it isn't a leaf atom
            # if an atom belongs to forbidden_num, we only delete other atoms in this cluster
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1 and a not in forbidden_num]
            """if any([a for a in leaf_atoms if a in forbidden_num]):
                continue"""
            print('leaf atoms: ', leaf_atoms)
            if leaf_atoms:
                if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                    new_atoms = cur_atoms - set(leaf_atoms)
                    try:
                        new_rsmi, add_num = extract_subgraph(orig_rsmi, new_atoms, check_radicals=True)
                        new_psmi, _ = extract_subgraph(orig_psmi, new_atoms, add_num, check_radicals=False)
                    except:
                        continue
                    if len(new_psmi.split('.')) != len(orig_psmi.split('.')):
                        continue
                    if drop_hs_map_num(new_rsmi) in state_map:
                        new_node = state_map[drop_hs_map_num(new_rsmi)]  # merge identical states
                    else:
                        new_node = MCTSNode(new_rsmi, new_psmi, new_atoms)
                    if new_rsmi:
                        node.children.append(new_node)

        state_map[drop_hs_map_num(node.rsmi)] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves

        scores = scoring_function([x.smarts for x in node.children])
        for child, score in zip(node.children, scores):
            child.set_score(*score)

    sum_count = sum(c.N for c in node.children)
    print(f'Q of current sate children is: ', [c.Q() for c in node.children])
    print(f'action function of current state is: ', [c.Q() + c.U(sum_count) for c in node.children])
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count))  # action function
    print(f'selected node is: ', selected_node)
    print(f'selected node score is: ', selected_node.P)
    v = mcts_rollout(selected_node, state_map, orig_rsmi, orig_psmi, clusters,
                     atom_cls, nei_cls, scoring_function, forbidden_num)
    selected_node.W += v
    selected_node.N += 1

    return v


def mcts(rsmi: str,
         psmi: str,
         scoring_function: ChempropModel,
         n_rollout: int,
         max_atoms: int,
         prop_delta: float,
         forbidden_num: Set[int]) -> List[MCTSNode]:
    """
    Runs the Monte Carlo Tree Search algorithm.

    :param smiles: The SMILES of the molecule to perform the search on.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :param n_rollout: THe number of MCTS rollouts to perform.
    :param max_atoms: The maximum number of atoms allowed in an extracted rationale.
    :param prop_delta: The minimum required property value for a satisfactory rationale.
    :return: A list of rationales each represented by a :class:`MCTSNode`.
    """

    rmol = Chem.MolFromSmiles(rsmi)

    clusters, atom_cls = find_clusters(rmol)
    print(f'clusters: {clusters}, atom clusters" {atom_cls}')

    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in atom_cls.keys():
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(rsmi, psmi, set([atom.GetAtomMapNum() for atom in rmol.GetAtoms() if atom.GetSymbol() != 'H']))
    root.set_score(0, scoring_function.base_pred,
                   scoring_function.base_model_unc,
                   scoring_function.base_data_unc,
                   scoring_function.base_total_unc)
    state_map = {drop_hs_map_num(rsmi): root}
    for _ in range(n_rollout):
        mcts_rollout(root, state_map, rsmi, psmi, clusters, atom_cls, nei_cls, scoring_function, forbidden_num)

    rationales = [node for _, node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]

    return rationales


def interpret(checkpoint_dir: str,
              all_rxns: List[str],
              rollout: int = 5,
              max_atoms: int = 20,
              prop_delta: float = 0.0,
              num_models: int = 5,
              score_func_value_type: Literal['pred', 'aleatoric', 'epistemic', 'total'] = 'pred'
              ) -> List[List[MCTSNode]]:
    """
    Runs interpretation of a Chemprop model using the Monte Carlo Tree Search algorithm.
    """
    global C_PUCT, MIN_ATOMS

    all_rationals = []
    for rxn in all_rxns:
        scoring_function = ChempropModel(num_models=num_models,
                                         cp_dir=checkpoint_dir,
                                         value_type=score_func_value_type)
        scoring_function.set_base_line(rxn)
        rsmi, psmi = rxn.split('>')[0], rxn.split('>')[-1]
        changed_atoms, changed_atom_tags, err = get_changed_atoms([str_to_mol(rsmi)],
                                                                  [str_to_mol(psmi)])
        FORBIDEN_NUM = set([int(idx) for idx in changed_atom_tags])
        print('forbidden num: ', FORBIDEN_NUM)
        rationales = mcts(
            rsmi=rsmi,
            psmi=psmi,
            scoring_function=scoring_function,
            n_rollout=rollout,
            max_atoms=max_atoms,
            prop_delta=prop_delta,
            forbidden_num=FORBIDEN_NUM
        )

        all_rationals.append(rationales)

    return all_rationals


def run_interpert(rxn, cp_dir):
    all_rat = interpret(cp_dir, [rxn])[0]
    nodes_dict = {}
    for i, node in enumerate(all_rat):
        rsmi = str(node.rsmi)
        psmi = str(node.psmi)
        score = float(node.P)
        pred = float(node.value)
        model_unc = float(node.model_unc)
        data_unc = float(node.data_unc)
        total_unc = float(node.total_unc)
        nodes_dict[i] = {'rsmi': rsmi, 'psmi': psmi, 'score': score, 'pred': pred,
                         'model_unc': model_unc, 'data_unc': data_unc, 'total_unc': total_unc}
    return nodes_dict


