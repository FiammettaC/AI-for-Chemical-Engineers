import torch
from rdkit import Chem
import random
import numpy as np


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GenFeatures:
    """
    class fo generating features to attribute the molecular graph and includes node and edge features.
    features are generated according to Table 1: https://doi.org/10.1021/acs.jmedchem.9b00959
    """

    def __init__(self):
        # define possible atoms
        self.symbols = [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
            'Te', 'I', 'At', 'other'
        ]
        # define possible hybridizations
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]
        # define possible stereochemistry
        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        # Generate mol structure
        mol = Chem.MolFromSmiles(data.smiles)

        # node attribution
        xs = []
        for atom in mol.GetAtoms():
            # node type: atoms
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            # atom degree: no. bounds
            degree = [0.] * 6
            degree[atom.GetDegree()] = 1.
            # atoms formal charge
            formal_charge = atom.GetFormalCharge()
            # number of radical electrons
            radical_electrons = atom.GetNumRadicalElectrons()
            # type of hybridization
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.
            # aromaticity
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            # no. of attached Hydrogen atoms
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            # type of chirality
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
            # converting the features into a tensor
            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            # collect all node features of the molecule
            xs.append(x)
        # stack the node features
        data.x = torch.stack(xs, dim=0)

        # edge attribution
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            # adjacency information
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
            # bond type
            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            # is bond conjugated
            conjugation = 1. if bond.GetIsConjugated() else 0.
            # is bond part of aromatic structure
            ring = 1. if bond.IsInRing() else 0.
            # stereo information
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.
            # converting features into tensor
            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


