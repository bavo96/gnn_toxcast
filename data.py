import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")


def smiles_to_data(smiles, label):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    Each atom is a node with its atomic number as a feature.
    Edges are bonds, and edge_attr includes bond type as one-hot vectors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Skip invalid molecules

    # Create node features: using the atomic number as a feature
    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Create edge_index and edge_attr
    edge_index = []
    edge_attr = []
    bond_type_to_feature = {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
        Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
    }

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        edge_feature = bond_type_to_feature.get(
            bond_type, [0, 0, 0, 0]
        )  # Default to [0,0,0,0] if unknown

        edge_index.append([start, end])
        edge_index.append([end, start])
        edge_attr.append(edge_feature)
        edge_attr.append(edge_feature)  # Undirected graph

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(
            (0, 4), dtype=torch.float
        )  # Shape (num_edges, num_features)

    # Create the label tensor (for classification)
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data
