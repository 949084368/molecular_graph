# -*- coding: UTF-8 -*-
'''
@Project ：XXX_GNN 
@File    ：feature3d.py
@Author  ：Mental-Flow
@Date    ：2024/4/3 18:01 
@introduction :
'''
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

def get_mol(smiles):  # 从smiles得到对应的3d信息
    mol = AllChem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except ValueError as e:  # 如果优化出错就返回优化前的
        print("优化分子时发生错误:", e)
        print(smiles)
    return mol


# 根据分子3D坐标，求出分子所占体积
def Get_volume(coordinate_matrix):
    x_max = 0
    y_max = 0
    z_max = 0
    for index in range(0, len(coordinate_matrix)):
        if abs(coordinate_matrix[index][0]) > x_max:
            x_max = abs(coordinate_matrix[index][0])
        if abs(coordinate_matrix[index][1]) > y_max:
            y_max = abs(coordinate_matrix[index][1])
        if abs(coordinate_matrix[index][2]) > z_max:
            z_max = abs(coordinate_matrix[index][2])
    max = x_max
    if y_max > max:
        max = y_max
    if z_max > max:
        max = z_max
    for node in coordinate_matrix:
        for n in node:
            if abs(n) == max:
                coordinate_matrix.remove(node)
                break

    volume = x_max * y_max * z_max
    return volume, coordinate_matrix

def get_molecule_positions(suppl):
    positions = []
    mol = suppl  # 自己生成的就这样，通过 cid 下载的 sdf 文件，需要 mol = suppl[0]
    if mol is not None:
        try:
            conformer = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            positions.append([list(conformer.GetAtomPosition(i)) for i in range(num_atoms)])
            return [list(conformer.GetAtomPosition(i)) for i in range(num_atoms)]
        except ValueError as e:
            print("获取分子位置时发生错误:", e)
    return positions

def Get_3layers_volume(coordinate_matrix):
    volume_vect = []
    volume1, coordinate_matrix1 = Get_volume(coordinate_matrix)
    # print(coordinate_matrix1)
    volume2, coordinate_matrix2 = Get_volume(coordinate_matrix1)
    # print(coordinate_matrix2)
    volume3, coordinate_matrix3 = Get_volume(coordinate_matrix2)
    # print(coordinate_matrix3)
    volume4, coordinate_matrix4 = Get_volume(coordinate_matrix3)
    # print(coordinate_matrix4)
    volume5, coordinate_matrix5 = Get_volume(coordinate_matrix4)
    # print(coordinate_matrix5)
    volume6, coordinate_matrix6 = Get_volume(coordinate_matrix5)
    # print(coordinate_matrix6)
    volume_vect.append(volume1)
    volume_vect.append(volume2)
    volume_vect.append(volume3)
    volume_vect.append(volume4)
    volume_vect.append(volume5)
    volume_vect.append(volume6)
    return volume_vect

def smile2volume(smile):
    mol_sdf = get_mol(smile)
    mol_list = get_molecule_positions(mol_sdf)
    v = torch.tensor(Get_3layers_volume(mol_list))
    return v

if __name__ == "__main__":
    print(smile2volume('O=C(c1cccs1)N1CCC(c2nc3ccccc3s2)CC1'))