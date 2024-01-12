import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import random
from transformers import AdamW
import pandas as pd
import json


# 分子数据集类，它能够处理分子并返回所需的特征
class MoleculeDataset(Dataset):
    def __init__(self, molecules):
        self.molecules = molecules

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):

        # 获取分子
        molecule = self.molecules[idx]
        mol = Chem.MolFromSmiles(molecule)

        if mol is None:
            raise ValueError(f"无法从SMILES中解析分子: {molecule}")

        # 获取分子的原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            # 添加更多想要获取的原子特征
            atom_features.append([
                atom.GetDegree(),
                atom.GetImplicitValence(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetIsAromatic(),
            ])

        # 生成分子的MACCS指纹
        maccs_keys = list(MACCSkeys.GenMACCSKeys(mol))

        # 生成分子的SMILES字符串
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

        # 随机化SMILES字符串
        randomized_smiles = self.randomize_smiles(smiles)

        return atom_features, maccs_keys, smiles, randomized_smiles

    def randomize_smiles(self, smiles):
        """此函数用于生成SMILES字符串的随机化版本。"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            randomized_smiles = Chem.MolToSmiles(mol, doRandom=True)
        else:
            randomized_smiles = smiles  # 如果无法解析原始smiles，返回未经处理的smiles

        return randomized_smiles


# 预训练模型，可以使用BERT或其他类似的transformer模型
class PretrainTransformer(nn.Module):
    def __init__(self, config):
        super(PretrainTransformer, self).__init__()
        self.bert_config = BertConfig()
        self.bert = BertModel(self.bert_config)
        self.atom_predictor = nn.Linear(self.bert_config.hidden_size, config['num_atom_features'])
        self.maccs_predictor = nn.Linear(self.bert_config.hidden_size, config['num_atom_features'])
        self.smiles_mlm = BertForMaskedLM(self.bert_config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 对输入过bert
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 拿到最后一层的特征
        sequence_output = outputs.last_hidden_state

        # 预测原子特征
        atom_feature_prediction = self.atom_predictor(sequence_output)

        # 预测MACCS指纹
        maccs_feature_prediction = self.maccs_predictor(sequence_output[:, 0, :])  # 可能仅使用CLS token的输出

        return atom_feature_prediction, maccs_feature_prediction, sequence_output


# 对比学习，可以定义一个简单的对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        sim = self.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        return -sim.logsumexp(dim=-1).mean()  # 对数和指数平均


# 主训练循环
def train_epoch(model,
                data_loader,
                optimizer,
                contrastive_loss,
                tokenizer,
                device):
    model.train()
    total_loss = 0

    for atom_features, maccs_fingerprint, smiles, randomized_smiles in data_loader:
        # 使用tokenizer批量处理SMILES字符串
        encoding = tokenizer(smiles,
                             padding=True,
                             truncation=True,
                             return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # 创建masked input ids
        mlm_labels = input_ids.clone()
        # 为了创建mask, 你需要根据策略创建mask的索引
        # 以下是简单的15%的随机mask
        rand = torch.rand(input_ids.shape).to(device)
        mask_arr = (rand < 0.15) * (input_ids != tokenizer.cls_token_id) * \
                   (input_ids != tokenizer.sep_token_id) * (input_ids != tokenizer.pad_token_id)
        selection = []

        for i in range(input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )

        for i in range(input_ids.shape[0]):
            input_ids[i, selection[i]] = tokenizer.mask_token_id

        optimizer.zero_grad()

        # 计算MLM损失
        mlm_output = model.smiles_mlm(input_ids=input_ids,
                                      labels=mlm_labels,
                                      attention_mask=attention_mask)
        mlm_loss = mlm_output.loss

        # 假定原子特征和MACCS指纹都已经被编码为类别的索引形式，如果是one-hot编码形式，需要进行转换
        atom_features_tensor = torch.tensor(atom_features).long().to(device)  # N x Atom_feature_size，其中N是批量大小
        maccs_fingerprint_tensor = torch.tensor(maccs_fingerprint).long().to(device)  # N x MACCS_size

        # 假定模型为每一种特征单独输出一个logits向量，我们这里举例使用nn.CrossEntropyLoss计算交叉熵损失
        cross_entropy_loss = nn.CrossEntropyLoss()

        # 执行model forward pass
        atom_feature_prediction, maccs_feature_prediction = model(atom_features_tensor, maccs_fingerprint_tensor)
        # 计算原子特征预测损失
        atom_loss = sum(cross_entropy_loss(atom_feature_prediction[:, i], atom_features_tensor[:, i]) for i in
                        range(atom_features_tensor.size(1)))
        # 计算MACCS指纹预测损失
        maccs_loss = cross_entropy_loss(maccs_feature_prediction, maccs_fingerprint_tensor)

        # 计算总损失
        total_loss = mlm_loss + atom_loss + maccs_loss + contrastive_loss
        # 反向传播和优化
        total_loss.backward()
        optimizer.step()

        total_loss += total_loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Training loss: {avg_loss}")
    return avg_loss
