# filepath: [train_mhattention.py](http://_vscodecontentref_/3)
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# 使用torch.cuda而不是torch.cuda as cuda
import torch.backends.cudnn as cudnn
import argparse
import logging
import random
# 如果tensorboardX不可用，可以使用torch.utils.tensorboard
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
# 如果sklearn不可用，可以添加条件导入
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
except ImportError:
    print("Warning: sklearn or seaborn not available, some visualizations will be disabled")
import copy
# 使用F作为别名来解决警告
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='UCI_HAR', help='dataset_name')
parser.add_argument('--exp', type=str, default='mh_attention', help='exp_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--epoch', type=int, default='180', help='epoch number')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
args = parser.parse_args()

snapshot_path = "./model/exp_{}_epoch_{}/{}".format(args.exp, args.epoch, args.dataset_name)
model_save_path = "./model/exp_{}_epoch_{}/{}".format(args.exp, args.epoch, args.dataset_name)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cuda:" + str(1 - int(args.gpu)))
print(f"使用设备：{device}")

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

# 定义数据集类
class HumanActivityDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
SIGNALS = ["body_acc_x_", "body_acc_y_", "body_acc_z_",
           "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
           "total_acc_x_", "total_acc_y_", "total_acc_z_"]

train_paths = ['/home/jovyan/work/dpw/data/UCI_HAR_Dataset/train/Inertial Signals/' + signal + 'train.txt' for signal in SIGNALS]
test_paths = ['/home/jovyan/work/dpw/data/UCI_HAR_Dataset/test/Inertial Signals/' + signal + 'test.txt' for signal in SIGNALS]

def __load_X(X_signal_paths):
    X_signals = []

    for signal_type_path in X_signal_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

X_train = __load_X(train_paths)
X_test = __load_X(test_paths)

y_train = np.loadtxt('/home/jovyan/work/dpw/data/UCI_HAR_Dataset/train/y_train.txt',  dtype=np.int32)
y_test = np.loadtxt('/home/jovyan/work/dpw/data/UCI_HAR_Dataset/test/y_test.txt', dtype=np.int32)

# 确保标签值在 [0, num_classes-1] 范围内
y_train = y_train - 1
y_test = y_test - 1

# 创建数据集和数据加载器
train_dataset = HumanActivityDataset(X_train, y_train)
test_dataset = HumanActivityDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义具有SE块、多头注意力、知识蒸馏和增强激活函数的GRU模型
class EnhancedAttentionGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads=4, dropout_rate=0.2):
        super(EnhancedAttentionGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size * 2 // num_heads
        
        # 输入特征层归一化
        self.input_ln = nn.LayerNorm(input_size)
        
        # 投影层将输入映射到隐藏维度，用于残差连接
        self.input_proj = nn.Linear(input_size, hidden_size * 2)
        
        # GRU 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # SE Block (Squeeze and Excitation)
        self.se_block = SEBlock(hidden_size * 2)
        
        # 多头注意力层
        self.multi_head_attention = MultiHeadAttention(hidden_size * 2, num_heads, dropout_rate)
        self.ln_mha = nn.LayerNorm(hidden_size * 2)
        
        # 输出层
        self.ln_out = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln_fc1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # 残差连接的投影层
        self.res_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 知识蒸馏预测层 - 为教师模型提供更深层次的特征
        self.distill_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.distill_ln = nn.LayerNorm(hidden_size)
        self.distill_fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 应用输入层归一化
        x = self.input_ln(x)
        
        # 保存原始输入用于残差连接
        x_proj = self.input_proj(x)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # GRU 前向传播
        self.gru.flatten_parameters()  # 确保GRU权重在内存中连续存储
        gru_out, _ = self.gru(x, h0)  # gru_out: [batch_size, seq_len, hidden_size*2]
        
        # 残差连接 - 为序列中的每个时间步添加残差连接
        gru_out = gru_out + x_proj
        
        # 应用 SE Block 增强通道注意力
        gru_out = self.se_block(gru_out)
        
        # 应用多头注意力并获取注意力权重
        mha_out, attention_weights = self.multi_head_attention(gru_out, gru_out, gru_out)
        
        # 残差连接和层归一化
        gru_out = gru_out + mha_out  # 残差连接
        gru_out = self.ln_mha(gru_out)  # 层归一化
        
        # 使用全局池化获取序列表示（代替之前的注意力加权和）
        # 这里我们使用平均池化来获取序列的全局表示
        context = torch.mean(gru_out, dim=1)  # [batch_size, hidden_size*2]
        
        # 应用层归一化和 Dropout
        context = self.ln_out(context)
        context = self.dropout(context)
        
        # 知识蒸馏特征 - 用于教师模型
        distill_features = self.distill_proj(context)
        distill_features = self.distill_ln(distill_features)
        distill_features = F.leaky_relu(distill_features, negative_slope=0.01)
        distill_logits = self.distill_fc(distill_features)
        
        # 通过全连接层得到输出，使用leaky_relu代替relu
        out = self.fc1(context)
        
        # 残差连接 - 从上下文向量到第一个全连接层的输出
        res_context = self.res_proj(context)
        out = out + res_context
        
        out = self.ln_fc1(out)
        out = F.leaky_relu(out, negative_slope=0.01)  # 使用 Leaky ReLU 替代 ReLU
        out = self.dropout(out)
        out = self.fc2(out)
        
        # 返回: 输出, 多头注意力的平均权重, 原始多头注意力权重, 蒸馏预测
        # 计算多头注意力权重的平均值用于可视化
        avg_attention_weights = attention_weights.mean(dim=1)  # 多头平均
        
        return out, avg_attention_weights, attention_weights, distill_logits


# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, seq_len, channel]
        batch_size, seq_len, channel = x.size()
        # 转置以便在通道维度上进行池化
        x_perm = x.permute(0, 2, 1)  # [batch, channel, seq_len]
        y = self.avg_pool(x_perm).view(batch_size, channel)  # [batch, channel]
        y = self.fc(y).view(batch_size, channel, 1)  # [batch, channel, 1]
        # 应用通道注意力权重
        x_perm = x_perm * y.expand_as(x_perm)
        # 转置回原始形状
        return x_perm.permute(0, 2, 1)  # [batch, seq_len, channel]


# 多头注意力实现
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 线性投影并分割成多头
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力得分
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        out = torch.matmul(attention_weights, v)
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.out_proj(out)
        
        return out, attention_weights

# 知识蒸馏损失函数
def distillation_loss(student_logits, teacher_logits, T=2.0):
    """
    计算知识蒸馏损失，使用软目标
    
    参数:
    - student_logits: 学生模型的输出
    - teacher_logits: 教师模型的输出
    - T: 温度参数，控制软目标的平滑程度
    
    返回:
    - 蒸馏损失
    """
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (T * T)
    
def test():
    # 测试模型，输出6位浮点数
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    all_attention_weights = []  # 用于存储所有的注意力权重
    
    with torch.no_grad():
        with tqdm(test_loader, desc='Testing') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, attention_weights, _, _ = model(inputs)  # 现在模型返回四个值
                _, preds = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_attention_weights.append(attention_weights.cpu().numpy())  # 存储注意力权重
                
                # 更新进度条
                pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})
    
    try:
        # 计算并打印模型性能指标
        accuracy = 100 * correct / total
        report = classification_report(all_labels, all_preds, digits=6)
        conf_mat = confusion_matrix(all_labels, all_preds)

        # 结果保存成图片文件 /result
        df = pd.DataFrame(conf_mat, index=LABELS, columns=LABELS)
        df.to_csv(snapshot_path + '/confusion_matrix.csv')
        
        # 生成注意力机制热力图
        all_attention_weights = np.concatenate(all_attention_weights, axis=0)
        all_attention_weights = all_attention_weights[:len(all_labels)]
        
        # 重新整形注意力权重，减少时间步的密度
        # 将注意力权重矩阵按类别分组
        attention_by_class = {}
        for i, (prediction, label) in enumerate(zip(all_preds, all_labels)):
            if label not in attention_by_class:
                attention_by_class[label] = []
            attention_by_class[label].append(all_attention_weights[i])
        
        # 每个类别计算平均注意力
        avg_attention_by_class = {}
        for label, attn_weights in attention_by_class.items():
            avg_attention_by_class[label] = np.mean(attn_weights, axis=0)
        
        # 创建一个平均注意力矩阵，每行代表一个类别
        num_classes = len(LABELS)
        # 将时间步长从128采样到32（步长为4）
        time_length = all_attention_weights.shape[1]  # 获取实际时间步长度
        sampled_timesteps = list(range(0, time_length, 4))  # 每4步采样一次
        
        # 初始化正确大小的矩阵
        avg_attention_matrix = np.zeros((num_classes, len(sampled_timesteps)))
        
        for i in range(num_classes):
            if i in avg_attention_by_class:
                for j, timestep in enumerate(sampled_timesteps):
                    if timestep < avg_attention_by_class[i].shape[0]:
                        avg_attention_matrix[i, j] = avg_attention_by_class[i][timestep]
        
        # 绘制热力图
        plt.figure(figsize=(14, 8))
        heatmap = sns.heatmap(avg_attention_matrix, cmap='viridis', 
                       xticklabels=[f'{t}' for t in sampled_timesteps],
                       yticklabels=LABELS)
        
        # 添加标题和标签
        plt.title('Average Attention Weights by Activity Class (Time Step Sampled)')
        plt.xlabel('Time Steps (sampled every 4 steps)')
        plt.ylabel('Activity')
        
        # 调整y轴标签的字体大小
        plt.yticks(rotation=0, fontsize=10)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(snapshot_path + '/attention_weights_sampled.png')
        
        # 另外生成一个信号级别的注意力可视化
        # 为每个活动类找到最有代表性的样本（注意力分布与平均最接近的）
        representative_samples = {}
        for label in range(num_classes):
            if label in attention_by_class and len(attention_by_class[label]) > 0:
                avg_attn = avg_attention_by_class[label]
                best_sample_idx = 0
                min_dist = float('inf')
                
                for i, attn in enumerate(attention_by_class[label]):
                    dist = np.sum((attn - avg_attn) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_sample_idx = i
                
                # 创建正确大小的数组
                rep_sample = np.zeros(len(sampled_timesteps))
                for j, timestep in enumerate(sampled_timesteps):
                    if timestep < attention_by_class[label][best_sample_idx].shape[0]:
                        rep_sample[j] = attention_by_class[label][best_sample_idx][timestep]
                
                representative_samples[label] = rep_sample
        
        # 绘制代表性样本的热力图
        if representative_samples:
            plt.figure(figsize=(14, 8))
            rep_attn_matrix = np.array([representative_samples[i] if i in representative_samples else np.zeros(len(sampled_timesteps)) 
                                      for i in range(num_classes)])
            sns.heatmap(rep_attn_matrix, cmap='viridis', 
                      xticklabels=[f'{t}' for t in sampled_timesteps],
                      yticklabels=LABELS)
            plt.title('Representative Sample Attention Weights by Activity Class')
            plt.xlabel('Time Steps (sampled every 4 steps)')
            plt.ylabel('Activity')
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            plt.savefig(snapshot_path + '/representative_attention_weights.png')
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    return accuracy
    
if __name__ == "__main__":

    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(model_save_path + '/code')

    logging_handler = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[logging_handler])
    
    try:
        writer = SummaryWriter(snapshot_path + '/log')
        logging.info("{} iterations per epoch".format(len(train_loader)))
    except Exception as e:
        print(f"SummaryWriter initialization error: {str(e)}")
        writer = None

    # 初始化模型、损失函数和优化器
    input_size = X_train.shape[2]
    hidden_size = 128
    num_layers = 2
    num_classes = len(np.unique(y_train))
    num_heads = 4  # 多头注意力的头数

    # 创建增强版模型
    model = EnhancedAttentionGRUModel(input_size, hidden_size, num_layers, num_classes, num_heads)
    model.to(device)
    
    # 自蒸馏设置 - 复制当前模型作为教师模型
    teacher_model = None  # 稍后在训练期间更新
    use_distillation = True  # 是否使用知识蒸馏
    distill_alpha = 0.5  # 蒸馏损失权重
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    num_epochs = args.epoch
    best_model_state = None
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 每10个epoch，将当前最佳模型作为教师模型
        if epoch % 10 == 0 and epoch > 0 and use_distillation:
            teacher_model = copy.deepcopy(model)
            teacher_model.eval()
            logging.info(f"Updated teacher model at epoch {epoch}")
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=120) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
                optimizer.zero_grad()
                
                # 前向传播
                outputs, _, _, distill_outputs = model(inputs)
                
                # 计算分类损失
                cls_loss = criterion(outputs, labels)
                
                # 计算蒸馏损失
                distill_loss = 0.0
                if use_distillation and teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs, _, _, _ = teacher_model(inputs)
                    distill_loss = distillation_loss(distill_outputs, teacher_outputs)
                    total_loss = (1 - distill_alpha) * cls_loss + distill_alpha * distill_loss
                else:
                    total_loss = cls_loss
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪，避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                running_loss += total_loss.item()
                
                # 更新进度条信息
                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.4f}', 
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'distill_loss': f'{distill_loss:.4f}' if isinstance(distill_loss, float) else f'{distill_loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

        # Reduce learning rate
        scheduler.step()
        
        # Record loss and learning rate
        if writer is not None:
            writer.add_scalar('train_loss', running_loss/len(train_loader), epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        logging.info(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Evaluate model every 10 epochs
        if epoch >= 10 and epoch % 10 == 0:
            accuracy = test()
            if writer is not None:
                writer.add_scalar('test_accuracy', accuracy, epoch)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, model_save_path + '/best_model.pth')
                logging.info(f"New best model saved with accuracy: {best_accuracy:.6f}")
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.6f}')
    
    # Load best performing model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Save final model
    torch.save(model.state_dict(), model_save_path + '/final_model.pth')
    logging.info(f"Final model saved to {model_save_path}/final_model.pth")
    if writer is not None:
        writer.close()
    
    # Final test
    test()