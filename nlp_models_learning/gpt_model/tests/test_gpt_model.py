import json
import tiktoken
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.gpt_model import GPTModel
from transformers import GPT2Model
from tests.gpt_training import train_model


class GPTDataset(Dataset):
    """
    自定义数据集类, 用于将文本切分为输入和目标序列对

    参数:
        txt (str): 输入的原始文本
        tokenizer: 用于将文本编码为token ID的分词器
        max_length (int): 每个输入序列的最大长度
        stride (int): 滑动窗口的步长, 用于生成重叠的序列

    属性:
        input_ids (list): 存储输入序列的token ID列表
        target_ids (list): 存储目标序列的token ID列表
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 将文本编码为token ID序列
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 计算可用于滑动窗口的token长度
        token_length = len(token_ids) - max_length

        # 使用滑动窗口将文本切分为重叠的max_length长度的序列
        for i in range(0, token_length, stride):
            input_chunk = token_ids[i:i + max_length]           # 当前窗口的输入序列
            target_chunk = token_ids[i + 1:i + max_length + 1]  # 对应的目标序列(右移一位)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本

        参数:
            idx (int): 样本的索引

        返回:
            tuple: 包含输入序列和目标序列的张量对
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256,
                      stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    创建一个数据加载器, 用于加载和批处理文本数据

    参数:
        txt (str): 输入的原始文本
        batch_size (int, optional): 每个批次的样本数量, 默认为4
        max_length (int, optional): 每个输入序列的最大长度, 默认为256
        stride (int, optional): 滑动窗口的步长, 默认为128
        shuffle (bool, optional): 是否在每个epoch开始时打乱数据, 默认为True
        drop_last (bool, optional): 如果数据集大小不能被batch_size整除, 是否丢弃最后一个不完整的批次, 默认为True
        num_workers (int, optional): 用于数据加载的子进程数量, 默认为0

    返回:
        DataLoader: 用于加载和批处理数据的数据加载器
    """

    # 使用GPT2的tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 使用tokenizer分割txt为token
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


def load_gpt_config(config_file='params/gpt_params.json', model_name='GPT_CONFIG_124M'):
    """
    从 JSON 文件加载 GPT 配置

    参数:
        config_file (str): 包含 GPT 配置的 JSON 文件路径, 默认为 'gpt_params.json'
        model_name (str): 要加载的模型名称, 默认为 'GPT_CONFIG_124M'

    返回:
        dict: 包含指定模型名称的 GPT 配置字典
    """
    with open(config_file, 'r') as file:
        configs = json.load(file)
    return configs[model_name]


def plot_training_curves(train_losses, val_losses, tokens_seen, lrs):
    """
    绘制训练过程中的损失曲线、已见令牌数和学习率变化曲线

    参数:
        train_losses (list): 训练集的损失值列表
        val_losses (list): 验证集的损失值列表
        tokens_seen (list): 每个 epoch 后模型已见过的令牌数量
        lrs (list): 每个 epoch 的学习率列表
    """
    # 创建一个包含两个子图的图形对象,  figsize 设置图形大小
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 第一个子图: 绘制训练和验证损失曲线
    # 绘制训练损失曲线, 使用圆形标记
    ax1.plot(train_losses, label='Train', marker='o')
    # 绘制验证损失曲线, 使用叉形标记
    ax1.plot(val_losses, label='Validation', marker='x')
    # 设置子图标题
    ax1.set_title('Training Progress')
    # 设置 y 轴标签为 "Loss"
    ax1.set_ylabel('Loss')
    # 设置 x 轴标签为 "Epoch"
    ax1.set_xlabel('Epoch')
    # 显示网格线
    ax1.grid(True)
    # 显示图例
    ax1.legend()

    # 第二个子图: 绘制已见令牌数和学习率变化曲线
    # 绘制令牌数曲线, 颜色为蓝色, 使用圆形标记
    ax2.plot(tokens_seen, color='tab:blue', marker='o')
    # 设置 x 轴标签为 "Epoch"
    ax2.set_xlabel('Epoch')
    # 设置 y 轴标签为 "Tokens Seen", 颜色为蓝色
    ax2.set_ylabel('Tokens Seen', color='tab:blue')
    # 设置 y 轴刻度颜色为蓝色
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 双 Y 轴: 在同一子图中绘制学习率变化曲线
    # 创建第二个 Y 轴
    ax2_lr = ax2.twinx()
    # 绘制学习率曲线, 颜色为红色, 使用虚线和三角形标记
    ax2_lr.plot(lrs, color='tab:red', linestyle='--',
                marker='^')
    # 设置第二个 Y 轴标签为 "Learning Rate", 颜色为红色
    ax2_lr.set_ylabel('Learning Rate', color='tab:red')
    # 设置第二个 Y 轴刻度颜色为红色
    ax2_lr.tick_params(axis='y', labelcolor='tab:red')

    # 调整布局以避免重叠
    plt.tight_layout()

    # 显示图形
    plt.show()


def main():
    # 从 JSON 文件中加载 GPT 配置
    gpt_config = load_gpt_config()

    # 检测是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子以确保结果可重复
    torch.manual_seed(123)

    # 定义可用的 GPT 模型及其对应的预训练模型名称
    model_names = {
        "gpt2-small (124M)": "openai-community/gpt2",          # 小型 GPT-2 模型
        "gpt2-medium (355M)": "openai-community/gpt2-medium",  # 中型 GPT-2 模型
        "gpt2-large (774M)": "openai-community/gpt2-large",    # 大型 GPT-2 模型
        "gpt2-xl (1558M)": "openai-community/gpt2-xl"          # 超大型 GPT-2 模型
    }

    # 当前选择小型 GPT-2 模型
    CHOOSE_MODEL = "gpt2-small (124M)"

    # 从 Hugging Face 加载预训练的 GPT 模型, 并将其设置为评估模式
    gpt_hf = GPT2Model.from_pretrained(
        model_names[CHOOSE_MODEL], cache_dir="checkpoints")
    # 设置模型为评估模式
    gpt_hf.eval()

    model = GPTModel(gpt_config)
    # 将 Hugging Face 模型的权重加载到自定义模型中
    model.load_weights(gpt_config, gpt_hf)
    # 设置自定义模型为评估模式
    model.eval()

    # 读取训练数据集文本文件
    with open("datasets/the-verdict.txt", "r", encoding="utf-8") as file:
        text_data = file.read()

    # 训练集占 80%, 验证集占 20%
    train_ratio = 0.80
    split_idx = int(train_ratio * len(text_data))

    # 创建训练数据加载器
    train_loader = create_dataloader(
        text_data[:split_idx],                    # 使用前 80% 数据作为训练集
        batch_size=2,                             # 每个批次包含 2 个样本
        max_length=gpt_config["context_length"],  # 每个输入序列的最大长度
        stride=gpt_config["context_length"],      # 滑动窗口步长
        drop_last=True,                           # 如果最后一个批次不完整, 则丢弃
        shuffle=True,                             # 每个 epoch 开始时打乱数据
        num_workers=0                             # 不使用多线程加载数据
    )

    # 创建验证数据加载器
    val_loader = create_dataloader(
        text_data[split_idx:],                    # 使用后 20% 数据作为验证集
        batch_size=2,                             # 每个批次包含 2 个样本
        max_length=gpt_config["context_length"],  # 每个输入序列的最大长度
        stride=gpt_config["context_length"],      # 滑动窗口步长
        drop_last=False,                          # 不丢弃最后一个不完整的批次
        shuffle=False,                            # 验证集不需要打乱数据
        num_workers=0                             # 不使用多线程加载数据
    )

    # 最大学习率
    peak_lr = 0.001
    # 总共训练 10 个 epoch
    n_epochs = 10
    # 总训练步数
    total_steps = len(train_loader) * n_epochs
    # 前 20% 的步数用于学习率预热
    warmup_steps = int(0.2 * total_steps)

    # 使用 AdamW 优化器, 设置初始学习率和权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=0.1)

    # 使用 GPT-2 的分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 训练模型
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
        eval_freq=5,                              # 每 5 个 step 进行一次验证
        eval_iter=1,                              # 每次验证生成 1 个样本
        start_context="Every effort moves you",   # 生成文本的起始上下文
        tokenizer=tokenizer,                      # 使用的分词器
        warmup_steps=warmup_steps,                # 学习率预热步数
        initial_lr=1e-5,                          # 初始学习率
        min_lr=1e-5                               # 最小学习率
    )

    # 绘制训练过程中的损失曲线、已见Token数和学习率变化曲线
    plot_training_curves(train_losses, val_losses, tokens_seen, lrs)


if __name__ == "__main__":
    main()
