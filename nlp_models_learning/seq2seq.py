import random
import torch
import torch.nn as nn


class Seq2SeqTranslator:
    def __init__(self):
        # 数据初始化，定义中英文句子对
        self.sentences = [
            ['我 喜欢 吃 苹果', '<sos> I like eating apple', 'I like eating apple <eos>'],
            ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
            ['深度学习 改变 世界', '<sos> DL changed the world',
                'DL changed the world <eos>'],
            ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
            ['神经网络 非常 复杂', '<sos> Neural-Nets are complex',
                'Neural-Nets are complex <eos>']
        ]

        # 构建词汇表
        self._build_vocab()
        # 初始化模型组件
        self._init_model()

    def _build_vocab(self):
        """构建中英文词汇表"""
        # 中文词汇处理
        cn_words = set()
        en_words = {'<sos>', '<eos>'}  # 添加起始和结束标记

        # 遍历所有句子对，收集词汇
        for cn, en_in, en_out in self.sentences:
            cn_words.update(cn.split())  # 分割中文句子
            en_words.update(en_in.split() + en_out.split())  # 分割英文句子

        # 创建映射字典
        self.word2idx_cn = {w: i for i, w in enumerate(
            sorted(cn_words))}  # 中文词到索引
        self.word2idx_en = {w: i for i, w in enumerate(
            sorted(en_words))}  # 英文词到索引
        self.idx2word_en = {i: w for w,
                            i in self.word2idx_en.items()}  # 英文索引到词

        # 记录词汇表大小
        self.vocab_cn_size = len(self.word2idx_cn)
        self.vocab_en_size = len(self.word2idx_en)

    def _init_model(self):
        """初始化模型组件"""
        self.hidden_size = 128  # 隐藏层大小
        # 初始化编码器和解码器
        self.encoder = Encoder(self.vocab_cn_size, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.vocab_en_size)
        self.model = Seq2Seq(self.encoder, self.decoder)  # 组合编码器和解码器
        print(f"Seq2Seq Encoder:\n{self.encoder}")
        print(f"Seq2Seq Decoder:\n{self.decoder}")
        print(f"Seq2Seq Model:\n{self.model}")
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _make_batch(self):
        """创建训练批次数据"""
        src, _, tgt = random.choice(self.sentences)  # 随机选择一个句子对
        # 将中文句子转换为索引序列
        enc_input = torch.LongTensor(
            [[self.word2idx_cn[w] for w in src.split()]])
        # 将英文输入句子转换为索引序列
        dec_input = torch.LongTensor(
            [[self.word2idx_en[w] for w in _.split()]])
        # 将目标句子转换为索引序列
        target = torch.LongTensor([[self.word2idx_en[w] for w in tgt.split()]])
        return enc_input, dec_input, target

    def train(self, epochs=400):
        """模型训练"""
        criterion = nn.CrossEntropyLoss()  # 定义损失函数

        for epoch in range(epochs):
            enc_in, dec_in, tgt = self._make_batch()  # 获取一批数据
            hidden = torch.zeros(1, enc_in.size(
                0), self.hidden_size)  # 初始化隐藏状态

            self.optimizer.zero_grad()  # 梯度清零
            output = self.model(enc_in, hidden, dec_in)  # 前向传播
            loss = criterion(
                output.view(-1, self.vocab_en_size), tgt.view(-1))  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数

            if (epoch+1) % 40 == 0:
                # 打印训练进度
                print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    def translate(self, sentence: str):
        """执行翻译"""
        # 编码阶段
        enc_input = torch.LongTensor(
            [[self.word2idx_cn[w] for w in sentence.split()]])        # 输入句子转索引
        hidden = torch.zeros(1, enc_input.size(0), self.hidden_size)  # 初始化隐藏状态

        # 解码阶段
        dec_input = torch.LongTensor([[self.word2idx_en['<sos>']]])  # 起始标记
        result = []

        with torch.no_grad():
            encoder_out, hidden = self.model.encoder(
                enc_input, hidden)  # 编码器输出

            for _ in range(20):  # 最大生成长度
                output, hidden = self.model.decoder(dec_input, hidden)  # 解码器输出
                pred = output.argmax(2)               # 获取预测的词索引
                word = self.idx2word_en[pred.item()]  # 索引转词
                if word == '<eos>':  # 如果遇到结束标记，停止生成
                    break
                result.append(word)  # 添加到结果列表
                dec_input = pred     # 更新解码器输入

        return ' '.join(result)  # 返回翻译结果


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)         # 词嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # RNN层

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)            # 输入词嵌入
        output, hidden = self.rnn(embedded, hidden)  # RNN前向传播
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)        # 词嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # RNN层
        self.fc = nn.Linear(hidden_size, output_size)                  # 全连接层

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)            # 输入词嵌入
        output, hidden = self.rnn(embedded, hidden)  # RNN前向传播
        return self.fc(output), hidden               # 输出全连接层结果


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器

    def forward(self, enc_input, hidden, dec_input):
        encoder_out, hidden = self.encoder(enc_input, hidden)  # 编码器前向传播
        decoder_out, _ = self.decoder(dec_input, hidden)       # 解码器前向传播
        return decoder_out


if __name__ == '__main__':
    # 初始化翻译器
    translator = Seq2SeqTranslator()

    # 训练模型
    print("Start training...")
    translator.train()

    # 测试翻译
    test_cases = ['我 喜欢 吃 苹果', '自然 语言 处理 很 强大']
    print("\nTranslation:")
    for case in test_cases:
        translation = translator.translate(case)
        print(f"{case} -> {translation}")
