# RNN
## Surname-Classification-with-RNNs截图
<img src="https://github.com/Closequiet/NLPwork/blob/140b3a8aa1da6c54ee074cf46447631faef673c5/image_readme/8.png" alt="图片描述" width = "800" height = "图片长度" />

## Model1_Unconditioned_Surname_Generation截图
<img src="https://github.com/Closequiet/NLPwork/blob/140b3a8aa1da6c54ee074cf46447631faef673c5/image_readme/9.png" alt="图片描述" width = "800" height = "图片长度" />

## Model2_Conditioned_Surname_Generation截图
<img src="https://github.com/Closequiet/NLPwork/blob/140b3a8aa1da6c54ee074cf46447631faef673c5/image_readme/10.png" alt="图片描述" width = "800" height = "图片长度" />

## 题目
①两个模型的核心差异体现在什么机制上？

B. 是否考虑国家信息作为生成条件


② 在条件生成模型中，国家信息通过什么方式影响生成过程？

B. 作为GRU的初始隐藏状态


③ 文件2中新增的nation_emb层的主要作用是：

B. 将国家标签转换为隐藏状态初始化向量


④ 对比两个文件的sample_from_model函数，文件2新增了哪个关键参数？

B. nationalities