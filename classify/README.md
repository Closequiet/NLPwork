# 文本分类任务书
## 代码核心功能说明

### 一、算法基础：多项式朴素贝叶斯分类器
1. 核心假设
 - 特征条件独立性：假设文本中每个词的出现概率相互独立（强假设），即词序和上下文不影响分类。
 - 多项式分布：建模词频统计，适用于离散型特征（如词出现次数）。

2. 贝叶斯定理应用形式
 - 对于邮件分类（垃圾邮件/正常邮件）：
$$
P(y|X) = \frac{P(X|y)P(y)}{P(X)} \quad \text{其中} \ X=(x_1,x_2,...,x_n)
$$ 
 - 具体展开：
$$
P(y|X) \propto P(y) \prod_{i=1}^n P(x_i|y)
$$
 - 分类决策：
$\hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i|y)$
3. 参数估计
 - 先验概率：P(y)为各类别在训练集中的比例。
 - 条件概率（平滑处理版）：
$$
P(x_i|y) = \frac{\text{count}(x_i, y) + \alpha}{\sum_{x \in V} (\text{count}(x, y) + \alpha)}
$$
- $\alpha$: 平滑系数  
- $V$: 词汇表 

### 二、数据处理流程
1. 分词处理
 - 实现逻辑：
 ``` python
 import jieba  
 def tokenize(text):
    return list(jieba.cut(text))  
```
2. 停用词过滤
 - 实现逻辑：
 ``` python
def remove_stopwords(tokens, stopwords):
    return [t for t in tokens if t not in stopwords]
```
 - 停用词来源：预定义列表（如中文"的、了"；英文"the, a"）或基于频率动态剔除。
3. 其他预处理
 - 标准化：统一转为小写```（text.lower()）```。
 - 正则过滤：移除标点、数字```（re.sub(r'[^a-zA-Z]', ' ', text)）```。
### 三、特征构建方法对比
1. 高频词特征选择
 - 数学表达：
   + 选择训练集中出现频率最高的前$k$个词作为特征。
   + 特征向量：$X∈R^k$值为词频（或二进制出现与否）。
 - 实现差异
 ``` python
from collections import Counter
vocab = [word for word, _ in Counter(all_words).most_common(k)]
``` 
2. TF-IDF加权
 - 数学表达：
   + 词频（TF）：
   $$
   \text{tf}(t, d) = \frac{\text{count}(t, d)}{\text{len}(d)}
   $$

   + 逆文档频率（IDF）
   $$
   \text{idf}(t) = \log \frac{N}{1 + \text{df}(t)}
   $$
   + $N$: 总文档数  
   + $\text{df}(t)$: 包含词$t$的文档数 
   + 最终权重
   $$
   \text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
   $$
 - 实现差异：
 ``` python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=k)
X = tfidf.fit_transform(corpus)
 ```
3. 对比分析

| 维度    | 	高频词选择  | TF-IDF          |
|:------|:-------:|-----------------|
| 特征权重  | 词频或二进制	 | 词频×逆文档频率（抑制常见词） |
| 计算效率  | 高（只需计数） | 较低（需全局统计）       |
| 适用场景  | 简单快速任务  | 长文本或词汇重要性差异大的场景 |

### 四、关键问题思考
1. 独立性假设的局限性：实际语言中词之间存在关联，可通过引入n-gram特征部分缓解。
2. TF-IDF的优势：通过IDF降低高频但无信息量词的权重（如"公司"在商务邮件中）。
3. 平滑的重要性：未登录词（测试集新词）通过平滑参数α分配非零概率。

### 五、代码截图
<img src="image_readme/4.png" alt="图片描述" width = "800" height = "图片长度" />