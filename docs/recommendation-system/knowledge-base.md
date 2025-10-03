# 推荐系统核心知识体系

## 📚 知识图谱概览

```
推荐系统知识体系：
┌─────────────────────────────────────────────────────────┐
│                   🎯 推荐系统核心理论                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │   基础理论   │   算法原理   │   工程实践   │   评估体系   │ │
│  │  • 协同过滤  │  • 相似度   │  • 架构设计  │  • 准确性   │ │
│  │  • 内容推荐  │  • 矩阵分解  │  • 性能优化  │  • 多样性   │ │
│  │  • 深度学习  │  • 因子分解  │  • 分布式   │  • 新颖性   │ │
│  │  • 多臂赌博  │  • 图神经网络│  • 实时计算  │  • 商业指标 │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🎓 基础理论知识

### 1. 推荐系统分类

#### 协同过滤 (Collaborative Filtering)
```python
"""
协同过滤核心思想：
- 基于用户的相似行为进行推荐
- 不需要物品的内容信息
- 存在冷启动和稀疏性问题
"""

# 用户相似度计算
def user_similarity(user_a_ratings, user_b_ratings):
    """用户相似度计算的核心公式"""
    common_items = set(user_a_ratings.keys()) & set(user_b_ratings.keys())

    if not common_items:
        return 0.0

    # 皮尔逊相关系数
    mean_a = np.mean([user_a_ratings[item] for item in common_items])
    mean_b = np.mean([user_b_ratings[item] for item in common_items])

    numerator = sum((user_a_ratings[item] - mean_a) *
                   (user_b_ratings[item] - mean_b)
                   for item in common_items)

    sum_sq_a = sum((user_a_ratings[item] - mean_a) ** 2
                  for item in common_items)
    sum_sq_b = sum((user_b_ratings[item] - mean_b) ** 2
                  for item in common_items)

    denominator = np.sqrt(sum_sq_a * sum_sq_b)

    return numerator / denominator if denominator != 0 else 0.0
```

#### 内容推荐 (Content-Based Recommendation)
```python
"""
内容推荐核心思想：
- 基于物品特征和用户历史偏好
- 需要提取物品的内容特征
- 可以解决冷启动问题
"""

# TF-IDF算法
def tfidf_score(term, document, corpus):
    """TF-IDF计算"""
    # 词频 (TF)
    tf = document.count(term) / len(document)

    # 逆文档频率 (IDF)
    doc_count = sum(1 for doc in corpus if term in doc)
    idf = np.log(len(corpus) / (doc_count + 1))

    return tf * idf

# 余弦相似度
def cosine_similarity(vector_a, vector_b):
    """余弦相似度计算"""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
```

### 2. 矩阵分解 (Matrix Factorization)

#### 基础矩阵分解
```python
"""
矩阵分解数学原理：
- 将用户-物品评分矩阵分解为用户特征矩阵和物品特征矩阵
- R ≈ P × Q^T
- P: 用户特征矩阵 (m × k)
- Q: 物品特征矩阵 (n × k)
- k: 隐因子数量
"""

class BasicMatrixFactorization:
    """基础矩阵分解实现"""

    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self, ratings_matrix, n_epochs=100):
        """训练矩阵分解模型"""
        n_users, n_items = ratings_matrix.shape

        # 初始化特征矩阵
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # 训练过程
        for epoch in range(n_epochs):
            for u, i, r in self._get_ratings(ratings_matrix):
                # 预测评分
                prediction = np.dot(self.P[u], self.Q[i])

                # 计算误差
                error = r - prediction

                # 梯度下降更新
                self.P[u] += self.learning_rate * (error * self.Q[i] -
                                                 self.regularization * self.P[u])
                self.Q[i] += self.learning_rate * (error * self.P[u] -
                                                 self.regularization * self.Q[i])

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        return np.dot(self.P[user_id], self.Q[item_id])
```

#### SVD++算法
```python
class SVDPlusPlus:
    """SVD++算法实现"""

    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self, ratings_matrix, n_epochs=20):
        """SVD++训练"""
        n_users, n_items = ratings_matrix.shape

        # 初始化参数
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = np.mean(ratings_matrix[ratings_matrix > 0])

        # 隐式反馈参数
        self.implicit_factors = np.random.normal(0, 0.1,
                                               (n_items, self.n_factors))

        for epoch in range(n_epochs):
            for u, i, r in self._get_ratings(ratings_matrix):
                # 获取用户显式和隐式反馈
                explicit_items = self._get_user_items(u, ratings_matrix)

                # 计算隐式反馈向量
                implicit_sum = np.zeros(self.n_factors)
                for j in explicit_items:
                    implicit_sum += self.implicit_factors[j]

                implicit_sum /= np.sqrt(len(explicit_items))

                # 预测评分
                prediction = (self.global_bias + self.user_bias[u] +
                            self.item_bias[i] +
                            np.dot(self.user_factors[u], self.item_factors[i]) +
                            np.dot(self.user_factors[u], implicit_sum))

                # 计算误差并更新参数
                error = r - prediction
                self._update_parameters(u, i, error, explicit_items, implicit_sum)
```

### 3. 深度学习推荐算法

#### Neural Collaborative Filtering (NCF)
```python
"""
NCF架构设计：
- GMF部分：广义矩阵分解
- MLP部分：多层感知机
- 融合层：结合GMF和MLP的输出
"""

import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    """神经协同过滤网络"""

    def __init__(self, n_users, n_items, embedding_dim=64, hidden_layers=[128, 64]):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # GMF部分
        self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)

        # MLP部分
        mlp_input_dim = embedding_dim * 2
        mlp_layers = []
        current_dim = mlp_input_dim

        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # 融合层
        self.fusion = nn.Linear(embedding_dim + current_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        # GMF部分
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item

        # MLP部分
        mlp_user = self.user_embedding(user_ids)
        mlp_item = self.item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_input)

        # 融合预测
        fusion_input = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.fusion(fusion_input)
        return self.sigmoid(output)
```

#### DeepFM模型
```python
"""
DeepFM架构：
- FM部分：因子分解机，捕获低阶特征交互
- Deep部分：深度神经网络，捕获高阶特征交互
- 共享输入：相同的特征嵌入
"""

class DeepFM(nn.Module):
    """Deep Factorization Machine"""

    def __init__(self, feature_dims, embedding_dim=8, hidden_dims=[128, 64]):
        super().__init__()
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim

        # 特征嵌入
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in feature_dims
        ])

        # FM一阶项
        self.fm_first_order = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in feature_dims
        ])

        # 深度网络
        deep_input_dim = len(feature_dims) * embedding_dim
        deep_layers = []
        current_dim = deep_input_dim

        for hidden_dim in hidden_dims:
            deep_layers.append(nn.Linear(current_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim

        self.deep = nn.Sequential(*deep_layers)

        # 输出层
        self.output_layer = nn.Linear(current_dim + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_indices):
        # 获取嵌入
        embeddings = [emb(indices) for emb, indices in
                     zip(self.embeddings, feature_indices)]

        # FM一阶项
        first_order = [emb(indices) for emb, indices in
                      zip(self.fm_first_order, feature_indices)]
        first_order_sum = sum(first_order)

        # FM二阶项
        embeddings_stack = torch.stack(embeddings, dim=1)
        square_of_sum = torch.sum(embeddings_stack, dim=1) ** 2
        sum_of_square = torch.sum(embeddings_stack ** 2, dim=1)
        fm_second_order = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        # 深度网络
        deep_input = torch.cat(embeddings, dim=1)
        deep_output = self.deep(deep_input)

        # 组合输出
        output_input = torch.cat([first_order_sum, fm_second_order, deep_output], dim=1)
        output = self.output_layer(output_input)

        return self.sigmoid(output)
```

### 4. 图神经网络推荐

#### GraphSAGE在推荐中的应用
```python
"""
图神经网络在推荐系统中的应用：
- 用户-物品二部图建模
- 节点嵌入学习
- 邻居聚合和传播
"""

import dgl
import dgl.nn as dglnn

class GraphSAGERecommender(nn.Module):
    """基于GraphSAGE的推荐模型"""

    def __init__(self, in_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()

        # GraphSAGE层
        self.layers.append(dglnn.SAGEConv(in_dim, hidden_dim, 'mean'))
        for _ in range(n_layers - 1):
            self.layers.append(dglnn.SAGEConv(hidden_dim, hidden_dim, 'mean'))

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, graph, features, user_nodes, item_nodes):
        # 图卷积
        h = features
        for layer in self.layers:
            h = layer(graph, h)

        # 获取用户和物品嵌入
        user_embeddings = h[user_nodes]
        item_embeddings = h[item_nodes]

        # 预测交互概率
        user_item_pairs = torch.cat([user_embeddings, item_embeddings], dim=1)
        scores = self.predictor(user_item_pairs)

        return scores
```

## 🔧 机器学习基础

### 1. 相似度计算方法

```python
class SimilarityMetrics:
    """相似度计算方法集合"""

    @staticmethod
    def euclidean_distance(vector_a, vector_b):
        """欧氏距离"""
        return np.linalg.norm(vector_a - vector_b)

    @staticmethod
    def manhattan_distance(vector_a, vector_b):
        """曼哈顿距离"""
        return np.sum(np.abs(vector_a - vector_b))

    @staticmethod
    def jaccard_similarity(set_a, set_b):
        """Jaccard相似度"""
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union != 0 else 0.0

    @staticmethod
    def pearson_correlation(x, y):
        """皮尔逊相关系数"""
        if len(x) != len(y):
            raise ValueError("向量长度必须相等")

        n = len(x)
        if n == 0:
            return 0.0

        mean_x, mean_y = np.mean(x), np.mean(y)

        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        correlation_matrix = np.corrcoef(x, y)
        return correlation_matrix[0, 1]

    @staticmethod
    def spearman_correlation(x, y):
        """斯皮尔曼相关系数"""
        if len(x) != len(y):
            raise ValueError("向量长度必须相等")

        # 转换为秩
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))

        # 计算皮尔逊相关系数
        return SimilarityMetrics.pearson_correlation(rank_x, rank_y)
```

### 2. 降维技术

#### PCA (主成分分析)
```python
class PCA:
    """主成分分析实现"""

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        """训练PCA模型"""
        # 中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)

        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 按特征值排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]
        self.explained_variance = eigenvalues[sorted_indices[:self.n_components]]

    def transform(self, X):
        """降维变换"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """训练并变换"""
        self.fit(X)
        return self.transform(X)
```

#### SVD (奇异值分解)
```python
class TruncatedSVD:
    """截断奇异值分解"""

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.singular_values = None
        self.explained_variance = None

    def fit_transform(self, X):
        """训练并变换"""
        # 执行SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # 截断到指定组件数
        self.components = Vt[:self.n_components].T
        self.singular_values = S[:self.n_components]

        # 计算解释方差
        self.explained_variance = (S[:self.n_components] ** 2) / (X.shape[0] - 1)

        # 返回降维后的数据
        X_reduced = U[:, :self.n_components] * S[:self.n_components]

        return X_reduced
```

### 3. 聚类算法

#### K-Means聚类
```python
class KMeans:
    """K-Means聚类算法"""

    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """训练K-Means模型"""
        n_samples, n_features = X.shape

        # 初始化质心
        if self.random_state:
            np.random.seed(self.random_state)

        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # 分配样本到最近的质心
            distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)

            # 更新质心
            new_centroids = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[i] = self.centroids[i]

            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        """预测样本的聚类标签"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)
```

## 📊 统计学基础

### 1. 概率分布

```python
class ProbabilityDistributions:
    """常见概率分布"""

    @staticmethod
    def normal_distribution(x, mu=0, sigma=1):
        """正态分布概率密度函数"""
        return (1 / (sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    @staticmethod
    def binomial_distribution(k, n, p):
        """二项分布概率质量函数"""
        from math import comb
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    @staticmethod
    def poisson_distribution(k, lam):
        """泊松分布概率质量函数"""
        return (lam ** k * np.exp(-lam)) / np.math.factorial(k)
```

### 2. 假设检验

```python
class HypothesisTesting:
    """假设检验方法"""

    @staticmethod
    def t_test(sample1, sample2, alpha=0.05):
        """独立样本t检验"""
        from scipy import stats

        t_statistic, p_value = stats.ttest_ind(sample1, sample2)

        result = {
            't_statistic': t_statistic,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'confidence_level': 1 - alpha
        }

        return result

    @staticmethod
    def chi_squared_test(observed, expected, alpha=0.05):
        """卡方检验"""
        from scipy import stats

        chi2_statistic, p_value = stats.chisquare(observed, expected)

        result = {
            'chi2_statistic': chi2_statistic,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'degrees_of_freedom': len(observed) - 1
        }

        return result
```

### 3. 相关性分析

```python
class CorrelationAnalysis:
    """相关性分析工具"""

    @staticmethod
    def correlation_matrix(data):
        """计算相关性矩阵"""
        return np.corrcoef(data.T)

    @staticmethod
    def partial_correlation(x, y, control_vars):
        """偏相关系数"""
        from scipy import stats

        # 计算残差
        def residual_regression(y, X):
            X = np.column_stack([np.ones(len(X)), X])
            coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
            predicted = X @ coefficients
            return y - predicted

        # 控制变量的残差
        x_residual = residual_regression(x, control_vars)
        y_residual = residual_regression(y, control_vars)

        # 计算残差的相关性
        correlation, _ = stats.pearsonr(x_residual, y_residual)

        return correlation
```

## 🎯 优化算法

### 1. 梯度下降

```python
class GradientDescent:
    """梯度下降优化算法"""

    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def minimize(self, objective_function, gradient_function, initial_params):
        """最小化目标函数"""
        params = np.array(initial_params, dtype=float)

        for iteration in range(self.max_iterations):
            # 计算梯度
            gradient = gradient_function(params)

            # 更新参数
            new_params = params - self.learning_rate * gradient

            # 检查收敛
            if np.linalg.norm(new_params - params) < self.tolerance:
                break

            params = new_params

        return params, iteration

    def stochastic_minimize(self, objective_function, gradient_function,
                           initial_params, data, batch_size=32):
        """随机梯度下降"""
        params = np.array(initial_params, dtype=float)
        n_samples = len(data)

        for iteration in range(self.max_iterations):
            # 随机采样
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_data = data[batch_indices]

            # 计算批量梯度
            gradient = gradient_function(params, batch_data)

            # 更新参数
            params = params - self.learning_rate * gradient

            # 学习率衰减
            self.learning_rate *= 0.999

        return params
```

### 2. 遗传算法

```python
class GeneticAlgorithm:
    """遗传算法优化"""

    def __init__(self, population_size=100, mutation_rate=0.01,
                 crossover_rate=0.8, elitism_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate

    def optimize(self, fitness_function, chromosome_length, generations=100):
        """遗传算法优化"""
        # 初始化种群
        population = np.random.rand(self.population_size, chromosome_length)

        for generation in range(generations):
            # 计算适应度
            fitness_scores = np.array([fitness_function(chromosome)
                                     for chromosome in population])

            # 选择
            selected_indices = self._selection(fitness_scores)
            selected_population = population[selected_indices]

            # 交叉
            offspring = self._crossover(selected_population)

            # 变异
            offspring = self._mutation(offspring)

            # 精英保留
            elite_size = int(self.population_size * self.elitism_rate)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elite = population[elite_indices]

            # 新种群
            population = np.vstack([elite, offspring[:self.population_size - elite_size]])

        # 返回最优解
        final_fitness = np.array([fitness_function(chromosome)
                                 for chromosome in population])
        best_index = np.argmax(final_fitness)

        return population[best_index], final_fitness[best_index]

    def _selection(self, fitness_scores):
        """轮盘赌选择"""
        probabilities = fitness_scores / np.sum(fitness_scores)
        selected_indices = np.random.choice(
            len(fitness_scores),
            size=self.population_size,
            p=probabilities
        )
        return selected_indices

    def _crossover(self, population):
        """单点交叉"""
        offspring = []
        for i in range(0, len(population) - 1, 2):
            parent1, parent2 = population[i], population[i + 1]

            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, len(parent1))
                child1 = np.concatenate([parent1[:crossover_point],
                                        parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point],
                                        parent1[crossover_point:]])
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        return np.array(offspring)

    def _mutation(self, population):
        """随机变异"""
        for chromosome in population:
            for i in range(len(chromosome)):
                if np.random.rand() < self.mutation_rate:
                    chromosome[i] = np.random.rand()
        return population
```

## 🔍 评估指标体系

### 1. 分类指标

```python
class ClassificationMetrics:
    """分类评估指标"""

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """混淆矩阵"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        cm = np.zeros((n_classes, n_classes), dtype=int)

        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

        return cm

    @staticmethod
    def precision_recall_f1(y_true, y_pred):
        """精确率、召回率、F1分数"""
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)

        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    @staticmethod
    def roc_auc_score(y_true, y_scores):
        """ROC-AUC分数"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_scores)
```

### 2. 回归指标

```python
class RegressionMetrics:
    """回归评估指标"""

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """均方误差"""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """平均绝对误差"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2_score(y_true, y_pred):
        """R²决定系数"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
```

### 3. 推荐系统专用指标

```python
class RecommendationMetrics:
    """推荐系统专用评估指标"""

    @staticmethod
    def hit_rate_at_k(recommended, relevant, k):
        """Hit Rate @ K"""
        recommended_k = recommended[:k]
        return int(any(item in relevant for item in recommended_k))

    @staticmethod
    def mean_reciprocal_rank(recommended, relevant):
        """平均倒数排名"""
        for i, item in enumerate(recommended, 1):
            if item in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def coverage(all_items, recommended_items):
        """覆盖率"""
        recommended_set = set(recommended_items)
        return len(recommended_set) / len(all_items)

    @staticmethod
    def serendipity(recommended, expected, item_similarity_matrix):
        """意外性/新颖性"""
        serendipity_scores = []

        for item in recommended:
            if item not in expected:
                # 计算与期望物品的平均相似度
                similarities = []
                for expected_item in expected:
                    if item in item_similarity_matrix and expected_item in item_similarity_matrix[item]:
                        similarities.append(item_similarity_matrix[item][expected_item])

                if similarities:
                    avg_similarity = np.mean(similarities)
                    serendipity_scores.append(1 - avg_similarity)

        return np.mean(serendipity_scores) if serendipity_scores else 0.0
```

这套知识体系为推荐系统的学习和实践提供了坚实的理论基础，涵盖了从基础概念到高级算法的完整知识框架。