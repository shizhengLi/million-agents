# 推荐算法核心实现

## 🧮 算法体系概览

本推荐系统实现了四大核心算法引擎，采用TDD方法论确保算法的准确性和可靠性：

```
推荐算法体系架构：
┌─────────────────────────────────────────────────────────┐
│                🎯 混合推荐引擎                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬─────────────┐           │
│  │协同过滤引擎  │内容推荐引擎  │社交推荐引擎  │           │
│  │  • 用户相似  │  • TF-IDF   │  • 影响力   │           │
│  │  • 物品相似  │  • 特征提取  │  • 信任传播  │           │
│  │  • 矩阵分解  │  • 相似度计算│  • 社交路径  │           │
│  └─────────────┴─────────────┴─────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## 🤝 协同过滤算法 (Collaborative Filtering)

### 算法原理

协同过滤基于"物以类聚，人以群分"的思想，通过分析用户的历史行为和偏好，找到相似的用户或物品，然后基于相似性进行推荐。

### 核心实现

#### 1. 用户相似度计算

```python
def calculate_user_similarity(self, user_a: str, user_b: str,
                             method: str = "cosine") -> float:
    """
    计算用户相似度

    Args:
        user_a, user_b: 用户ID
        method: 相似度计算方法 (cosine, pearson, jaccard)

    Returns:
        float: 相似度分数 [0, 1]
    """
    # 获取两个用户的共同物品
    common_items = self.get_common_items(user_a, user_b)
    if not common_items:
        return 0.0

    if method == "cosine":
        return self._cosine_similarity(user_a, user_b, common_items)
    elif method == "pearson":
        return self._pearson_correlation(user_a, user_b, common_items)
    elif method == "jaccard":
        return self._jaccard_similarity(user_a, user_b)
    else:
        raise ValueError(f"不支持的相似度计算方法: {method}")
```

#### 2. 余弦相似度实现

```python
def _cosine_similarity(self, user_a: str, user_b: str,
                      common_items: Set[str]) -> float:
    """余弦相似度计算"""

    # 构建向量
    vector_a = np.array([self.matrix[user_a][item] for item in common_items])
    vector_b = np.array([self.matrix[user_b][item] for item in common_items])

    # 计算余弦相似度
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)
```

#### 3. 皮尔逊相关系数

```python
def _pearson_correlation(self, user_a: str, user_b: str,
                        common_items: Set[str]) -> float:
    """皮尔逊相关系数计算"""

    ratings_a = [self.matrix[user_a][item] for item in common_items]
    ratings_b = [self.matrix[user_b][item] for item in common_items]

    mean_a = np.mean(ratings_a)
    mean_b = np.mean(ratings_b)

    numerator = sum((ra - mean_a) * (rb - mean_b)
                   for ra, rb in zip(ratings_a, ratings_b))

    sum_sq_a = sum((ra - mean_a) ** 2 for ra in ratings_a)
    sum_sq_b = sum((rb - mean_b) ** 2 for rb in ratings_b)

    denominator = np.sqrt(sum_sq_a * sum_sq_b)

    if denominator == 0:
        return 0.0

    return numerator / denominator
```

#### 4. 基于用户的推荐生成

```python
def user_based_recommend(self, user_id: str, k: int = 10) -> List[RecommendationItem]:
    """
    基于用户的协同过滤推荐

    算法步骤：
    1. 找到与目标用户最相似的N个用户
    2. 获取这些相似用户评分高但目标用户未评分的物品
    3. 根据相似度和评分计算推荐分数
    """

    # 1. 计算用户相似度
    similarities = []
    for other_user in self.matrix.users:
        if other_user != user_id:
            sim = self.calculate_user_similarity(user_id, other_user)
            if sim > 0:
                similarities.append((other_user, sim))

    # 2. 排序并选择TopN相似用户
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_users = similarities[:self.n_similar_users]

    # 3. 生成推荐
    recommendations = {}
    user_items = set(self.matrix[user_id].keys())

    for similar_user, similarity in top_similar_users:
        for item, rating in self.matrix[similar_user].items():
            if item not in user_items and rating > 0:
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += similarity * rating

    # 4. 归一化并排序
    normalized_recommendations = []
    for item, score in recommendations.items():
        normalized_score = min(score / len(top_similar_users), 1.0)
        normalized_recommendations.append(
            RecommendationItem(item, normalized_score)
        )

    normalized_recommendations.sort(key=lambda x: x.score, reverse=True)
    return normalized_recommendations[:k]
```

### 算法复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 | 优化策略 |
|------|------------|------------|----------|
| 用户相似度计算 | O(m²n) | O(n²) | 稀疏矩阵、近似计算 |
| 推荐生成 | O(mn) | O(k) | 预计算、缓存 |
| 相似用户查找 | O(m log m) | O(m) | KD树、LSH |

### 算法优化

#### 稀疏矩阵优化
```python
class SparseUserItemMatrix:
    """稀疏用户-物品矩阵"""

    def __init__(self):
        self.user_to_items = defaultdict(dict)  # 用户到物品的映射
        self.item_to_users = defaultdict(dict)  # 物品到用户的映射
        self.user_averages = {}                 # 用户平均评分

    def get_user_vector(self, user_id: str) -> dict:
        """获取用户评分向量（稀疏表示）"""
        return self.user_to_items.get(user_id, {})

    def get_common_items(self, user_a: str, user_b: str) -> Set[str]:
        """获取两个用户的共同物品"""
        items_a = set(self.user_to_items[user_a].keys())
        items_b = set(self.user_to_items[user_b].keys())
        return items_a & items_b
```

#### 近似最近邻优化
```python
class ApproximateNearestNeighbors:
    """近似最近邻算法"""

    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []

    def build_index(self, users: List[str], embeddings: np.ndarray):
        """构建随机森林索引"""
        for _ in range(self.n_trees):
            tree = self._build_random_tree(users, embeddings)
            self.trees.append(tree)

    def find_similar_users(self, query_user: str, k: int = 10) -> List[Tuple[str, float]]:
        """查找相似用户"""
        candidates = set()
        for tree in self.trees:
            candidates.update(tree.search(query_user, k * 2))

        # 精确计算候选用户的相似度
        similarities = []
        for candidate in candidates:
            sim = self._exact_similarity(query_user, candidate)
            similarities.append((candidate, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
```

## 📚 内容推荐算法 (Content-Based Recommendation)

### 算法原理

内容推荐基于物品的内容特征和用户的历史偏好，通过特征匹配进行推荐。核心思想是推荐与用户历史喜好的物品相似的内容。

### 核心实现

#### 1. 特征提取器

```python
class FeatureExtractor:
    """特征提取器"""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.feature_cache = {}

    def extract_text_features(self, text: str) -> np.ndarray:
        """提取文本特征"""
        # 清理文本
        cleaned_text = self._clean_text(text)

        # TF-IDF特征
        tfidf_features = self.tfidf_vectorizer.fit_transform([cleaned_text])

        return tfidf_features.toarray()[0]

    def extract_structured_features(self, item_metadata: dict) -> np.ndarray:
        """提取结构化特征"""
        features = []

        # 类别特征（One-Hot编码）
        categories = item_metadata.get('categories', [])
        category_features = self._encode_categories(categories)
        features.extend(category_features)

        # 数值特征
        numeric_features = [
            item_metadata.get('price', 0),
            item_metadata.get('popularity', 0),
            item_metadata.get('rating', 0)
        ]
        features.extend(numeric_features)

        # 时间特征
        timestamp = item_metadata.get('timestamp')
        if timestamp:
            time_features = self._extract_time_features(timestamp)
            features.extend(time_features)

        return np.array(features)
```

#### 2. 用户画像构建

```python
class UserProfileBuilder:
    """用户画像构建器"""

    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.user_profiles = {}

    def build_profile(self, user_id: str,
                     interaction_history: List[Interaction]) -> UserProfile:
        """构建用户画像"""

        # 收集用户交互过的物品特征
        positive_features = []
        negative_features = []

        for interaction in interaction_history:
            item_features = self.feature_extractor.extract_features(
                interaction.item_id
            )

            weight = self._calculate_interaction_weight(interaction)
            weighted_features = item_features * weight

            if interaction.rating >= 4.0:  # 正面评价
                positive_features.append(weighted_features)
            else:  # 负面评价
                negative_features.append(weighted_features)

        # 计算偏好向量
        preference_vector = self._calculate_preference_vector(
            positive_features, negative_features
        )

        # 构建用户画像
        profile = UserProfile(
            user_id=user_id,
            preference_vector=preference_vector,
            favorite_categories=self._extract_favorite_categories(interaction_history),
            interaction_patterns=self._analyze_interaction_patterns(interaction_history)
        )

        self.user_profiles[user_id] = profile
        return profile
```

#### 3. 相似度计算

```python
def calculate_content_similarity(self, user_profile: UserProfile,
                               item_features: np.ndarray) -> float:
    """
    计算用户画像与物品的内容相似度

    支持多种相似度计算方法：
    - 余弦相似度
    - 欧氏距离
    - 曼哈顿距离
    """

    user_vector = user_profile.preference_vector

    # 余弦相似度
    cosine_sim = np.dot(user_vector, item_features) / (
        np.linalg.norm(user_vector) * np.linalg.norm(item_features)
    )

    # 加权相似度（考虑类别偏好）
    category_weight = self._calculate_category_weight(
        user_profile.favorite_categories, item_features
    )

    # 综合相似度
    final_similarity = 0.7 * cosine_sim + 0.3 * category_weight

    return max(0, min(1, final_similarity))
```

#### 4. 推荐生成

```python
def generate_content_recommendations(self, user_id: str,
                                   k: int = 10) -> RecommendationResult:
    """基于内容的推荐生成"""

    # 1. 获取用户画像
    user_profile = self.get_user_profile(user_id)
    if not user_profile:
        return RecommendationResult(user_id, "content_based", [])

    # 2. 获取候选物品
    candidate_items = self.get_candidate_items(user_id)

    # 3. 计算相似度分数
    recommendations = []
    user_seen_items = set(self.get_user_seen_items(user_id))

    for item_id in candidate_items:
        if item_id not in user_seen_items:
            # 提取物品特征
            item_features = self.feature_extractor.extract_features(item_id)

            # 计算相似度
            similarity = self.calculate_content_similarity(
                user_profile, item_features
            )

            if similarity > 0.1:  # 相似度阈值
                recommendations.append(
                    RecommendationItem(item_id, similarity)
                )

    # 4. 多样性优化
    diversified_recommendations = self._diversify_recommendations(
        recommendations, k
    )

    # 5. 排序并返回
    diversified_recommendations.sort(key=lambda x: x.score, reverse=True)

    return RecommendationResult(
        user_id,
        "content_based",
        diversified_recommendations[:k]
    )
```

### 特征工程技巧

#### 1. TF-IDF优化
```python
class OptimizedTFIDF:
    """优化的TF-IDF实现"""

    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.document_frequency = defaultdict(int)
        self.total_documents = 0

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """训练并转换文档"""
        self._build_vocabulary(documents)
        return self._transform(documents)

    def _build_vocabulary(self, documents: List[str]):
        """构建词汇表"""
        word_freq = defaultdict(int)

        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_freq[word] += 1
                self.document_frequency[word] += 1

        # 选择高频词
        sorted_words = sorted(word_freq.items(),
                            key=lambda x: x[1], reverse=True)
        self.word_to_idx = {word: idx for idx, (word, _)
                           in enumerate(sorted_words[:self.max_features])}
        self.idx_to_word = {idx: word for word, idx
                           in self.word_to_idx.items()}

        self.total_documents = len(documents)
```

#### 2. 特征降维
```python
class FeatureDimensionalityReduction:
    """特征降维处理"""

    def __init__(self, method: str = "pca", n_components: int = 100):
        self.method = method
        self.n_components = n_components
        self.reducer = None

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """训练并转换特征"""
        if self.method == "pca":
            self.reducer = PCA(n_components=self.n_components)
        elif self.method == "svd":
            self.reducer = TruncatedSVD(n_components=self.n_components)
        elif self.method == "nmf":
            self.reducer = NMF(n_components=self.n_components)

        return self.reducer.fit_transform(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """转换新特征"""
        if self.reducer is None:
            raise ValueError("模型未训练，请先调用fit_transform")
        return self.reducer.transform(features)
```

## 🌐 社交推荐算法 (Social Recommendation)

### 算法原理

社交推荐基于用户之间的社交关系，利用社交网络中的信任传播和影响力扩散进行推荐。核心假设是朋友或信任的用户会喜欢相似的物品。

### 核心实现

#### 1. 社交网络建模

```python
class SocialNetworkModel:
    """社交网络模型"""

    def __init__(self):
        self.social_graph = defaultdict(dict)  # 社交图
        self.user_influence = {}               # 用户影响力
        self.trust_scores = {}                 # 信任分数

    def add_social_connection(self, user_a: str, user_b: str,
                            strength: float):
        """添加社交连接"""
        # 验证连接强度
        if not 0 <= strength <= 1:
            raise ValueError("连接强度必须在0-1之间")

        # 建立双向连接
        self.social_graph[user_a][user_b] = strength
        self.social_graph[user_b][user_a] = strength

        # 清空相关缓存
        self._clear_trust_cache(user_a, user_b)
```

#### 2. 影响力计算

```python
def calculate_social_influence(self, source_user: str,
                             target_user: str, max_depth: int = 3) -> float:
    """
    计算社交影响力

    使用BFS算法在社交网络中传播影响力
    """
    if source_user == target_user:
        return 1.0

    if source_user not in self.social_graph:
        return 0.0

    # BFS搜索社交路径
    visited = set()
    queue = deque([(source_user, 1.0, 0)])  # (用户, 影响力, 深度)

    while queue:
        current_user, current_influence, depth = queue.popleft()

        if current_user in visited or depth >= max_depth:
            continue

        visited.add(current_user)

        if current_user == target_user:
            return current_influence

        # 传播影响力到朋友
        if current_user in self.social_graph:
            source_influence = self.user_influence.get(current_user, 0.5)

            for friend, connection_strength in self.social_graph[current_user].items():
                if friend not in visited:
                    # 影响力衰减计算
                    decay_factor = 0.8 ** depth
                    new_influence = (current_influence *
                                  connection_strength *
                                  source_influence *
                                  decay_factor)
                    queue.append((friend, new_influence, depth + 1))

    return 0.0
```

#### 3. 信任度计算

```python
def calculate_trust_score(self, user_a: str, user_b: str,
                         max_depth: int = 4) -> float:
    """
    计算信任分数

    信任度基于：
    1. 直接连接强度
    2. 间接路径衰减
    3. 共同朋友数量
    """
    if user_a == user_b:
        return 1.0

    # 检查缓存
    cache_key = tuple(sorted([user_a, user_b]))
    if cache_key in self.trust_scores:
        return self.trust_scores[cache_key]

    # 直接连接
    if user_b in self.social_graph.get(user_a, {}):
        direct_trust = self.social_graph[user_a][user_b]
        self.trust_scores[cache_key] = direct_trust
        return direct_trust

    # 间接连接信任度计算
    trust_score = 0.0
    visited = set()
    queue = deque([(user_a, 1.0, 0)])  # (用户, 信任度, 深度)

    while queue:
        current_user, current_trust, depth = queue.popleft()

        if current_user in visited or depth >= max_depth:
            continue

        visited.add(current_user)

        if current_user == user_b:
            trust_score = max(trust_score, current_trust)
            break

        if current_user in self.social_graph:
            for friend, connection_strength in self.social_graph[current_user].items():
                if friend not in visited:
                    # 信任度衰减
                    trust_decay = 0.7 ** depth
                    new_trust = current_trust * connection_strength * trust_decay
                    queue.append((friend, new_trust, depth + 1))

    self.trust_scores[cache_key] = trust_score
    return trust_score
```

#### 4. 社交推荐生成

```python
def generate_social_recommendations(self, user_id: str,
                                 user_activities: Dict[str, Dict[str, float]],
                                 k: int = 10) -> RecommendationResult:
    """生成社交推荐"""

    # 1. 获取朋友推荐
    friends_recommendations = self._get_friends_recommendations(
        user_id, user_activities, k * 2
    )

    # 2. 获取影响力推荐
    influence_recommendations = self._get_influence_based_recommendations(
        user_id, user_activities, k * 2
    )

    # 3. 合并推荐结果
    combined_scores = defaultdict(float)
    user_items = set(user_activities.get(user_id, {}).keys())

    # 朋友推荐权重：0.6
    for item_id, score in friends_recommendations:
        if item_id not in user_items:
            combined_scores[item_id] += score * 0.6

    # 影响力推荐权重：0.4
    for item_id, score in influence_recommendations:
        if item_id not in user_items:
            combined_scores[item_id] += score * 0.4

    # 4. 生成最终推荐
    recommendations = []
    for item_id, score in combined_scores.items():
        if score > 0:
            normalized_score = min(score / 5.0, 1.0)
            recommendations.append(
                RecommendationItem(item_id, normalized_score)
            )

    recommendations.sort(key=lambda x: x.score, reverse=True)

    return RecommendationResult(user_id, "social_based", recommendations[:k])
```

### 社交路径分析

```python
def find_social_paths(self, source_user: str, target_user: str,
                     max_depth: int = 4) -> List[List[str]]:
    """
    查找两个用户之间的社交路径

    用于：
    1. 信任传播路径分析
    2. 影响力扩散路径
    3. 社交网络可视化
    """
    if source_user == target_user:
        return [[]]

    if source_user not in self.social_graph:
        return []

    paths = []
    visited_global = set()
    queue = deque([(source_user, [source_user])])  # (当前用户, 路径)

    while queue:
        current_user, path = queue.popleft()

        if current_user in visited_global or len(path) > max_depth + 1:
            continue

        # 为每个路径维护独立的visited集合
        visited_path = set(path)

        if current_user == target_user:
            paths.append(path[1:])  # 排除源用户
            continue

        visited_global.add(current_user)

        if current_user in self.social_graph:
            for friend in self.social_graph[current_user]:
                if friend not in visited_path:  # 避免循环
                    queue.append((friend, path + [friend]))

    return paths
```

## 🎯 混合推荐算法 (Hybrid Recommendation)

### 算法原理

混合推荐结合多种推荐算法的优势，通过加权融合、级联混合或切换策略来提高推荐质量。我们的系统采用加权融合策略，动态调整各算法的权重。

### 核心实现

#### 1. 加权融合策略

```python
class WeightedHybridStrategy:
    """加权融合策略"""

    def __init__(self):
        self.weights = {
            "collaborative": 0.5,
            "content": 0.3,
            "social": 0.2
        }
        self.personalized_weights = {}

    def calculate_hybrid_score(self, cf_score: Optional[float],
                             content_score: Optional[float],
                             social_score: Optional[float]) -> float:
        """
        计算混合推荐分数

        处理策略：
        1. 有分数的引擎参与计算
        2. 无分数的引擎权重被重新分配
        3. 确保分数归一化到[0,1]
        """
        scores = []
        weights = []

        # 收集有效分数和对应权重
        if cf_score is not None:
            scores.append(cf_score)
            weights.append(self.weights["collaborative"])

        if content_score is not None:
            scores.append(content_score)
            weights.append(self.weights["content"])

        if social_score is not None:
            scores.append(social_score)
            weights.append(self.weights["social"])

        if not scores:
            return 0.0

        # 重新分配权重
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            hybrid_score = sum(score * weight
                             for score, weight in zip(scores, normalized_weights))
            return hybrid_score

        return 0.0
```

#### 2. 自适应权重调整

```python
class AdaptiveWeightAdjustment:
    """自适应权重调整"""

    def __init__(self):
        self.performance_history = defaultdict(list)
        self.adjustment_rate = 0.1

    def update_weights(self, user_id: str,
                      engine_performance: Dict[str, float]):
        """
        基于性能指标调整权重

        性能指标包括：
        1. 点击率 (CTR)
        2. 转化率
        3. 用户满意度
        4. 推荐多样性
        """

        # 记录性能历史
        for engine, performance in engine_performance.items():
            self.performance_history[engine].append(performance)

        # 计算平均性能
        avg_performance = {}
        for engine in self.weights.keys():
            history = self.performance_history[engine]
            if history:
                avg_performance[engine] = np.mean(history[-10:])  # 最近10次
            else:
                avg_performance[engine] = 0.5  # 默认值

        # 调整权重
        self._adjust_weight_based_on_performance(avg_performance)

    def _adjust_weight_based_on_performance(self, performance: Dict[str, float]):
        """基于性能调整权重"""

        # 计算性能分数总和
        total_performance = sum(performance.values())

        if total_performance > 0:
            # 计算新权重
            new_weights = {}
            for engine, base_weight in self.weights.items():
                perf_score = performance.get(engine, 0.5)
                # 结合基础权重和性能分数
                new_weight = (base_weight * 0.7 + perf_score * 0.3)
                new_weights[engine] = new_weight

            # 归一化权重
            total_new_weight = sum(new_weights.values())
            if total_new_weight > 0:
                for engine in new_weights:
                    new_weights[engine] /= total_new_weight

                # 平滑调整（避免突变）
                for engine in self.weights:
                    self.weights[engine] = (
                        self.weights[engine] * (1 - self.adjustment_rate) +
                        new_weights[engine] * self.adjustment_rate
                    )
```

#### 3. 多样性增强

```python
class DiversityEnhancer:
    """推荐多样性增强"""

    def __init__(self):
        self.category_weights = defaultdict(float)
        self.recommendation_history = defaultdict(list)

    def enhance_diversity(self, recommendations: List[RecommendationItem],
                         k: int) -> List[RecommendationItem]:
        """
        增强推荐多样性

        策略：
        1. 类别平衡
        2. 时间分散
        3. 兴趣探索
        """
        if len(recommendations) <= k:
            return recommendations

        diverse_recommendations = []
        used_categories = set()
        category_count = defaultdict(int)
        max_per_category = max(1, k // 3)  # 每个类别最多占1/3

        # 按分数排序
        sorted_recs = sorted(recommendations, key=lambda x: x.score, reverse=True)

        # 第一轮：类别平衡选择
        for rec in sorted_recs:
            if len(diverse_recommendations) >= k:
                break

            category = self._extract_category(rec.item_id)

            # 类别多样性控制
            if (category not in used_categories or
                category_count[category] < max_per_category):
                diverse_recommendations.append(rec)
                used_categories.add(category)
                category_count[category] += 1

        # 第二轮：填充剩余位置
        remaining_items = [rec for rec in sorted_recs
                          if rec not in diverse_recommendations]
        diverse_recommendations.extend(
            remaining_items[:k - len(diverse_recommendations)]
        )

        return diverse_recommendations
```

#### 4. 推荐解释生成

```python
def generate_recommendation_explanation(self, user_id: str,
                                      item_id: str) -> Dict[str, Any]:
    """
    生成推荐解释

    解释包括：
    1. 各引擎贡献分数
    2. 推荐原因分析
    3. 相似用户/物品信息
    """

    explanation = {
        "item_id": item_id,
        "explanations": [],
        "confidence_scores": {},
        "similar_users": [],
        "similar_items": []
    }

    # 协同过滤解释
    cf_score = self._get_cf_score(user_id, item_id)
    if cf_score > 0:
        similar_users = self._get_similar_users_for_item(user_id, item_id)
        explanation["explanations"].append({
            "engine": "collaborative",
            "reason": f"与您相似的用户也喜欢{item_id}",
            "score": cf_score,
            "support": len(similar_users)
        })
        explanation["similar_users"] = similar_users[:3]

    # 内容推荐解释
    content_score = self._get_content_score(user_id, item_id)
    if content_score > 0:
        similar_items = self._get_similar_items(user_id, item_id)
        explanation["explanations"].append({
            "engine": "content",
            "reason": f"基于您的历史偏好推荐{item_id}",
            "score": content_score,
            "support": len(similar_items)
        })
        explanation["similar_items"] = similar_items[:3]

    # 社交推荐解释
    social_score = self._get_social_score(user_id, item_id)
    if social_score > 0:
        friends_liked = self._get_friends_who_liked(user_id, item_id)
        explanation["explanations"].append({
            "engine": "social",
            "reason": f"您的朋友喜欢{item_id}",
            "score": social_score,
            "support": len(friends_liked)
        })

    # 计算整体置信度
    explanation["overall_confidence"] = self._calculate_overall_confidence(
        explanation["explanations"]
    )

    return explanation
```

## 📊 算法评估指标

### 准确性指标

```python
class RecommendationMetrics:
    """推荐算法评估指标"""

    @staticmethod
    def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """Precision@K"""
        if k == 0:
            return 0.0

        recommended_k = recommended[:k]
        relevant_set = set(relevant)

        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        """Recall@K"""
        if not relevant:
            return 0.0

        recommended_k = recommended[:k]
        relevant_set = set(relevant)

        hits = sum(1 for item in recommended_k if item in relevant_set)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: List[str],
                  relevance_scores: List[float], k: int) -> float:
        """NDCG@K"""
        def dcg_at_k(relevances: List[float], k: int) -> float:
            relevances_k = relevances[:k]
            return sum(rel / np.log2(i + 2)
                      for i, rel in enumerate(relevances_k))

        # 实际DCG
        actual_relevances = []
        for item in recommended[:k]:
            if item in relevant:
                idx = relevant.index(item)
                actual_relevances.append(relevance_scores[idx])
            else:
                actual_relevances.append(0.0)

        actual_dcg = dcg_at_k(actual_relevances, k)

        # 理想DCG
        ideal_relevances = sorted(relevance_scores, reverse=True)
        ideal_dcg = dcg_at_k(ideal_relevances, k)

        if ideal_dcg == 0:
            return 0.0

        return actual_dcg / ideal_dcg
```

### 多样性指标

```python
@staticmethod
def intra_list_diversity(recommendations: List[str],
                        item_features: Dict[str, np.ndarray]) -> float:
    """列表内多样性"""
    if len(recommendations) < 2:
        return 0.0

    total_similarity = 0.0
    pair_count = 0

    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            item_i = recommendations[i]
            item_j = recommendations[j]

            if item_i in item_features and item_j in item_features:
                similarity = np.dot(item_features[item_i], item_features[item_j])
                total_similarity += similarity
                pair_count += 1

    if pair_count == 0:
        return 0.0

    avg_similarity = total_similarity / pair_count
    return 1.0 - avg_similarity  # 转换为多样性分数
```

### 新颖性指标

```python
@staticmethod
def novelty(recommendations: List[str],
           item_popularity: Dict[str, float]) -> float:
    """推荐新颖性"""
    if not recommendations:
        return 0.0

    # 计算每个物品的负对数流行度
    novelty_scores = []
    for item in recommendations:
        popularity = item_popularity.get(item, 0.001)  # 避免除零
        novelty_score = -np.log2(popularity)
        novelty_scores.append(novelty_score)

    return np.mean(novelty_scores)
```

## 🎯 算法优化策略

### 1. 冷启动问题解决

```python
class ColdStartSolver:
    """冷启动问题解决"""

    def handle_new_user(self, user_id: str,
                       minimal_info: Dict[str, Any]) -> List[str]:
        """处理新用户冷启动"""

        # 策略1：基于人口统计学的推荐
        if "demographics" in minimal_info:
            demo_recommendations = self._demographic_based_recommendation(
                minimal_info["demographics"]
            )

        # 策略2：基于注册时选择的兴趣
        if "interests" in minimal_info:
            interest_recommendations = self._interest_based_recommendation(
                minimal_info["interests"]
            )

        # 策略3：热门物品推荐
        popular_recommendations = self._get_popular_items()

        # 混合策略
        final_recommendations = self._combine_cold_start_strategies([
            demo_recommendations,
            interest_recommendations,
            popular_recommendations
        ])

        return final_recommendations[:10]
```

### 2. 实时推荐优化

```python
class RealTimeRecommendation:
    """实时推荐优化"""

    def __init__(self):
        self.user_state_cache = {}
        self.item_candidate_pool = {}
        self.precomputed_similarities = {}

    async def realtime_recommend(self, user_id: str,
                               context: Dict[str, Any]) -> List[str]:
        """实时推荐生成"""

        # 1. 获取用户当前状态
        user_state = await self._get_user_state(user_id)

        # 2. 基于上下文过滤候选池
        filtered_candidates = self._filter_by_context(
            self.item_candidate_pool[user_id], context
        )

        # 3. 快速相似度计算
        scores = []
        for item in filtered_candidates:
            score = self._fast_similarity_calculation(user_state, item)
            scores.append((item, score))

        # 4. 排序并返回
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scores[:10]]
```

这套推荐算法体系通过TDD方法论确保了代码质量和算法准确性，为百万级智能体平台提供了高质量的推荐服务。每个算法都经过充分测试，具备良好的扩展性和性能表现。