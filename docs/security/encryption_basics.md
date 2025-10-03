# 数据加密基础知识

## 加密概念

加密是将明文数据转换为密文的过程，只有拥有正确密钥的人才能解密还原数据。加密是保护敏感信息安全传输和存储的核心技术。

## 加密算法分类

### 1. 对称加密（Symmetric Encryption）
使用相同的密钥进行加密和解密。

#### AES (Advanced Encryption Standard)
- **密钥长度**: 128, 192, 256位
- **块大小**: 128位
- **工作模式**: ECB, CBC, CFB, OFB, GCM, CTR
- **应用场景**: 大量数据加密，数据库字段加密

```python
from cryptography.fernet import Fernet

# 生成密钥
key = Fernet.generate_key()
cipher = Fernet(key)

# 加密
plaintext = b"sensitive data"
ciphertext = cipher.encrypt(plaintext)

# 解密
decrypted = cipher.decrypt(ciphertext)
```

### 2. 非对称加密（Asymmetric Encryption）
使用公钥加密，私钥解密。

#### RSA (Rivest-Shamir-Adleman)
- **密钥长度**: 1024, 2048, 4096位
- **安全性**: 基于大数分解难题
- **应用场景**: 密钥交换，数字签名，小数据加密

```python
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# 生成密钥对
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)
public_key = private_key.public_key()

# 加密
message = b"secret message"
ciphertext = public_key.encrypt(
    message,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)

# 解密
plaintext = private_key.decrypt(
    ciphertext,
    padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
```

### 3. 哈希函数（Hash Functions）
单向函数，将任意长度数据映射为固定长度哈希值。

#### 常见哈希算法
- **SHA-256**: 256位哈希值
- **SHA-512**: 512位哈希值
- **PBKDF2**: 密码哈希函数
- **bcrypt**: 专为密码设计的哈希函数

```python
import hashlib
import bcrypt

# SHA-256哈希
password = "my_password".encode()
hash_object = hashlib.sha256(password)
hex_dig = hash_object.hexdigest()

# bcrypt密码哈希
salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(password, salt)

# 验证密码
if bcrypt.checkpw(password, hashed):
    print("密码正确")
```

## 加密模式

### 1. ECB (Electronic Codebook)
- **特点**: 每个块独立加密
- **缺点**: 相同明文产生相同密文
- **安全性**: 不推荐使用

### 2. CBC (Cipher Block Chaining)
- **特点**: 前一个密文块影响后续加密
- **要求**: 初始化向量(IV)
- **安全性**: 相对安全

```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os

# AES-CBC加密
key = os.urandom(32)  # 256位密钥
iv = os.urandom(16)   # 128位IV
cipher = AES.new(key, AES.MODE_CBC, iv)

plaintext = b"this is a secret message"
padded_text = pad(plaintext, AES.block_size)
ciphertext = cipher.encrypt(padded_text)

# 解密
cipher = AES.new(key, AES.MODE_CBC, iv)
decrypted_padded = cipher.decrypt(ciphertext)
decrypted = unpad(decrypted_padded, AES.block_size)
```

### 3. GCM (Galois/Counter Mode)
- **特点**: 提供认证加密
- **优势**: 同时保证机密性和完整性
- **应用**: 现代应用推荐使用

## 密钥管理

### 1. 密钥生成
```python
import secrets
from cryptography.hazmat.primitives.asymmetric import rsa

# 生成随机密钥
symmetric_key = secrets.token_bytes(32)  # 256位

# 生成RSA密钥对
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)
```

### 2. 密钥存储
```python
import keyring
from cryptography.fernet import Fernet

# 使用系统密钥环存储
keyring.set_password("my_app", "encryption_key", key)

# 从密钥环获取
stored_key = keyring.get_password("my_app", "encryption_key")
```

### 3. 密钥轮换
```python
class KeyManager:
    def __init__(self):
        self.current_key_id = 0
        self.keys = {}

    def rotate_key(self):
        """轮换加密密钥"""
        new_key_id = self.current_key_id + 1
        new_key = Fernet.generate_key()
        self.keys[new_key_id] = new_key
        self.current_key_id = new_key_id

    def get_current_key(self):
        """获取当前密钥"""
        return self.keys[self.current_key_id]
```

## 加密应用场景

### 1. 数据库字段加密
```python
class User:
    def __init__(self, email, phone):
        self.email = self.encrypt_field(email)
        self.phone = self.encrypt_field(phone)

    def encrypt_field(self, data):
        cipher = Fernet(get_encryption_key())
        return cipher.encrypt(data.encode()).decode()

    def decrypt_field(self, encrypted_data):
        cipher = Fernet(get_encryption_key())
        return cipher.decrypt(encrypted_data.encode()).decode()
```

### 2. 文件加密
```python
def encrypt_file(input_file, output_file, key):
    cipher = Fernet(key)

    with open(input_file, 'rb') as f:
        file_data = f.read()

    encrypted_data = cipher.encrypt(file_data)

    with open(output_file, 'wb') as f:
        f.write(encrypted_data)

def decrypt_file(input_file, output_file, key):
    cipher = Fernet(key)

    with open(input_file, 'rb') as f:
        encrypted_data = f.read()

    decrypted_data = cipher.decrypt(encrypted_data)

    with open(output_file, 'wb') as f:
        f.write(decrypted_data)
```

### 3. API通信加密
```python
import requests
from cryptography.fernet import Fernet

class SecureAPIClient:
    def __init__(self, base_url, encryption_key):
        self.base_url = base_url
        self.cipher = Fernet(encryption_key)

    def send_secure_request(self, endpoint, data):
        # 加密请求数据
        json_data = json.dumps(data).encode()
        encrypted_data = self.cipher.encrypt(json_data)

        # 发送加密请求
        response = requests.post(
            f"{self.base_url}{endpoint}",
            data=encrypted_data,
            headers={'Content-Type': 'application/octet-stream'}
        )

        # 解密响应
        encrypted_response = response.content
        decrypted_response = self.cipher.decrypt(encrypted_response)
        return json.loads(decrypted_response.decode())
```

## 安全最佳实践

### 1. 密钥安全
- ✅ 使用强随机数生成器
- ✅ 定期轮换密钥
- ✅ 安全存储密钥
- ❌ 硬编码密钥在代码中
- ❌ 使用简单密码作为密钥

### 2. 算法选择
- ✅ 使用AES-256进行对称加密
- ✅ 使用RSA-2048或更长密钥
- ✅ 使用GCM模式提供认证加密
- ❌ 使用DES等过时算法
- ❌ 使用ECB模式

### 3. 实现安全
- ✅ 使用标准加密库
- ✅ 验证输入数据
- ✅ 处理加密异常
- ❌ 自行实现加密算法
- ❌ 忽略错误处理

## 性能考虑

### 加密性能对比
| 算法 | 加密速度 | 解密速度 | 安全性 |
|------|----------|----------|--------|
| AES-128 | 快 | 快 | 高 |
| AES-256 | 中等 | 中等 | 很高 |
| RSA-2048 | 慢 | 快 | 高 |
| ChaCha20 | 快 | 快 | 高 |

### 优化建议
1. **批量加密**: 对大量数据进行批量处理
2. **密钥缓存**: 缓存常用密钥避免重复生成
3. **异步加密**: 对大文件使用流式加密
4. **硬件加速**: 利用CPU的AES指令集

## 常见攻击及防护

### 1. 中间人攻击
**防护**:
- 使用证书验证
- 实施完整性检查
- 使用签名验证

### 2. 密钥泄露
**防护**:
- 安全存储密钥
- 定期轮换密钥
- 使用硬件安全模块(HSM)

### 3. 时序攻击
**防护**:
- 使用常量时间比较
- 避免基于响应时间的错误信息

## 法律合规

### 数据保护法规
- **GDPR**: 欧盟数据保护条例
- **CCPA**: 加州消费者隐私法案
- **网络安全法**: 中国网络安全法

### 加密标准
- **FIPS 140-2**: 美国联邦信息处理标准
- **Common Criteria**: 国际通用安全标准
- **国密标准**: 中国商用密码标准

## 推荐工具和库

### Python
- **cryptography**: 功能最全面的加密库
- **PyCryptodome**: 易用的加密库
- **passlib**: 密码哈希库

### 其他语言
- **OpenSSL**: C/C++加密库
- **Bouncy Castle**: Java加密库
- **Crypto++**: C++加密库

## 参考资源

- [NIST密码学指南](https://csrc.nist.gov/projects/cryptographic-guidelines)
- [OWASP加密指南](https://owasp.org/www-project-cryptography/)
- [cryptography.io文档](https://cryptography.io/)