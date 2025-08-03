# SEM Synthetic Image Generator

一个用于生成合成SEM（扫描电子显微镜）图像的完整工具包，支持双材质纹理合成和智能mask生成。

## 🎯 功能特性

- **交互式标注工具**: 图形化界面用于选择和保存感兴趣的纹理patch
- **多种纹理合成方法**: 
  - Neural Texture Synthesis (基于VGG特征的神经纹理合成)
  - Image Quilting (经典的patch拼接算法)
  - Mixed Method (混合方法，结合两者优势)
- **多样化mask生成**: 支持7种不同类型的材质分布mask
- **完整的生成流水线**: 从patch标注到最终合成图像的一站式解决方案
- **SEM特效后处理**: 添加噪声、亮度变化等SEM图像特有效果

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 或者在conda环境中安装
conda activate langchain  # 使用你现有的环境
pip install -r requirements.txt
```

### 2. 快速演示

如果你想快速看到效果，可以运行演示模式：

```bash
python main.py --demo
```

这会自动从你的SEM样本图像中提取patches并生成几张示例图像。

### 3. 完整工作流程

#### 步骤1: 标注感兴趣的纹理patches

```bash
python main.py --annotate
```

这会启动图形化标注工具：
1. 点击 "Load Image" 加载你的SEM参考图像
2. 用鼠标拖拽选择感兴趣的纹理区域
3. 点击 "Save Patches" 保存所有选中的patches
4. 关闭窗口完成标注

#### 步骤2: 生成合成SEM图像

```bash
# 生成10张512x512的图像（默认参数）
python main.py --generate

# 自定义参数生成
python main.py --generate --count 50 --size 1024 --method mixed --device cuda
```

## 📁 项目结构

```
SEM_gen/
├── src/                          # 源代码模块
│   ├── patch_annotator.py        # 交互式patch标注工具
│   ├── texture_synthesizer.py    # Neural texture synthesis
│   ├── quilting.py               # Image quilting算法
│   ├── mask_generator.py         # Mask生成器
│   └── sem_generator.py          # 主生成器
├── main.py                       # 主入口脚本
├── requirements.txt              # 依赖包列表
├── README.md                     # 项目说明
└── 你的SEM样本图像文件
```

## 🔧 详细使用说明

### 命令行参数

#### 基本操作
- `--annotate`: 启动patch标注工具
- `--generate`: 生成合成图像
- `--demo`: 运行快速演示

#### 生成参数
- `--count N`: 生成图像数量（默认：10）
- `--size N`: 输出图像尺寸（默认：512）
- `--method {neural,quilting,mixed}`: 生成方法（默认：mixed）
- `--output-dir DIR`: 输出目录（默认：generated_sem）
- `--device {cpu,cuda}`: 计算设备（自动检测）
- `--seed N`: 随机种子（用于复现结果）
- `--mask-types TYPE [TYPE ...]`: Mask类型（默认：perlin voronoi cellular）
- `--no-effects`: 禁用SEM特效后处理

### 生成方法说明

1. **Neural Method** (`--method neural`)
   - 使用VGG19特征提取器
   - 基于Gram矩阵的纹理合成
   - 质量最高，但计算时间较长
   - 适合追求最佳视觉效果

2. **Quilting Method** (`--method quilting`)
   - 经典的Image Quilting算法
   - 直接拼接真实的patch
   - 速度快，保真度高
   - 适合快速原型和验证

3. **Mixed Method** (`--method mixed`) **[推荐]**
   - 结合neural synthesis和quilting
   - 平衡质量和速度
   - 综合效果最佳

### Mask类型说明

- `perlin`: Perlin噪声生成自然边界
- `voronoi`: Voronoi图生成多边形区域
- `cellular`: 细胞自动机生成有机形状
- `fractal`: 分形几何生成复杂边界
- `layered`: 分层结构模拟地质材料
- `blob`: 斑点状分布
- `network`: 网络/裂纹状分布

## 📊 输出文件结构

生成完成后，你会在输出目录中找到：

```
generated_sem/
├── material_1/                   # 材质1的生成过程
│   ├── texture_material_1.png    # 最终纹理
│   ├── neural_progress/          # 神经合成中间结果
│   └── quilting_progress/        # Quilting中间结果
├── material_2/                   # 材质2的生成过程
├── masks/                        # 生成的所有mask
│   ├── mask_000_perlin.png
│   ├── mask_001_voronoi.png
│   └── ...
└── dataset/                      # 最终合成数据集
    ├── sem_synthetic_000.png     # 合成SEM图像
    ├── mask_000.png              # 对应的材质分布mask
    ├── sem_synthetic_001.png
    ├── mask_001.png
    ├── ...
    └── generation_metadata.json  # 生成元数据
```

## 🎨 使用示例

### 示例1: 快速生成小批量图像

```bash
# 标注patches
python main.py --annotate

# 生成5张256x256的图像用于快速验证
python main.py --generate --count 5 --size 256 --method quilting
```

### 示例2: 高质量大批量生成

```bash
# 生成100张1024x1024高质量图像
python main.py --generate --count 100 --size 1024 --method mixed --device cuda
```

### 示例3: 特定风格的mask

```bash
# 只使用分形和细胞自动机mask
python main.py --generate --count 20 --mask-types fractal cellular
```

## 🔬 技术原理

### Neural Texture Synthesis
基于Gatys et al.的"Texture Synthesis Using Convolutional Neural Networks"，使用预训练的VGG19网络提取多层特征，通过Gram矩阵匹配实现纹理合成。

### Image Quilting
实现Efros & Freeman的"Image Quilting for Texture Synthesis and Transfer"算法，通过智能patch拼接和边界优化实现高质量纹理合成。

### Mask Generation
提供多种数学和程序化方法生成材质分布mask：
- 分形几何
- 噪声函数
- 细胞自动机
- Voronoi图
- 程序化形状生成

## 🛠️ 高级配置

### 自定义纹理合成参数

可以通过修改源代码中的类初始化参数来调整：

```python
# 在 src/texture_synthesizer.py 中
synthesizer = TextureSynthesizer(device='cuda')
result = synthesizer.generate_from_patches(
    patch_dir,
    output_size=(512, 512),
    num_iterations=1000,  # 增加迭代次数提高质量
    save_progress=True
)
```

### 自定义Mask生成

```python
# 在 src/mask_generator.py 中
generator = MaskGenerator(seed=42)
mask = generator.generate_mask(
    shape=(512, 512),
    mask_type='perlin',
    threshold=0.4,  # 调整材质比例
    scale=80        # 调整纹理尺度
)
```

## 🐛 故障排除

### 常见问题

1. **ImportError: No module named 'torch'**
   ```bash
   pip install torch torchvision
   ```

2. **CUDA out of memory**
   ```bash
   # 使用CPU模式
   python main.py --generate --device cpu
   ```

3. **标注工具无法启动**
   - 确保安装了tkinter：`sudo apt-get install python3-tk` (Linux)
   - 确保有图形界面环境

4. **生成的图像质量不理想**
   - 增加patch数量和多样性
   - 尝试不同的生成方法
   - 调整neural synthesis迭代次数

## 📝 开发者说明

### 扩展新的纹理合成方法

在`src/`目录下创建新的合成器类，并在`SEMGenerator`中集成：

```python
class MyTextureMethod:
    def generate_from_patches(self, patch_dir, output_size, **kwargs):
        # 实现你的方法
        return generated_texture
```

### 添加新的Mask类型

在`MaskGenerator`类中添加新方法：

```python
def generate_my_mask(self, shape, **kwargs):
    # 实现新的mask生成算法
    return mask
```

## 📄 许可证

本项目采用MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系

如有问题或建议，请通过GitHub Issues联系。 