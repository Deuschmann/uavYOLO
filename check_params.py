# check_params.py

from src.models import build_model
from src.utils.config import load_config

# 1. 加载你的配置文件
# 确保这里的路径是正确的
config_path = 'configs/base.yaml' 
config = load_config(config_path)

# 2. 构建模型
# 使用和你训练时完全相同的参数
model = build_model(
    config=config,
    num_classes=config['classes']['num_classes'],
    img_size=config['data']['img_size']
)

# 3. 调用函数并打印信息
model_info = model.get_model_info()

# 打印详细信息
print("="*40)
print("Model Information")
print("="*40)
for key, value in model_info.items():
    if 'params' in key:
        print(f"{key:<20}: {value:,}") # 加上千位分隔符，更易读
    else:
        print(f"{key:<20}: {value}")

print("="*40)

# 帮你直接给出结论
total_params_M = model_info['total_params'] / 1_000_000
print(f"\nTotal parameters: {total_params_M:.2f} M")

if total_params_M <= 7:
    print("Suggested comparison model: YOLOv8n (3.2 M params)")
elif total_params_M <= 20:
    print("Suggested comparison model: YOLOv8s (11.2 M params)")
else:
    print("Suggested comparison model: YOLOv8m (25.9 M params)")
