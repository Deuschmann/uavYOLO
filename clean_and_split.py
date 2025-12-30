import os
import shutil
import random
from glob import glob

# ================= 配置 =================
# 你的源文件夹名字
source_root = 'data_light' 
# 输出的完美文件夹名字
output_root = 'clean_baseline_dataset'
# =======================================

def main():
    if not os.path.exists(source_root):
        print(f"❌ 找不到文件夹 {source_root}，请检查路径！")
        return

    print("🔍 正在扫描所有干净图片 (自动忽略 hazy_images)...")

    # 1. 搜集所有数据
    # 我们只找 data_light/*/images/*.jpg (或者png)
    # 这样就物理隔绝了 hazy_images 文件夹
    all_image_paths = []
    
    # 遍历 train, val, test 子文件夹
    sub_dirs = ['train', 'val', 'test']
    for sub in sub_dirs:
        img_dir = os.path.join(source_root, sub, 'images')
        lbl_dir = os.path.join(source_root, sub, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        # 获取该目录下所有图片
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in exts:
            # 找到所有图片路径
            found_imgs = glob(os.path.join(img_dir, ext))
            
            for img_path in found_imgs:
                # 构造对应的 label 路径
                file_name = os.path.basename(img_path)
                base_name = os.path.splitext(file_name)[0]
                lbl_path = os.path.join(lbl_dir, base_name + '.txt')
                
                # 只有当图片和标签都存在时，才算有效数据
                if os.path.exists(lbl_path):
                    all_image_paths.append((img_path, lbl_path))

    total = len(all_image_paths)
    print(f"📦 共收集到 {total} 组有效干净数据 (Images + Labels)")
    
    if total == 0:
        print("❌ 没找到数据，请检查 data_light 里的结构是不是 train/images 这种格式")
        return

    # 2. 打乱数据
    random.shuffle(all_image_paths)

    # 3. 按 7:1:1 划分
    train_end = int(total * 0.7)
    val_end = int(total * 0.8) # 0.7 + 0.1

    splits = {
        'train': all_image_paths[:train_end],
        'val': all_image_paths[train_end:val_end],
        'test': all_image_paths[val_end:]
    }

    print(f"📊 划分结果 -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # 4. 复制文件
    for split_name, files in splits.items():
        # 创建标准的 YOLO 目录结构: output/images/train, output/labels/train
        save_img_dir = os.path.join(output_root, 'images', split_name)
        save_lbl_dir = os.path.join(output_root, 'labels', split_name)
        
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_lbl_dir, exist_ok=True)
        
        print(f"🚀 正在生成 {split_name} 集...")
        for src_img, src_lbl in files:
            shutil.copy(src_img, save_img_dir)
            shutil.copy(src_lbl, save_lbl_dir)

    print(f"\n✅ 清洗完成！")
    print(f"📂 新的数据集在: {os.path.abspath(output_root)}")
    print(f"🚫 所有的 hazy_images 都已被剔除。")
    print("👉 请去 yaml 文件里把 data 路径改成这个新文件夹！")

if __name__ == "__main__":
    main()