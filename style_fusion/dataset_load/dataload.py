import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FusionDataset(Dataset):
    """
    加载包含content、style、stage1、stage2四个文件夹中图片的数据集类
    四个文件夹中文件名相同的图片视为一组数据
    """
    def __init__(self, root_dir, cs_transform=None, other_transform=None):
        """
        初始化数据集

        参数:
            root_dir (string): 包含5个子文件夹的根目录路径
            transform (callable, optional): 可选的变换操作，用于处理加载的图片
        """
        self.root_dir = root_dir
        self.cs_transform = cs_transform
        self.other_transform = other_transform

        # 定义四个子文件夹的名称
        self.folders = ['content', 'style', 'stage0', 'stage1', 'stage2']

        # 检查所有文件夹是否存在
        for folder in self.folders:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                raise ValueError(f"文件夹不存在: {folder_path}")

        # 获取所有共同的文件名（不包含扩展名）
        self.common_filenames = self._get_common_filenames()

        if len(self.common_filenames) == 0:
            raise ValueError("5个文件夹中没有找到共同的图片文件")

    def _get_common_filenames(self):
        """获取5个文件夹中共同的文件名（不含扩展名）"""
        # 存储每个文件夹中的文件名集合（不含扩展名）
        folder_files = {}

        for folder in self.folders:
            folder_path = os.path.join(self.root_dir, folder)
            files = set()

            # 遍历文件夹中的所有文件
            for filename in os.listdir(folder_path):
                # 检查是否为图片文件
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # 获取不带扩展名的文件名
                    name_without_ext = os.path.splitext(filename)[0]
                    files.add(name_without_ext)

            folder_files[folder] = files

        # 计算四个文件夹中共同的文件名
        common_files = folder_files[self.folders[0]]
        for folder in self.folders[1:]:
            common_files.intersection_update(folder_files[folder])

        return sorted(list(common_files))

    def _get_image_path(self, index, folder):
        """获取指定索引和文件夹中的图片路径"""
        base_name = self.common_filenames[index]
        folder_path = os.path.join(self.root_dir, folder)

        # 查找对应的图片文件（处理不同扩展名的情况）
        for filename in os.listdir(folder_path):
            if os.path.splitext(filename)[0] == base_name:
                return os.path.join(folder_path, filename)

        # 理论上不会走到这里，因为已经通过_common_filenames验证
        raise FileNotFoundError(f"在{folder}文件夹中找不到{base_name}对应的图片文件")

    def __len__(self):
        """返回数据集的大小"""
        return len(self.common_filenames)

    def __getitem__(self, index):
        """
        根据索引获取一组数据

        返回:
            字典，包含四个文件夹中的图片数据，键为文件夹名称
        """
        if torch.is_tensor(index):
            index = index.tolist()

        # 加载四个文件夹中的对应图片
        data = {}
        for folder in self.folders:
            img_path = self._get_image_path(index, folder)

            # 打开图片并转换为RGB格式
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"无法打开图片 {img_path}: {str(e)}")

            if folder in ['content', 'style']:
                # 对content和style图片应用cs_transform
                if self.cs_transform:
                    image = self.cs_transform(image)
            else:
                # 对stage0, stage1, stage2图片应用other_transform
                if self.other_transform:
                    image = self.other_transform(image)

            data[folder] = image

        return data

class FusionDataset_v2(Dataset):
    def __init__(self, root_dir, transform=None, folders=None):
        """
        Args:
            root_dir (str): 数据根目录，里面有 content, style, stage0, stage1, stage2 文件夹
            transform (callable, optional): 图像变换
            folders (list, optional): 需要加载的文件夹名，默认加载 ['content', 'style', 'stage0', 'stage1', 'stage2']
        """
        self.root_dir = root_dir
        self.transform = transform
        self.folders = folders if folders is not None else ['content', 'style', 'stage0', 'stage1', 'stage2']

        # 假设所有文件夹下文件名完全对应，以 content 文件夹作为基准
        self.file_names = sorted(os.listdir(os.path.join(root_dir, self.folders[0])))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        sample = {}
        fname = self.file_names[idx]

        for folder in self.folders:
            img_path = os.path.join(self.root_dir, folder, fname)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            sample[folder] = img

        return sample

# 使用示例
if __name__ == "__main__":
    # 示例：创建数据集并测试
    root_dir = '/data/yueyisui/datasets/DIOR/JPEGImages-1024-256_fusion'  # 替换为实际的根目录路径

    # 定义一些基本的图片变换
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        dataset = FusionDataset(root_dir, cs_transform=transform)
        print(f"数据集加载成功，共包含 {len(dataset)} 组数据")
        print(len(dataset))
        # 获取第一组数据
        sample = dataset[0]
        print(f"第一组数据包含 {len(sample)} 张图片")
        for key, value in sample.items():
            print(f"{key} 图片形状: {value.shape}")
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
