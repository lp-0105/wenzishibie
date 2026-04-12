import os
import subprocess
import zipfile

def extract_split_zip(zip_file, output_dir):
    """
    自动合并并解压缩分卷 Zip 文件
    """
    if not os.path.exists(zip_file):
        print(f"错误: 找不到主压缩文件 {zip_file}")
        return False

    combined_zip = "temp_combined.zip"
    
    # 1. 使用 zip 命令合并分卷 (Linux/Mac 环境)
    # zip -F data_archive.zip --out temp_combined.zip
    print(f"正在合并分卷文件至 {combined_zip}...")
    try:
        subprocess.run(["zip", "-F", zip_file, "--out", combined_zip], check=True)
    except Exception as e:
        print(f"合并失败: {e}")
        return False

    # 2. 解压合并后的文件
    print(f"正在解压 {combined_zip} 到 {output_dir}...")
    try:
        with zipfile.ZipFile(combined_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("解压成功！")
    except Exception as e:
        print(f"解压失败: {e}")
        return False
    finally:
        # 3. 清理临时合并文件
        if os.path.exists(combined_zip):
            os.remove(combined_zip)
            print("清理临时文件完成。")
            
    return True

if __name__ == "__main__":
    # 配置
    DATA_ZIP = "data_archive.zip"
    TARGET_DIR = "."  # 解压到当前目录，即恢复 data/ 文件夹
    
    print("=== 数据集自动恢复工具 ===")
    if extract_split_zip(DATA_ZIP, TARGET_DIR):
        print("\n数据集已就绪！您现在可以运行: python prepare_data.py")
    else:
        print("\n数据集恢复失败，请检查分卷文件是否完整。")
