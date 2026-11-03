# 先确认 conda 已在 PATH 中
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "conda not found. Install Miniconda/Anaconda first, or add it to PATH." -ForegroundColor Red
    Write-Host "Typical location: %USERPROFILE%\miniconda3\Scripts" -ForegroundColor Yellow
    exit 1
}

conda init powershell

# 1. 创建新环境 (使用 Python 3.10)
Write-Host "Creating environment 'geo_prepare'..."
conda create -n geo_prepare python=3.10 -y

# 2. 激活环境
Write-Host "Activating environment..."
conda activate geo_prepare

# 3. 安装核心地理空间库 (强烈建议使用 conda-forge 渠道，解决 GDAL/GEOS 依赖)
# geopandas 会自动带上 fiona, shapely, pyproj, pandas 等
# rasterio 用于处理栅格数据
Write-Host "Installing geospatial libraries from conda-forge..."
conda install -c conda-forge geopandas rasterio -y

# 4. 安装其他工具库
Write-Host "Installing opencv and tqdm..."
pip install opencv-python tqdm

# 5. 验证安装
Write-Host "Verifying installation..."
python -c "import geopandas; import rasterio; import cv2; import numpy; print(f'Environment ready! NumPy: {numpy.__version__}')"

Write-Host "Done! You can now run the script using: conda activate geo_prepare"
