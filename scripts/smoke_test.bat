@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Running Image Similarity Search Smoke Test
echo ==========================================

echo Creating sample data...
mkdir sample_images\cats 2>nul
mkdir sample_images\dogs 2>nul

echo Creating dummy images...
for /l %%i in (1,1,3) do (
    echo dummy image data > "sample_images\cats\cat_%%i.jpg"
    echo dummy image data > "sample_images\dogs\dog_%%i.jpg"
)

echo Creating sample CSV...
(
echo id,image_path,label
echo 1,sample_images/cats/cat_1.jpg,cat
echo 2,sample_images/cats/cat_2.jpg,cat
echo 3,sample_images/cats/cat_3.jpg,cat
echo 4,sample_images/dogs/dog_1.jpg,dog
echo 5,sample_images/dogs/dog_2.jpg,dog
echo 6,sample_images/dogs/dog_3.jpg,dog
) > sample_images\dataset.csv

echo Installing dependencies...
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo Error installing dependencies
    exit /b 1
)

echo Running extraction...
python -m img_similarity extract --data-dir sample_images\dataset.csv --out %TEMP%\vec.npy
if !errorlevel! neq 0 (
    echo Error during extraction
    exit /b 1
)

echo Building index...
python -m img_similarity index --vec %TEMP%\vec.npy --out %TEMP%\idx.faiss
if !errorlevel! neq 0 (
    echo Error building index
    exit /b 1
)

echo Running query...
python -m img_similarity query --image sample_images\cats\cat_1.jpg --index %TEMP%\idx.faiss --metadata %TEMP%\vec.metadata.csv
if !errorlevel! neq 0 (
    echo Error running query
    exit /b 1
)

echo Running evaluation...
python -m img_similarity evaluate --index %TEMP%\idx.faiss --metadata %TEMP%\vec.metadata.csv --embeddings %TEMP%\vec.npy --num-queries 3
if !errorlevel! neq 0 (
    echo Error running evaluation
    exit /b 1
)

echo Cleaning up...
rmdir /s /q sample_images 2>nul
del /q %TEMP%\vec.npy %TEMP%\vec.metadata.csv %TEMP%\idx.faiss 2>nul

echo ==========================================
echo ALL CHECKS PASSED
echo ==========================================
