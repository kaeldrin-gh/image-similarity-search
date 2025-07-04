#!/usr/bin/env bash
set -e

echo "=========================================="
echo "Running Image Similarity Search Smoke Test"
echo "=========================================="

# Create sample data directory
echo "Creating sample data..."
mkdir -p sample_images/cats
mkdir -p sample_images/dogs

# Create some dummy image files for testing
echo "Creating dummy images..."
for i in {1..3}; do
    echo "dummy image data" > "sample_images/cats/cat_$i.jpg"
    echo "dummy image data" > "sample_images/dogs/dog_$i.jpg"
done

# Create a simple CSV file for testing
echo "Creating sample CSV..."
cat > sample_images/dataset.csv << EOF
id,image_path,label
1,sample_images/cats/cat_1.jpg,cat
2,sample_images/cats/cat_2.jpg,cat
3,sample_images/cats/cat_3.jpg,cat
4,sample_images/dogs/dog_1.jpg,dog
5,sample_images/dogs/dog_2.jpg,dog
6,sample_images/dogs/dog_3.jpg,dog
EOF

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running extraction..."
python -m img_similarity extract --data-dir sample_images/dataset.csv --out /tmp/vec.npy

echo "Building index..."
python -m img_similarity index --vec /tmp/vec.npy --out /tmp/idx.faiss

echo "Running query..."
python -m img_similarity query --image sample_images/cats/cat_1.jpg --index /tmp/idx.faiss --metadata /tmp/vec.metadata.csv

echo "Running evaluation..."
python -m img_similarity evaluate --index /tmp/idx.faiss --metadata /tmp/vec.metadata.csv --embeddings /tmp/vec.npy --num-queries 3

echo "Cleaning up..."
rm -rf sample_images/
rm -f /tmp/vec.npy /tmp/vec.metadata.csv /tmp/idx.faiss

echo "=========================================="
echo "ALL CHECKS PASSED"
echo "=========================================="
