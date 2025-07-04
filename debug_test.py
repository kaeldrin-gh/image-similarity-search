import tempfile
from pathlib import Path
from img_similarity.data.loader import load_dataset

with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    
    # Create test directory structure
    cat_dir = temp_path / 'cats'
    dog_dir = temp_path / 'dogs'
    cat_dir.mkdir()
    dog_dir.mkdir()
    
    # Create test image files
    (cat_dir / 'cat1.jpg').touch()
    (cat_dir / 'cat2.png').touch()
    (dog_dir / 'dog1.jpg').touch()
    
    # Load dataset
    df = load_dataset(temp_path)
    print('DataFrame shape:', df.shape)
    print(df)
