# msc_thesis report


## Code usage; 

### Virtual Environment 
```bash
python3 -m venv .venv 
``` 
```bash
source .venv/bin/activate  
```
Deactivate virtual environment and remove venv directory:
```bash
deactivate
rm -r .venv 
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Env. variables 
```bash
source set_env.sh
```

## Segcontrast
https://github.com/PRBonn/segcontrast

```bash
sudo apt install build-essential python3-dev libopenblas-dev
```
```bash
pip3 install --no-cache-dir -r requirements.txt
```
```bash
pip3 install torch ninja
```
```bash
pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
```




## General algortihm flowchart: 
```bash
python3 src/main.py --xodr_path "/mnt/data/bard_gu/xodr/2024-07-25_2129_OKULar_Schwarzer_Berg_ODR.xodr" --pcd_dir "/mnt/data/bard_gu/pcd/ScL/" --output_dir "/mnt/data/bard_gu/pcd/Datasets/SemanticKITTI/dataset/sequences/03"
```

![Code flowchart](img/algo.svg)
