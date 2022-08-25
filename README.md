Install dependencies with conda:
```
conda create --name anycsp python=3.8
conda activate anycsp
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```
Change "+cu113" to "+cpu" if you have no gpu.

Our validation and test data can be found under the following link:

https://drive.google.com/file/d/1ZZndtrNJfiJRz18aSG_oxwIYzGyKNvrv/view?usp=sharing

Download the file and extract the data in the top level directory of the repository:
```
unzip data.zip
```

Optionally, install our sparse generalized coo_spmm cuda implementation:
```
cd src/spmm_coo
python setup.py install
cd ../..
```
This is not necessary to train or evaluation models, but it will increase memory efficiency and speed up runtime.

We provide scripts for training and testing for each problem of our experiments.
For example, to train a model for MaxCut simply call:
```
bash scripts/train_maxcut.sh
```

To evaluate the models, use the corresponding evaluation scripts:
```
bash scripts/eval_maxcut.sh
```

