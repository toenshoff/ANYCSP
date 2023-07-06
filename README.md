Install dependencies with conda:
```
conda create --name anycsp python=3.10
conda activate anycsp
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
Change "+cu113" to "+cpu" if you have no gpu.

Our validation and test data can be found under the following link:

https://drive.google.com/file/d/1ZZndtrNJfiJRz18aSG_oxwIYzGyKNvrv/view?usp=sharing

Download the file and extract the data in the top-level directory of the repository:
```
unzip data.zip
```

Optionally, install our sparse generalized coo_spmm cuda implementation:
```
cd src/spmm_coo
python setup.py install
cd ../..
```
This is not necessary to train or evaluate models, but it will increase memory efficiency and speed up the runtime.

We provide scripts for training and testing for each problem of our experiments.
For example, to train a model for MaxCut simply call:
```
bash scripts/train_maxcut.sh
```

To evaluate the models, use the corresponding evaluation scripts:
```
bash scripts/eval_maxcut.sh
```

