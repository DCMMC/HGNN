## Citation
if you find our work useful in your research, please consider citing:

    @article{feng2018hypergraph,
      title={Hypergraph Neural Networks},
      author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
      journal={AAAI 2019},
      year={2018}
    }

## Usage

### Citation

```bash
python train_citation.py
```

Dependencies:

* networkx==1.11

### Point Cloud

**Firstly, you should download the feature files of modelnet40 and ntu2012 datasets.
Then, configure the "data_root" and "result_root" path in config/config.yaml.**

Download datasets for training/evaluation  (should be placed under "data_root")
- [ModelNet40_mvcnn_gvcnn_feature](https://drive.google.com/file/d/1euw3bygLzRQm_dYj1FoRduXvsRRUG2Gr/view?usp=sharing)
- [NTU2012_mvcnn_gvcnn_feature](https://drive.google.com/file/d/1Vx4K15bW3__JPRV0KUoDWtQX8sB-vbO5/view?usp=sharing)



To train and evaluate HGNN for node classification:
```
python train_pointcloud.py
```
You can select the feature that contribute to construct hypregraph incidence matrix by changing the status of parameters "use_mvcnn_feature_for_structure" and "use_gvcnn_feature_for_structure" in config.yaml file. Similarly, changing the status of parameter "use_gvcnn_feature" and "use_gvcnn_feature" can control the feature HGNN feed, and both true will concatenate the mvcnn feature and gvcnn feature as the node feature in HGNN.

```yaml
# config/config.yaml
use_mvcnn_feature_for_structure: True
use_gvcnn_feature_for_structure: True
use_mvcnn_feature: False
use_gvcnn_feature: True
```
To change the experimental dataset (ModelNet40 or NTU2012)
```yaml
# config/config.yaml
#Model
on_dataset: &o_d ModelNet40
#on_dataset: &o_d NTU2012
```