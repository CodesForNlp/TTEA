TTEA: Type-Enhanced ensemble Triple Representation based on Triple-aware Attention for Cross-lingual Entity Alignment

Simplified DBP15K：
* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;

Environment:
* apex
* pytorch
* torch_geometric

Model Training:
* Train.py    semi-supervised
CUDA_VISIBLE_DEVICES=0 python Train.py --data data/DBP15K --lang zh_en

* train1.py   non-semi-supervised
CUDA_VISIBLE_DEVICES=0 python train1.py --data data/DBP15K --lang zh_en

