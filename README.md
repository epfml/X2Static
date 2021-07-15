# X2Static vectors
This folder contains the code for training X2Static models.

## Code
The folder src contains the code and instructions to train and reproduce the models trained for our experiments.

## Evaluation
Our models are evaluated using the standard evaluation tool in the [MUSE](https://github.com/facebookresearch/MUSE) repository by Facebook AI Research.

## Pretrained vectors
Pretrained vectors in .bin format can be download from [here](https://zenodo.org/record/5055755). They can be converted to standard word2vec format using the script provided in the src folder. For example, to convert "X2Static_best.bin" to "X2Static_best.vec", use
```
python convert_bin_to_w2v.py X2Static_best.bin X2Static_best.vec
```

## Dataset
Dataset in the required format for the code can be download from [here](https://zenodo.org/record/5055755).

## References
When using this code or some of our pretrained vectors for your application, please cite the following paper:

  Prakhar Gupta,  Martin Jaggi, [*Obtaining Better Static Word Embeddings Using Contextual Embedding Models*](https://arxiv.org/abs/2106.04302) ACL 2021

```
@inproceedings{Gupta2021ObtainingPC,
  title={Obtaining Better Static Word Embeddings Using Contextual Embedding Models},
  author={Prakhar Gupta and Martin Jaggi},
  booktitle={ACL},
  year={2021}
}
```

