{
  "dataset_reader": {
    "type": "summariser_dataset_reader",
    "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
            },
        }
  },

  "train_data_path": "data/train.json",
  "validation_data_path": "data/dev.json",

  "model": {
    "type": "summarizer-model",
    "word_embeddings": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "embedding_dim": 100,
        "trainable": false
      }
      },
     "encoder": {
          "type": "lstm",
          "bidirectional": true,
          "input_size": 100,
          "hidden_size": 64,
          "num_layers": 1,
        },
       "attention": {
          "type": "lstm",
          "bidirectional": true,
          "input_size": 128,
          "hidden_size": 64,
          "num_layers": 1,
        },
     "classifier": {
      "input_dim": 128,
      "num_layers": 3,
      "hidden_dims": [64, 32, 1],
      "activations": ["linear", "linear", 'sigmoid']
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 1,
    "sorting_keys": [
            [
                "content",
                "list_num_tokens"
            ]
        ],
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 50,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
