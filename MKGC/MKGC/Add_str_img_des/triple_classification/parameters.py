import os
triple_classification_base_dir = "./data/WN9-D-IMG/best_result/"



# Triple files
triples_base_dir = "./data/WN9-D-IMG/"
relation_file = "./data/WN9-D-IMG/relations.txt"


# Embeddings files

text_entity_embeddings_file = "./data/WN9-D-IMG/structure.pkl"
relation_embeddings_file = "./data/WN9-D-IMG/structure.pkl"
description_entity_embeddings_file = "./data/WN9-D-IMG/description.pkl"
image_entity_embeddings_file = "./data/WN9-D-IMG/image.pkl"




# weights files
checkpoint_best_valid_dir = "./best_weight/1800_8_best_models_weights/best_WN9D-IMG/"
best_valid_model_meta_file = checkpoint_best_valid_dir + "WN9D-IMG.meta"

model_name = "1800_8"
dataset = "WN9D-IMG"
model_results_dir = triple_classification_base_dir + model_name + "/"
os.mkdir(model_results_dir)

# Validation and test triples score files
valid_triple_file = model_results_dir + "dataset_" + model_name + "_valid.txt"
test_triple_file = model_results_dir + "dataset_" + model_name + "_test.txt"

valid_triple_score_file = model_results_dir + "dataset_" + model_name + "_valid_score.txt"
test_triple_score_file = model_results_dir + "dataset_" + model_name + "_test_score.txt"

triple_classification_results_file = model_results_dir + "triple_classification_results.txt"

