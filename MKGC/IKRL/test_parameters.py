# Specify the parameters for the test

entity_structural_embeddings_file = "./data/WN9-D-IMG/structure.pkl"
entity_img_embedding_file = "./data/WN9-D-IMG/image.pkl"
entity_description_file = "./data/WN9-D-IMG/description.pkl"
entity_multimodal_embeddings_file = "./data/WN9-D-IMG/multimodel.pkl"
relation_structural_embeddings_file = "./data/WN9-D-IMG/structure.pkl"


all_triples_file = "./data/WN9-D-IMG/all.txt"
test_triples_file = "./data/WN9-D-IMG/test.txt"



model_id = "WN9D-IMG"

# where to load the weights for the model
checkpoint_best_valid_dir = "./best_weight/(IKRL)1500_10_best_models_weights/best_"+model_id+"/"
model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"

checkpoint_current_dir ="./best_weight/(IKRL)1500_10_best_models_weights/current_"+model_id+"/"
model_current_weights_file = checkpoint_current_dir + model_id + "_current"
current_model_meta_file = checkpoint_current_dir + model_id + "_current.meta"

# Results location
results_dir = "./best_weight/(IKRL)1500_10_best_models_weights/results_"+model_id+"/"
result_file = results_dir+model_id+"_results.txt"

