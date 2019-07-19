import tensorflow as tf
import os

relation_structural_embeddings_size = 100
mapping_size = 200
entity_structural_embeddings_size = 100


entity_image_embeddings_size = 1324
entity_description_embeddings_size = 200


entity_multimodal_embeddings_size = 428
#nr_neuron_dense_layer_sum = 100
nr_neuron_dense_layer_1 = 2048
nr_neuron_dense_layer_2 = 1024
dropout_ratio = 0.0
margin = 11
training_epochs = 1500
batch_size = 300
display_step = 1
activation_function = tf.nn.tanh
initial_learning_rate = 0.001


# Loading the data

all_triples_file = "./data/WN9-D-IMG/alls.txt"
train_triples_file = "./data/WN9-D-IMG/train.txt"
test_triples_file = "./data/WN9-D-IMG/test.txt"
valid_triples_file = "./data/WN9-IMG/valid.txt"

entity_structural_embeddings_file = "./data/WN9-D-IMG/structure.pkl"
entity_img_embedding_file = "./data/WN9-D-IMG/image.pkl"
entity_description_file = "./data/WN9-D-IMG/description.pkl"
entity_multimodal_embeddings_file = "./data/WN9-D-IMG/multimodal.pkl"
relation_structural_embeddings_file = "./data/WN9-D-IMG/structure.pkl"


model_id = "WN9D-IMG"
#_mm_loss_10m" #"HMS_standard_vgg128_noreg" #"HMS_standard_full_mapping_elu_300_100"
model_path = "(200)1500_11_best_models_weights/"
checkpoint_best_valid_dir = "./best_weight/" + model_path + "best_" + model_id + "/"
checkpoint_current_dir ="./best_weight/" + model_path + "current_" + model_id + "/"
results_dir = "./best_weight/" + model_path + "results_" + model_id + "/"

# os.makedirs(results_dir, exist_ok=False)

if not os.path.exists(checkpoint_best_valid_dir):
    os.mkdir(checkpoint_best_valid_dir)

if not os.path.exists(checkpoint_current_dir):
    os.mkdir(checkpoint_current_dir)

if not os.path.exists(results_dir):
    os.mkdir(results_dir)


model_current_weights_file = checkpoint_current_dir + model_id + "_current"
current_model_meta_file = checkpoint_current_dir + model_id + "_current.meta"

model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"


result_file = results_dir+model_id+"_results.txt"
log_file = results_dir+model_id+"_log.txt"

