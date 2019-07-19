import os

import numpy as np
import tensorflow as tf

import MKGC.parameters as param
import MKGC.Cross.util as u

#tf.set_random_seed(1234)
#np.random.seed(7)

logs_path = "log"
# .... Loading the data ....
print("load all triples")
relation_embeddings = u.load_binary_file(param.relation_structural_embeddings_file, 3)
entity_embeddings_structural = u.load_binary_file(param.entity_structural_embeddings_file, 3)
entity_embeddings_description = u.load_binary_file(param.entity_description_file, 3)
entity_embeddings_image = u.load_binary_file(param.entity_img_embedding_file, 3)
entity_embeddings_multimodal = u.load_binary_file(param.entity_multimodal_embeddings_file, 3)

# entity_embeddings_img = u.load_binary_file(param.entity_multimodal_embeddings_file, 3)


all_train_test_valid_triples, entity_list = u.load_training_triples(param.all_triples_file)
triples_set = [t[0] + "_" + t[1] + "_" + t[2] for t in all_train_test_valid_triples]
triples_set = set(triples_set)
entity_list_filtered = []
for e in entity_list:
    # key:bytes = str(e).encode("utf8")
    if e in entity_embeddings_structural:
        entity_list_filtered.append(e)
entity_list = entity_list_filtered
print("#entities", len(entity_list), "#total triples", len(all_train_test_valid_triples))

training_data = u.load_freebase_triple_data_multimodal(param.train_triples_file, entity_embeddings_structural,
                                                       entity_embeddings_description, entity_embeddings_image,
                                                       entity_embeddings_multimodal, relation_embeddings)

#training_data= training_data[:1000]
#training_data = training_data[:10000]
print("#training data", len(training_data))

valid_data = u.load_freebase_triple_data_multimodal(param.valid_triples_file, entity_embeddings_structural,
                                                    entity_embeddings_description, entity_embeddings_image,
                                                    entity_embeddings_multimodal, relation_embeddings)
#valid_data = valid_data[:len(valid_data)//3]

print("valid_data", len(valid_data))

def max_norm_regulizer(threshold, axes=1, name="max_norm", collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes) # 控制梯度
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None
    return max_norm

max_norm_reg = max_norm_regulizer(threshold=1.0)

def my_dense(x, nr_hidden, scope, activation_fn=param.activation_function, reuse=None):
    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(x, nr_hidden,
                                              activation_fn=activation_fn,
                                              reuse=reuse,
                                              scope=scope
                                              )  # weights_regularizer= max_norm_reg

        return h



# ........... Creating the model
with tf.name_scope('input'):
    r_input = tf.placeholder(dtype=tf.float32, shape=[None, param.relation_structural_embeddings_size], name="r_input")

    # head entity

    h_pos_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size],
                                           name="h_pos_structure_input")
    h_neg_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size],
                                           name="h_neg_structure_input")

    h_pos_description_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_description_embeddings_size],
                                             name="h_pos_description_input")
    h_neg_description_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_description_embeddings_size],
                                             name="h_neg_description_input")

    h_pos_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size],
                                       name="h_pos_image_input")
    h_neg_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size],
                                       name="h_neg_image_input")

    h_pos_multi_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size],
                                       name="h_pos_multi_input")
    h_neg_multi_inpit = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size],
                                       name="h_neg_multi_input")

    # tail entity

    t_pos_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size],
                                           name="t_pos_structure_input")
    t_neg_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size],
                                           name="t_neg_structure_input")

    t_pos_description_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_description_embeddings_size],
                                             name="t_pos_description_input")
    t_neg_description_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_description_embeddings_size],
                                             name="t_neg_description_input")

    t_pos_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size],
                                       name="t_pos_image_input")
    t_neg_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size],
                                       name="t_neg_image_input")

    t_pos_multi_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size],
                                       name="t_pos_multi_input")
    t_neg_multi_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size],
                                       name="t_neg_multi_input")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    print('1')

with tf.name_scope('head_relation'):
    # relation
    r_mapped = my_dense(r_input, param.mapping_size, activation_fn=param.activation_function, scope="structure_proj",
                        reuse=None) # 向量变换
    r_mapped = tf.nn.dropout(r_mapped, keep_prob)

    # structure

    h_pos_structure_mapped = my_dense(h_pos_structure_input, param.mapping_size, activation_fn=param.activation_function,
                                      scope="structure_proj", reuse=True)
    h_pos_structure_mapped = tf.nn.dropout(h_pos_structure_mapped, keep_prob)

    h_neg_structure_mapped = my_dense(h_neg_structure_input, param.mapping_size, activation_fn=param.activation_function,
                                      scope="structure_proj", reuse=True)
    h_neg_structure_mapped = tf.nn.dropout(h_neg_structure_mapped, keep_prob)

    t_pos_structure_mapped = my_dense(t_pos_structure_input, param.mapping_size, activation_fn=param.activation_function,
                                      scope="structure_proj", reuse=True)
    t_pos_structure_mapped = tf.nn.dropout(t_pos_structure_mapped, keep_prob)

    t_neg_structure_mapped = my_dense(t_neg_structure_input, param.mapping_size, activation_fn=param.activation_function,
                                      scope="structure_proj", reuse=True)
    t_neg_structure_mapped = tf.nn.dropout(t_neg_structure_mapped, keep_prob)


    # Visual
    h_pos_image_mapped = my_dense(h_pos_image_input, param.mapping_size, activation_fn=param.activation_function,
                                  scope="image_proj", reuse=None)
    h_pos_image_mapped = tf.nn.dropout(h_pos_image_mapped, keep_prob)

    h_neg_image_mapped = my_dense(h_neg_image_input, param.mapping_size, activation_fn=param.activation_function,
                                  scope="image_proj", reuse=True)
    h_neg_image_mapped = tf.nn.dropout(h_neg_image_mapped, keep_prob)

    t_pos_image_mapped = my_dense(t_pos_image_input, param.mapping_size, activation_fn=param.activation_function,
                                  scope="image_proj", reuse=True)
    t_pos_image_mapped = tf.nn.dropout(t_pos_image_mapped, keep_prob)

    t_neg_image_mapped = my_dense(t_neg_image_input, param.mapping_size, activation_fn=param.activation_function,
                                  scope="image_proj", reuse=True)
    t_neg_image_mapped = tf.nn.dropout(t_neg_image_mapped, keep_prob)

    # description

    h_pos_description_mapped = my_dense(h_pos_description_input, param.mapping_size,
                                        activation_fn=param.activation_function,
                                        scope='description_proj', reuse=None)
    h_pos_description_mapped = tf.nn.dropout(h_pos_description_mapped, keep_prob)

    h_neg_description_mapped = my_dense(h_neg_description_input, param.mapping_size,
                                        activation_fn=param.activation_function,
                                        scope='description_proj', reuse=True)
    h_neg_description_mapped = tf.nn.dropout(h_neg_description_mapped, keep_prob)

    t_pos_description_mapped = my_dense(t_pos_description_input, param.mapping_size,
                                        activation_fn=param.activation_function,
                                        scope='description_proj', reuse=True)
    t_pos_description_mapped = tf.nn.dropout(t_pos_description_mapped, keep_prob)


    t_neg_description_mapped = my_dense(t_neg_description_input, param.mapping_size,
                                        activation_fn=param.activation_function,
                                        scope='description_proj', reuse=True)
    t_neg_description_mapped = tf.nn.dropout(t_neg_description_mapped, keep_prob)


    # multi-modal

    h_pos_multi_mapped = my_dense(h_pos_multi_input, param.mapping_size,
                                  activation_fn=param.activation_function,
                                  scope='multi_proj', reuse=None)
    h_pos_multi_mapped = tf.nn.dropout(h_pos_multi_mapped, keep_prob)

    h_neg_multi_mapped = my_dense(h_neg_multi_inpit, param.mapping_size, activation_fn=param.activation_function,
                                  scope='multi_proj', reuse=True)
    h_neg_multi_mapped = tf.nn.dropout(h_neg_multi_mapped, keep_prob)

    t_pos_multi_mapped = my_dense(t_pos_multi_input, param.mapping_size, activation_fn=param.activation_function,
                                  scope='multi_proj', reuse=True)
    t_pos_multi_mapped = tf.nn.dropout(t_pos_multi_mapped, keep_prob)

    t_neg_multi_mapped = my_dense(t_neg_multi_input, param.mapping_size, activation_fn=param.activation_function,
                                  scope='multi_proj', reuse=True)
    t_neg_multi_mapped = tf.nn.dropout(t_neg_multi_mapped, keep_prob)




with tf.name_scope('cosine'):
    # fusion structure, description, image

    # h_fusion_mapped_pos = h_pos_structure_mapped + h_pos_description_mapped + h_pos_image_mapped
    # h_fusion_mapped_neg = h_neg_structure_mapped + h_neg_description_mapped + h_neg_image_mapped
    #
    # t_fusion_mapped_pos = t_pos_structure_mapped + t_pos_description_mapped + t_pos_image_mapped
    # t_fusion_mapped_neg = t_neg_structure_mapped + t_neg_description_mapped + t_neg_image_mapped
    #
    # # Head model
    #
    # energy_fusion_pos = tf.reduce_sum(abs(h_fusion_mapped_pos + r_mapped - t_fusion_mapped_pos), 1,
    #                                   keep_dims=True, name="energy_fusion_pos")
    # energy_fusion_neg = tf.reduce_sum(abs(h_fusion_mapped_pos + r_mapped - t_fusion_mapped_neg), 1,
    #                                   keep_dims=True, name="energy_fusion_neg")
    energy_ss_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_pos_structure_mapped), 1, keep_dims=True,
                                  name="pos_s_s")
    energy_ss_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_neg_structure_mapped), 1, keep_dims=True,
                                  name="neg_s_s")

    energy_sm_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_pos_multi_mapped), 1, keep_dims=True,
                                  name="pos_s_m")
    energy_sm_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_neg_multi_mapped), 1, keep_dims=True,
                                  name="neg_s_m")

    energy_ms_pos = tf.reduce_sum(abs(h_pos_multi_mapped + r_mapped - t_pos_structure_mapped), 1, keep_dims=True,
                                  name="pos_m_s")
    energy_ms_neg = tf.reduce_sum(abs(h_pos_multi_mapped + r_mapped - t_neg_structure_mapped), 1, keep_dims=True,
                                  name="neg_m_s")

    energy_mm_pos = tf.reduce_sum(abs(h_pos_multi_mapped + r_mapped - t_pos_multi_mapped), 1, keep_dims=True,
                                  name="pos_m_m")
    energy_mm_neg = tf.reduce_sum(abs(h_pos_multi_mapped + r_mapped - t_neg_multi_mapped), 1, keep_dims=True,
                                  name="neg_m_m")

    energy_sd_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_pos_description_mapped), 1, keep_dims=True,
                                  name="pos_s_d")
    energy_sd_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_neg_description_mapped), 1, keep_dims=True,
                                  name="neg_s_d")

    energy_si_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_pos_image_mapped), 1, keep_dims=True,
                                  name="pos_s_i")
    energy_si_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_neg_image_mapped), 1, keep_dims=True,
                                  name="neg_s_i")

    energy_dd_pos = tf.reduce_sum(abs(h_pos_description_mapped + r_mapped - t_pos_description_mapped), 1, keep_dims=True,
                                  name="pos_d_d")
    energy_dd_neg = tf.reduce_sum(abs(h_pos_description_mapped + r_mapped - t_neg_description_mapped), 1, keep_dims=True,
                                  name="neg_d_d")

    energy_ds_pos = tf.reduce_sum(abs(h_pos_description_mapped + r_mapped - t_pos_structure_mapped), 1, keep_dims=True,
                                  name="pos_d_s")
    energy_ds_neg = tf.reduce_sum(abs(h_pos_description_mapped + r_mapped - t_neg_structure_mapped), 1, keep_dims=True,
                                  name="neg_d_s")

    energy_di_pos = tf.reduce_sum(abs(h_pos_description_mapped + r_mapped - t_pos_image_mapped), 1, keep_dims=True,
                                  name="pos_d_i")
    energy_di_neg = tf.reduce_sum(abs(h_pos_description_mapped + r_mapped - t_neg_image_mapped), 1, keep_dims=True,
                                  name="neg_d_i")

    energy_ii_pos = tf.reduce_sum(abs(h_pos_image_mapped + r_mapped - t_pos_image_mapped), 1, keep_dims=True,
                                  name="pos_i_i")
    energy_ii_neg = tf.reduce_sum(abs(h_pos_image_mapped + r_mapped - t_neg_image_mapped), 1, keep_dims=True,
                                  name="neg_i_i")

    energy_id_pos = tf.reduce_sum(abs(h_pos_image_mapped + r_mapped - t_pos_description_mapped), 1, keep_dims=True,
                                  name="pos_i_d")
    energy_id_neg = tf.reduce_sum(abs(h_pos_image_mapped + r_mapped - t_neg_description_mapped), 1, keep_dims=True,
                                  name="neg_i_d")

    energy_is_pos = tf.reduce_sum(abs(h_pos_image_mapped + r_mapped - t_pos_structure_mapped), 1, keep_dims=True,
                                  name="pos_i_s")
    energy_is_neg = tf.reduce_sum(abs(h_pos_image_mapped + r_mapped - t_neg_structure_mapped), 1, keep_dims=True,
                                  name="neg_i_s")

    # energy_concat_pos = tf.reduce_sum(abs((h_pos_structure_mapped + h_pos_description_mapped + h_pos_image_mapped) +
    #                                       r_mapped - (t_pos_structure_mapped + t_pos_description_mapped) +
    #                                       t_pos_image_mapped), 1, keep_dims=True, name="pos_concat")
    # energy_concat_neg = tf.reduce_sum(abs(h_pos_structure_mapped + h_pos_description_mapped + h_pos_image_mapped) +
    #                                   r_mapped - (t_neg_structure_mapped + t_neg_description_mapped + t_neg_image_mapped),
    #                                   1, keep_dims=True, name="neg_concat")

    # h_r_t_pos = tf.reduce_sum([1 * energy_ss_pos, 0.5 * energy_sd_pos,  0.5 * energy_si_pos, 1 * energy_dd_pos, 0.5 * energy_ds_pos,  0.5 * energy_di_pos,  1 * energy_ii_pos,  0.5 *energy_id_pos, 0.5 * energy_is_pos,  1 * energy_concat_pos], 0,
    #                           name="h_r_t_pos")
    h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_sd_pos, energy_si_pos, energy_dd_pos, energy_ds_pos,  energy_di_pos,  energy_ii_pos, energy_id_pos, energy_is_pos,
                               energy_sm_pos, energy_ms_pos, energy_mm_pos], 0,
                              name="h_r_t_pos")
    # h_r_t_pos = tf.divide(h_r_t_pos, tf.cast(7, dtype=tf.float32), name='energy_pos')
    # h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_sd_pos, energy_si_pos, energy_dd_pos, energy_ds_pos, energy_di_pos, energy_ii_pos,  energy_id_pos, energy_is_pos, energy_concat_pos], 0, name="h_r_t_pos")

    # h_r_t_neg = tf.reduce_sum([1 * energy_ss_neg, 0.5 * energy_sd_neg, 0.5 * energy_si_neg, 1 * energy_dd_neg,  0.5 * energy_ds_neg, 0.5 * energy_di_neg,  1 * energy_ii_neg, 0.5 * energy_id_neg, 0.5 * energy_is_neg, 1 * energy_concat_neg], 0,
    #                           name="h_r_t_neg")
    h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_sd_neg, energy_si_neg, energy_dd_neg, energy_ds_neg, energy_di_neg, energy_ii_neg, energy_id_neg, energy_is_neg,
                               energy_sm_neg, energy_ms_neg, energy_mm_neg], 0,
                              name="h_r_t_neg")
    # h_r_t_neg = tf.divide(h_r_t_neg, tf.cast(7, dtype=tf.float32), name='energy_neg')

    # h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_sd_neg, energy_si_neg, energy_dd_neg, energy_ds_neg, energy_di_neg, energy_ii_neg, energy_id_neg, energy_is_neg, energy_concat_neg], 0,
    #                           name="h_r_t_neg")


    # Head model
    # energy_ss_pos = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_pos_txt_mapped), 1, keep_dims=True, name="pos_s_s")
    # energy_ss_neg = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_neg_txt_mapped), 1, keep_dims=True, name="neg_s_s")
    #
    # energy_is_pos = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_pos_txt_mapped), 1, keep_dims=True, name="pos_i_i")
    # energy_is_neg = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_neg_txt_mapped), 1, keep_dims=True, name="neg_i_i")
    #
    # energy_si_pos = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_pos_img_mapped), 1, keep_dims=True, name="pos_s_i")
    # energy_si_neg = tf.reduce_sum(abs(h_pos_txt_mapped + r_mapped - t_neg_img_mapped), 1, keep_dims=True, name="neg_s_i")
    #
    # energy_ii_pos = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_pos_img_mapped), 1, keep_dims=True, name="pos_i_i")
    # energy_ii_neg = tf.reduce_sum(abs(h_pos_img_mapped + r_mapped - t_neg_img_mapped), 1, keep_dims=True, name="neg_i_i")
    #
    # energy_concat_pos = tf.reduce_sum(abs((h_pos_txt_mapped + h_pos_img_mapped) + r_mapped - (t_pos_txt_mapped + t_pos_img_mapped)), 1, keep_dims=True, name="energy_concat_pos")
    # energy_concat_neg = tf.reduce_sum(abs((h_pos_txt_mapped + h_pos_img_mapped) + r_mapped - (t_neg_txt_mapped + t_neg_img_mapped)), 1, keep_dims=True, name="energy_concat_neg")
    #
    # h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_is_pos, energy_si_pos, energy_ii_pos, energy_concat_pos], 0,   name="h_r_t_pos")
    # h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_is_neg, energy_si_neg, energy_ii_neg, energy_concat_neg], 0, name="h_r_t_neg")

    #Tail model

    # score_fusion_pos = tf.reduce_sum(abs(t_fusion_mapped_pos - r_mapped - h_fusion_mapped_pos), 1,
    #                                  keep_dims=True, name="score_fusion_pos")
    # score_fusion_neg = tf.reduce_sum(abs(t_fusion_mapped_pos - r_mapped - h_fusion_mapped_neg), 1,
    #                                  keep_dims=True, name="score_fusion_neg")


    score_ss_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_pos_structure_mapped), 1, keep_dims=True,
                                 name="pos_s_s")
    score_ss_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_neg_structure_mapped), 1, keep_dims=True,
                                 name="neg_s_s")

    score_sd_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_pos_description_mapped), 1, keep_dims=True,
                                 name="pos_s_d")
    score_sd_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_neg_description_mapped), 1, keep_dims=True,
                                 name="neg_s_d")

    score_si_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_pos_image_mapped), 1, keep_dims=True,
                                 name="pos_s_i")
    score_si_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_neg_image_mapped), 1, keep_dims=True,
                                 name="neg_s_i")

    score_dd_pos = tf.reduce_sum(abs(t_pos_description_mapped - r_mapped - h_pos_description_mapped), 1, keep_dims=True,
                                 name="pos_d_d")
    score_dd_neg = tf.reduce_sum(abs(t_pos_description_mapped - r_mapped - h_neg_description_mapped), 1, keep_dims=True,
                                 name="neg_d_d")

    score_ds_pos = tf.reduce_sum(abs(t_pos_description_mapped - r_mapped - h_pos_structure_mapped), 1, keep_dims=True,
                                 name="pos_d_s")
    score_ds_neg = tf.reduce_sum(abs(t_pos_description_mapped - r_mapped - h_neg_structure_mapped), 1, keep_dims=True,
                                 name="neg_d_s")

    score_di_pos = tf.reduce_sum(abs(t_pos_description_mapped - r_mapped - h_pos_image_mapped), 1, keep_dims=True,
                                 name="pos_d_i")
    score_di_neg = tf.reduce_sum(abs(t_pos_description_mapped - r_mapped - h_neg_image_mapped), 1, keep_dims=True,
                                 name="neg_d_i")

    score_ii_pos = tf.reduce_sum(abs(t_pos_image_mapped - r_mapped - h_pos_image_mapped), 1, keep_dims=True,
                                 name="pos_i_i")
    score_ii_neg = tf.reduce_sum(abs(t_pos_image_mapped - r_mapped - h_neg_image_mapped), 1, keep_dims=True,
                                 name="neg_i_i")

    score_is_pos = tf.reduce_sum(abs(t_pos_image_mapped - r_mapped - h_pos_structure_mapped), 1, keep_dims=True,
                                 name="pos_i_s")
    score_is_neg = tf.reduce_sum(abs(t_pos_image_mapped - r_mapped - h_neg_structure_mapped), 1, keep_dims=True,
                                 name="neg_i_s")

    score_id_pos = tf.reduce_sum(abs(t_pos_image_mapped - r_mapped - h_pos_description_mapped), 1, keep_dims=True,
                                 name="pos_i_d")
    score_id_neg = tf.reduce_sum(abs(t_pos_image_mapped - r_mapped - h_neg_description_mapped), 1, keep_dims=True,
                                 name="neg_i_d")

    score_sm_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_pos_multi_mapped), 1, keep_dims=True,
                                 name="pos_s_m")
    score_sm_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_neg_multi_mapped),1, keep_dims=True,
                                 name="neg_s_m")

    score_ms_pos = tf.reduce_sum(abs(t_pos_multi_mapped - r_mapped - h_pos_structure_mapped), 1, keep_dims=True,
                                 name="pos_m_s")
    score_ms_neg = tf.reduce_sum(abs(t_pos_multi_mapped - r_mapped - h_neg_structure_mapped), 1, keep_dims=True,
                                 name="neg_m_s")

    score_mm_pos = tf.reduce_sum(abs(t_pos_multi_mapped - r_mapped - h_pos_multi_mapped), 1, keep_dims=True,
                                 name="pos_m_m")
    score_mm_neg = tf.reduce_sum(abs(t_pos_multi_mapped - r_mapped - h_neg_multi_mapped), 1, keep_dims=True,
                                 name="neg_m_m")

    # score_concat_pos = tf.reduce_sum(abs((t_pos_structure_mapped + t_pos_image_mapped + t_pos_image_mapped) - r_mapped -
    #                                      (h_pos_structure_mapped + h_pos_description_mapped + h_pos_image_mapped)), 1, keep_dims=True,
    #                                  name="score_concat_pos")
    # score_concat_neg = tf.reduce_sum(abs((t_pos_structure_mapped + t_pos_description_mapped + t_pos_image_mapped) - r_mapped -
    #                                      (h_neg_structure_mapped + h_neg_description_mapped + h_neg_image_mapped)), 1, keep_dims=True,
    #                                  name="score_concat_neg")

    # t_r_h_pos = tf.reduce_sum([1 * score_ss_pos, 0.5 * score_sd_pos, 0.5 * score_si_pos, 1 * score_dd_pos, 0.5 * score_ds_pos, 0.5 * score_di_pos, 1 * score_ii_pos, 0.5 * score_is_pos, 0.5 * score_id_pos, 1 * score_concat_pos], 0,
    #                           name="t_r_h_pos")
    t_r_h_pos = tf.reduce_sum([score_ss_pos, score_sd_pos, score_si_pos, score_dd_pos, score_ds_pos, score_di_pos, score_ii_pos, score_is_pos, score_id_pos, score_sm_pos,
                               score_ms_pos, score_mm_pos], 0,
                              name="t_r_h_pos")
    # t_r_h_pos = tf.divide(t_r_h_pos, tf.cast(7, dtype=tf.float32), name='score_pos')
    # t_r_h_neg = tf.reduce_sum([1 * score_ss_neg, 0.5 * score_sd_neg, 0.5 * score_si_neg, 1 * score_dd_neg, 0.5 * score_ds_neg, 0.5 * score_di_neg, 1 * score_ii_neg, 0.5 * score_is_neg, 0.5 * score_id_neg, 1 * score_concat_neg], 0,
    #                           name="t_r_h_neg")
    t_r_h_neg = tf.reduce_sum([score_ss_neg, score_sd_neg, score_si_neg, score_dd_neg, score_ds_neg, score_di_neg, score_ii_neg, score_is_neg, score_id_neg, score_sm_neg,
                               score_ms_neg, score_mm_neg], 0,
                              name="t_r_h_neg")

    # t_r_h_neg = tf.divide(t_r_h_neg, tf.cast(7, dtype=tf.float32), name='score_neg')
    # Tail model

    # score_t_t_pos = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_pos_txt_mapped), 1, keep_dims=True, name="pos_s_s")
    # score_t_t_neg = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_neg_txt_mapped), 1, keep_dims=True, name="neg_s_s")
    #
    # score_i_t_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_pos_txt_mapped), 1, keep_dims=True, name="pos_i_i")
    # score_i_t_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_neg_txt_mapped), 1, keep_dims=True, name="neg_i_i")
    #
    # score_t_i_pos = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_pos_img_mapped), 1, keep_dims=True, name="pos_s_i")
    # score_t_i_neg = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_neg_img_mapped), 1, keep_dims=True, name="neg_s_i")
    #
    # score_i_i_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_pos_img_mapped), 1, keep_dims=True, name="pos_i_i")
    # score_i_i_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_neg_img_mapped), 1, keep_dims=True, name="neg_i_i")
    #
    # energy_concat_pos_tail = tf.reduce_sum(abs((t_pos_txt_mapped + t_pos_img_mapped) - r_mapped - (h_pos_txt_mapped + h_pos_img_mapped)), 1, keep_dims=True, name="energy_concat_pos_tail")
    # energy_concat_neg_tail = tf.reduce_sum(abs((t_pos_txt_mapped + t_pos_img_mapped) - r_mapped - (h_neg_txt_mapped + h_neg_img_mapped)), 1, keep_dims=True, name="energy_concat_neg_tail")
    #
    # t_r_h_pos = tf.reduce_sum([score_t_t_pos, score_i_t_pos, score_t_i_pos, score_i_i_pos,energy_concat_pos_tail], 0, name="t_r_h_pos")
    # t_r_h_neg = tf.reduce_sum( [score_t_t_neg, score_i_t_neg, score_t_i_neg, score_i_i_neg,energy_concat_neg_tail], 0,name="t_r_h_neg")


    # kbc_loss1 = tf.maximum(0., param.margin - h_r_t_neg + h_r_t_pos)
    # kbc_loss2 = tf.maximum(0., param.margin - t_r_h_neg + t_r_h_pos)

    kbc_loss1 = tf.maximum(0., param.margin - h_r_t_neg + h_r_t_pos)
    kbc_loss2 = tf.maximum(0., param.margin - t_r_h_neg + t_r_h_pos)


    kbc_loss = kbc_loss1 + kbc_loss2

    tf.summary.histogram("loss", kbc_loss)

#epsilon= 0.1
optimizer = tf.train.AdamOptimizer().minimize(kbc_loss)

summary_op = tf.summary.merge_all()

#..... start the training
saver = tf.train.Saver()
log_file = open(param.log_file, "w")

log_file.write("relation_input_size = " + str(param.relation_structural_embeddings_size) + "\n")
log_file.write("entity_input_size = " + str(param.entity_structural_embeddings_size) + "\n")
log_file.write("image_input_size = " + str(param.entity_image_embeddings_size) + "\n")
log_file.write("description_input_size = " + str(param.entity_description_embeddings_size) + "\n")
log_file.write("multimodal_input_size = " + str(param.entity_multimodal_embeddings_size) + "\n")
log_file.write("nr_neuron_dense_layer1 = " + str(param.nr_neuron_dense_layer_1) + "\n")
log_file.write("nr_neuron_dense_layer2 = " + str(param.nr_neuron_dense_layer_2) + "\n")
log_file.write("dropout_ratio = " + str(param.dropout_ratio) + "\n")
log_file.write("margin = " + str(param.margin) + "\n")
log_file.write("training_epochs = " + str(param.training_epochs) + "\n")
log_file.write("batch_size = " + str(param.batch_size) + "\n")
log_file.write("activation_function = " + str(param.activation_function) + "\n")
log_file.write("initial_learning_rate = " + str(param.initial_learning_rate) + "\n")

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

#np.random.shuffle(valid_data)

h_data_valid_structure, h_data_valid_description, h_data_valid_image, r_data_valid, \
t_data_valid_structure, t_data_valid_description, t_data_valid_image, t_neg_data_valid_structure, t_neg_data_valid_description, t_neg_data_valid_image, \
h_neg_data_valid_stucture, h_neg_data_valid_description, h_neg_data_valid_image, h_data_valid_multi, t_data_valid_multi, \
    h_neg_data_valid_multi, t_neg_data_valid_multi = \
    u.get_batch_with_neg_heads_and_neg_tails_multimodal(valid_data,
                                                        triples_set,
                                                        entity_list,
                                                        0, len(valid_data),
                                                        entity_embeddings_structural,
                                                        entity_embeddings_description,
                                                        entity_embeddings_image,
                                                        entity_embeddings_multimodal)
#clip_all_weights = tf.get_collection("max_norm")

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(param.best_valid_model_meta_file):
        print("restore the weights", param.checkpoint_best_valid_dir)
        saver = tf.train.import_meta_graph(param.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(param.checkpoint_best_valid_dir))
    else:
        print("no weights to load :(")




    initial_valid_loss = 100


    for epoch in range(param.training_epochs):

        np.random.shuffle(training_data)

       # training_data2 = training_data[:len(training_data)//3]
        training_loss = 0.
        total_batch = len(training_data) // param.batch_size + 1
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for i in range(total_batch):

            batch_loss = 0
            start = i * param.batch_size
            end = (i + 1) * param.batch_size

            h_data_structure, h_data_description, h_data_image, r_data, t_data_structure, \
            t_data_description, t_data_image, t_neg_data_structure, t_neg_data_description, t_neg_data_image, \
            h_neg_data_structure, h_neg_data_description, h_neg_data_image, h_data_multi, t_data_multi, h_neg_data_multi, \
            t_neg_data_multi = u.get_batch_with_neg_heads_and_neg_tails_multimodal(
                training_data, triples_set, entity_list, start,
                end, entity_embeddings_structural, entity_embeddings_description, entity_embeddings_image,
                entity_embeddings_multimodal)

            _, loss, summary = sess.run(
                [optimizer, kbc_loss, summary_op],
                feed_dict={r_input: r_data,
                           h_pos_structure_input: h_data_structure,
                           h_pos_description_input: h_data_description,
                           h_pos_image_input: h_data_image,
                           h_pos_multi_input: h_data_multi,

                           t_pos_structure_input: t_data_structure,
                           t_pos_description_input: t_data_description,
                           t_pos_image_input: t_data_image,
                           t_pos_multi_input: t_data_multi,

                           t_neg_structure_input: t_neg_data_structure,
                           t_neg_description_input: t_neg_data_description,
                           t_neg_image_input: t_neg_data_image,
                           t_neg_multi_input: t_neg_data_multi,

                           h_neg_structure_input: h_neg_data_structure,
                           h_neg_description_input: h_neg_data_description,
                           h_neg_image_input: h_neg_data_image,
                           h_neg_multi_inpit: h_neg_data_multi,
                           # learning_rate: param.initial_learning_rate
                           keep_prob: 1 - param.dropout_ratio,
                           # learning_rate: param.initial_learning_rate
                           })
            #sess.run(clip_all_weights)

            batch_loss = np.sum(loss)/param.batch_size

            training_loss += batch_loss

            writer.add_summary(summary, epoch * total_batch + i)

        training_loss = training_loss / total_batch

        # validating by sampling every epoch


        val_loss = sess.run([kbc_loss],
                            feed_dict={r_input: r_data_valid,
                                       h_pos_structure_input: h_data_valid_structure,
                                       h_pos_description_input: h_data_valid_description,
                                       h_pos_image_input: h_data_valid_image,
                                       h_pos_multi_input: h_data_valid_multi,

                                       t_pos_structure_input: t_data_valid_structure,
                                       t_pos_description_input: t_data_valid_description,
                                       t_pos_image_input: t_data_valid_image,
                                       t_pos_multi_input: t_data_valid_multi,

                                       t_neg_structure_input: t_neg_data_valid_structure,
                                       t_neg_description_input: t_neg_data_valid_description,
                                       t_neg_image_input: t_neg_data_valid_image,
                                       t_neg_multi_input: t_neg_data_valid_multi,

                                       h_neg_structure_input: h_neg_data_valid_stucture,
                                       h_neg_description_input: h_neg_data_valid_description,
                                       h_neg_image_input: h_neg_data_valid_image,
                                       h_neg_multi_inpit: h_neg_data_valid_multi,

                                       keep_prob: 1
                                       })

        val_score = np.sum(val_loss) / len(valid_data)


        print("Epoch:", (epoch + 1), "loss=", str(round(training_loss, 4)), "val_loss", str(round(val_score, 4)))

        if val_score < initial_valid_loss:
            saver.save(sess, param.model_weights_best_valid_file)
            log_file.write("save model best validation loss: " + str(initial_valid_loss) + "==>" + str(val_score) + "\n")
            print("save model valid loss: ", str(initial_valid_loss), "==>", str(val_score))
            initial_valid_loss = val_score


        saver.save(sess, param.model_current_weights_file)

        log_file.write("Epoch:\t" + str(epoch + 1) + "\tloss:\t" + str(round(training_loss, 5)) + "\tval_loss:\t" + str(
            round(val_score, 5)) + "\n")
        log_file.flush()

        writer.close()




