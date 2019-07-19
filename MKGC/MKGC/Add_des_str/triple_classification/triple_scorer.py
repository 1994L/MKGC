import tensorflow as tf
import numpy as np
import MKGC.helper_functions as u
from MKGC.Add_des_str.triple_classification import parmeters as p




graph = tf.get_default_graph()



def calculate_triple_score(triple, rel_embd_dict, entity_structure_embd_dict, entity_description_embd_dict):
    head = triple[0]
    tail = triple[1]
    relation = triple[2]

    head_structure_embedding = [entity_structure_embd_dict[head]]
    head_description_embedding = [entity_description_embd_dict[head]]
    rel_embedding = [rel_embd_dict[relation]]
    tail_structure_embedding = [entity_structure_embd_dict[tail]]
    tail_description_embedding = [entity_description_embd_dict[tail]]

    r_input = graph.get_tensor_by_name("input/r_input:0")
    h_pos_structure_input = graph.get_tensor_by_name("input/h_pos_structure_input:0")
    t_pos_structure_input = graph.get_tensor_by_name("input/t_pos_structure_input:0")

    h_pos_description_input = graph.get_tensor_by_name("input/h_pos_description_input:0")
    t_pos_description_input = graph.get_tensor_by_name("input/t_pos_description_input:0")


    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    energy_fusion_pos = graph.get_tensor_by_name("cosine/energy_fusion_pos:0")
    score_fusion_pos = graph.get_tensor_by_name("cosine/score_fusion_pos:0")


    predictions1 = energy_fusion_pos.eval(feed_dict={r_input: rel_embedding,
                                                     h_pos_structure_input: head_structure_embedding,
                                                     t_pos_structure_input: tail_structure_embedding,
                                                     h_pos_description_input: head_description_embedding,
                                                     t_pos_description_input: tail_description_embedding,
                                                     keep_prob: 1.0})

    #print(predictions1)
    predictions2 = score_fusion_pos.eval(feed_dict={r_input: rel_embedding,
                                                    h_pos_structure_input: head_structure_embedding,
                                                    t_pos_structure_input: tail_structure_embedding,
                                                    h_pos_description_input: head_description_embedding,
                                                    t_pos_description_input: tail_description_embedding,
                                                    keep_prob: 1.0})

    score = (predictions1 + predictions2)/2
    return [score]






relation_embeddings = u.load_binary_file(p.relation_embeddings_file, 3)
entity_structure_embeddings = u.load_binary_file(p.text_entity_embeddings_file, 3)
entity_description_embeddings = u.load_binary_file(p.description_entity_embeddings_file, 3)




def main():

    modes = ["valid", "test"]


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        saver = tf.train.import_meta_graph(p.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(p.checkpoint_best_valid_dir))
        print("model loaded", p.checkpoint_best_valid_dir)

        for mode in modes:
            print("Scoring " + mode + " triples")

            if mode == "valid":
                test_triples_file = p.valid_triple_file
                result_file = p.valid_triple_score_file
            else:
                test_triples_file = p.test_triple_file
                result_file = p.test_triple_score_file

            all_test_triples = u.load_triples_with_labels(test_triples_file)

            f = open(result_file, "w")

            for triple in all_test_triples:
                score = calculate_triple_score(triple, relation_embeddings, entity_structure_embeddings,
                                               entity_description_embeddings)
                #print(triple,score[0][0])
                f.write(triple[0] + "\t" + triple[2] + "\t" + triple[1] + "\t" + str(score[0][0]) + "\t" + triple[3] +
                        "\n")
                f.flush()
            f.close()

if __name__ == '__main__':
    main()
