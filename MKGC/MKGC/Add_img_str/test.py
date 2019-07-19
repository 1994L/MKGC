import operator
import os

import numpy as np
import tensorflow as tf

import MKGC.test_parameters as param
from MKGC.Add_img_str import util as u

graph = tf.get_default_graph()

batch_size = 300


def predict_best_tail(test_triple, full_triple_list, full_entity_list,
                      entity_embeddings_structure,
                      entity_embeddings_description,
                      full_relation_embeddings):
    results = {}
    gt_head = test_triple[0]
    gt_head_embeddings_structure = entity_embeddings_structure[gt_head]
    get_head_embeddings_description = entity_embeddings_description[gt_head]

    gt_rel = test_triple[2]
    gt_relation_embeddings = full_relation_embeddings[gt_rel]
    gt_tail_org = test_triple[1]
    gt_tail = u.get_correct_tails(gt_head, gt_rel, full_triple_list)


    total_batches = len(full_entity_list)//batch_size +1

    predictions = []
    for batch_i in range(total_batches):
        start = batch_size * (batch_i)
        end = batch_size * (batch_i + 1)


        tails_embeddings_list_structure = []
        tails_embeddings_list_description = []

        head_embeddings_list_structure = np.tile(gt_head_embeddings_structure, (len(full_entity_list[start:end]), 1))
        head_embeddings_list_description = np.tile(get_head_embeddings_description, (len(full_entity_list[start:end]), 1))
        full_relation_embeddings = np.tile(gt_relation_embeddings, (len(full_entity_list[start:end]), 1))


        for i in range(len(full_entity_list[start:end])):

            tails_embeddings_list_structure.append(entity_embeddings_structure[full_entity_list[start+i]])
            tails_embeddings_list_description.append(entity_embeddings_description[full_entity_list[start+i]])

        sub_predictions = predict_tail(head_embeddings_list_structure, head_embeddings_list_description,
                                       full_relation_embeddings, tails_embeddings_list_structure,
                                       tails_embeddings_list_description)
        for p in sub_predictions:
            predictions.append(p)

    predictions = [predictions]
    for i in range(0, len(predictions[0])):
        if full_entity_list[i] == gt_head and gt_head not in gt_tail:
            pass
            #results[full_entity_list[i]] = 0
        else:
            results[full_entity_list[i]] = predictions[0][i]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    top_10_predictions = [x[0] for x in sorted_x[:10]]
    top_1_predictions = [x[0] for x in sorted_x[:1]]
    sorted_keys = [x[0] for x in sorted_x]
    index_correct_tail_raw = sorted_keys.index(gt_tail_org)

    gt_tail_to_filter = [x for x in gt_tail if x != gt_tail_org]
    # remove the correct tails from the predictions
    for key in gt_tail_to_filter:
        del results[key]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    sorted_keys = [x[0] for x in sorted_x]
    index_tail_head_filter = sorted_keys.index(gt_tail_org)

    return (index_correct_tail_raw + 1), (index_tail_head_filter + 1), top_10_predictions, top_1_predictions

def predict_tail(head_embedding_structure, head_embedding_description, relation_embedding,
                 tails_embedding_structure, tails_embedding_description):


    r_input = graph.get_tensor_by_name("input/r_input:0")
    h_pos_structure_input = graph.get_tensor_by_name("input/h_pos_structure_input:0")
    t_pos_structure_input = graph.get_tensor_by_name("input/t_pos_structure_input:0")

    h_pos_description_input = graph.get_tensor_by_name("input/h_pos_image_input:0")
    t_pos_description_input = graph.get_tensor_by_name("input/t_pos_image_input:0")



    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    energy_fusion_pos = graph.get_tensor_by_name("cosine/energy_fusion_pos:0")


    predictions = energy_fusion_pos.eval(feed_dict={r_input: relation_embedding,
                                                    h_pos_structure_input: np.asarray(head_embedding_structure),
                                                    t_pos_structure_input: np.asarray(tails_embedding_structure),
                                                    h_pos_description_input: np.asarray(head_embedding_description),
                                                    t_pos_description_input: np.asarray(tails_embedding_description),
                                                    keep_prob: 1.0})
    return predictions



def predict_head(tail_embeddings_list_structure, tail_embeddings_list_description,
                 full_relation_embeddings, heads_embeddings_list_structure, heads_embeddings_list_description):


    r_input = graph.get_tensor_by_name("input/r_input:0")
    h_pos_structure_input = graph.get_tensor_by_name("input/h_pos_structure_input:0")
    t_pos_structure_input = graph.get_tensor_by_name("input/t_pos_structure_input:0")

    h_pos_description_input = graph.get_tensor_by_name("input/h_pos_image_input:0")
    t_pos_description_input = graph.get_tensor_by_name("input/t_pos_image_input:0")



    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    score_fusion_pos = graph.get_tensor_by_name("cosine/score_fusion_pos:0")

    predictions = score_fusion_pos.eval(feed_dict={r_input: full_relation_embeddings,
                                                   h_pos_structure_input: np.asarray(heads_embeddings_list_structure),
                                                   t_pos_structure_input: np.asarray(tail_embeddings_list_structure),
                                                   h_pos_description_input: np.asarray(heads_embeddings_list_description),
                                                   t_pos_description_input: np.asarray(tail_embeddings_list_description),
                                                   keep_prob: 1.0})




    return predictions



def predict_best_head(test_triple, full_triple_list, full_entity_list, entity_embeddings_structure, entity_embeddings_description, full_relation_embeddings):

    #triple: head, tail, relation
    results = {}
    gt_tail = test_triple[1] #tail
    gt_tail_embeddings_structure = entity_embeddings_structure[gt_tail] #tail embeddings
    gt_tail_embeddings_description = entity_embeddings_description[gt_tail]

    gt_rel = test_triple[2]
    gt_relation_embeddings = full_relation_embeddings[gt_rel]

    gt_head_org = test_triple[0]
    gt_head = u.get_correct_heads(gt_tail, gt_rel, full_triple_list)



    total_batches = len(full_entity_list) // batch_size + 1

    predictions = []
    for batch_i in range(total_batches):
        start = batch_size * (batch_i)
        end = batch_size * (batch_i + 1)
        heads_embeddings_list_structure = []
        heads_embeddings_list_description = []

        tail_embeddings_list_structure = np.tile(gt_tail_embeddings_structure, (len(full_entity_list[start:end]), 1))
        tail_embeddings_list_description = np.tile(gt_tail_embeddings_description, (len(full_entity_list[start:end]), 1))
        full_relation_embeddings = np.tile(gt_relation_embeddings, (len(full_entity_list[start:end]), 1))


        for i in range(len(full_entity_list[start:end])):


            heads_embeddings_list_structure.append(entity_embeddings_structure[full_entity_list[start+i]])
            heads_embeddings_list_description.append(entity_embeddings_description[full_entity_list[start+i]])

        sub_predictions = predict_head(tail_embeddings_list_structure, tail_embeddings_list_description,
                                       full_relation_embeddings, heads_embeddings_list_structure,
                                       heads_embeddings_list_description)

        for p in sub_predictions:
            predictions.append(p)


    predictions = [predictions]

    for i in range(0, len(predictions[0])):
        if full_entity_list[i] == gt_tail and gt_tail not in gt_head:

        #    #results[full_entity_list[i]] = 0
            pass
        else:
            results[full_entity_list[i]] = predictions[0][i]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    top_10_predictions = [x[0] for x in sorted_x[:10]]
    top_1_predictions = [x[0] for x in sorted_x[:1]]
    sorted_keys = [x[0] for x in sorted_x]
    index_correct_head_raw = sorted_keys.index(gt_head_org)

    gt_tail_to_filter = [x for x in gt_head if x != gt_head_org]
    # remove the correct tails from the predictions
    for key in gt_tail_to_filter:
        del results[key]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    sorted_keys = [x[0] for x in sorted_x]
    index_head_filter = sorted_keys.index(gt_head_org)

    return (index_correct_head_raw + 1), (index_head_filter + 1), top_10_predictions, top_1_predictions

############ Testing Part #######################
relation_embeddings = u.load_binary_file(param.relation_structural_embeddings_file, 3)
entity_embeddings_structure = u.load_binary_file(param.entity_structural_embeddings_file, 3)
entity_embeddings_imgae = u.load_binary_file(param.entity_img_embedding_file, 3)
# entity_embeddings_description = u.load_binary_file(param.entity_description_embeddings_file, 3)
# entity_embeddings_img = u.load_binary_file(param.entity_multimodal_embeddings_file, 3)

entity_list = u.load_entity_list(param.all_triples_file, entity_embeddings_structure, entity_embeddings_imgae)

print("#Entities", len(entity_list))
all_triples = u.load_triples(param.all_triples_file, entity_list)
all_test_triples = u.load_triples(param.test_triples_file, entity_list)
#all_test_triples = all_test_triples[:1000]
print("#Test triples", len(all_test_triples))  # Triple: head, tail, relation


tail_ma_raw = 0
tail_ma_filter = 0
tail_hits10_raw = 0
tail_hits10_filter = 0
tail_hits1_raw = 0
tail_hits1_filter = 0
head_ma_raw = 0
head_ma_filter = 0
head_hits10_raw = 0
head_hits10_filter = 0
head_hits1_raw = 0
head_hits1_filter = 0

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
        #print("Model restored from file: %s" % param.current_model_meta_file)
        avg_rank_raw = 0.0
        avg_rank_filter = 0.0
        hits_at_10_raw = 0.0
        hits_at_10_filter = 0.0
        hits_at_1_raw = 0.0
        hits_at_1_filter = 0.0
        lines = []

        #new_saver = tf.train.import_meta_graph(param.model_meta_file)
        # new_saver.restore(sess, param.model_weights_best_file)

        saver = tf.train.import_meta_graph(param.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(param.checkpoint_best_valid_dir))

        graph = tf.get_default_graph()
        #Warning only for relation classification
        #entity_list = u.load_relation_list(param.all_triples_file, entity_embeddings)
        counter = 1
        for triple in all_test_triples:
            rank_raw, rank_filter, top_10, top_1 = predict_best_tail(triple, all_triples, entity_list,
                                                              entity_embeddings_structure,
                                                              entity_embeddings_imgae,
                                                              relation_embeddings)

            line = triple[0] + "\t" + triple[2] + "\t" + triple[1] + "\t" + str(top_10) + "\t" + str(top_1) + "\t" + \
                   str(rank_raw) + "\t" + str(rank_filter) + "\n"

            #print(line)
            lines.append(line)
            print(str(counter) + "/" + str(len(all_test_triples)) + " " + str(rank_raw) + " " + str(rank_filter))
            counter += 1
            avg_rank_raw += rank_raw
            avg_rank_filter += rank_filter
            if rank_raw <= 10:
                hits_at_10_raw += 1
            if rank_filter <= 10:
                hits_at_10_filter += 1
            if rank_raw <= 1:
                hits_at_1_raw += 1
            if rank_filter <= 1:
                hits_at_1_filter += 1

        avg_rank_raw /= len(all_test_triples)
        avg_rank_filter /= len(all_test_triples)
        hits_at_10_raw /= len(all_test_triples)
        hits_at_10_filter /= len(all_test_triples)
        hits_at_1_raw /= len(all_test_triples)
        hits_at_1_filter /= len(all_test_triples)

        print("MAR Raw", avg_rank_raw, "MAR Filter", avg_rank_filter)
        print("Hits@10 Raw", hits_at_10_raw, "Hits@10 Filter", hits_at_10_filter)
        print("Hits@1 Raw", hits_at_1_raw, "Hits@1 Filter", hits_at_1_filter)

        # Write to a file
        #results_file = param.result_file
        results_file = param.result_file.replace(".txt", "tail_prediction.txt")

        if os.path.isfile(results_file):
            results_file = results_file.replace(".txt", "_1.txt")

        print("write the results into", results_file)
        with open(results_file, "w") as f:
            f.write("MAR Raw" + "\t" + str(avg_rank_raw) + "\t" + "MAR Filter" + "\t" + str(avg_rank_filter) + "\n")
            f.write("Hits@10 Raw" + "\t" + str(hits_at_10_raw) + "\t" + "Hits@10 Filter" + "\t" + str(
                hits_at_10_filter) + "\n" + "\n")
            f.write("Hits@1 Raw" + "\t" + str(hits_at_1_raw) + "\t" + "Hits@1 Filter" + "\t" + str(hits_at_1_filter) +
                    "\n" + "\n")
            f.write("Head \t Relation \t Gold Tail \t Top Predicted Tails \t Raw Rank \t Filtered Rank\n")
            for l in lines:
                f.write(str(l))

        tail_ma_raw = avg_rank_raw
        tail_ma_filter = avg_rank_filter
        tail_hits10_raw = hits_at_10_raw
        tail_hits10_filter = hits_at_10_filter
        tail_hits1_raw = hits_at_1_raw
        tail_hits1_filter = hits_at_1_filter


        avg_rank_raw = 0.0
        avg_rank_filter = 0.0
        hits_at_10_raw = 0.0
        hits_at_10_filter = 0.0
        hits_at_1_raw = 0.0
        hits_at_1_filter = 0.0
        lines = []

        counter = 1
        for triple in all_test_triples:
            rank_raw, rank_filter, top_10, top_1 = predict_best_head(triple, all_triples, entity_list,
                                                              entity_embeddings_structure,
                                                              entity_embeddings_imgae,
                                                              relation_embeddings)

            line = triple[1] + "\t" + triple[2] + "\t" + triple[0] + "\t" + str(top_10) + "\t" + str(top_1) + "\t" + str(rank_raw) + "\t" + str(
                    rank_filter) + "\n"

            #print(line)
            lines.append(line)
            print(str(counter) + "/" + str(len(all_test_triples)) + " " + str(rank_raw) + " " + str(rank_filter))
            counter += 1
            avg_rank_raw += rank_raw
            avg_rank_filter += rank_filter
            if rank_raw <= 10:
                hits_at_10_raw += 1
            if rank_filter <= 10:
                hits_at_10_filter += 1
            if rank_raw <= 1:
                hits_at_1_raw += 1
            if rank_filter <= 1:
                hits_at_1_filter += 1

        avg_rank_raw /= len(all_test_triples)
        avg_rank_filter /= len(all_test_triples)
        hits_at_10_raw /= len(all_test_triples)
        hits_at_10_filter /= len(all_test_triples)
        hits_at_1_raw /= len(all_test_triples)
        hits_at_1_filter /= len(all_test_triples)

        print("MAR Raw", avg_rank_raw, "MAR Filter", avg_rank_filter)
        print("Hits@10 Raw", hits_at_10_raw, "Hits@10 Filter", hits_at_10_filter)
        print("hits@1 Raw", hits_at_1_raw, "Hits@1 Filter", hits_at_1_filter)

        # Write to a file
        results_file = param.result_file.replace(".txt", "head_prediction.txt")
        if os.path.isfile(results_file):
            results_file = results_file.replace(".txt", "_1.txt")

        print("write the results into", results_file)
        with open(results_file, "w") as f:
            f.write("MAR Raw" + "\t" + str(avg_rank_raw) + "\t" + "MAR Filter" + "\t" + str(avg_rank_filter) + "\n")
            f.write("Hits@10 Raw" + "\t" + str(hits_at_10_raw) + "\t" + "Hits@10 Filter" + "\t" + str(
                hits_at_10_filter) + "\n" + "\n")
            f.write("Hits@1 Raw" + "\t" + str(hits_at_1_raw) + "\t" + "Hits@1 Filter" + "\t" + str(hits_at_1_filter)
                    + "\n" + "\n")
            f.write("Tail \t Relation \t Gold Head \t Top Predicted Heads \t Raw Rank \t Filtered Rank\n")
            for l in lines:
                f.write(str(l))

        head_ma_raw = avg_rank_raw
        head_ma_filter = avg_rank_filter
        head_hits10_raw = hits_at_10_raw
        head_hits10_filter = hits_at_10_filter
        head_hits1_raw = hits_at_1_raw
        head_hits1_filter = hits_at_1_filter

print("+++++++++++++++ Evaluation Summary ++++++++++++++++")
print("MA Raw Tail \t MA Filter Tail \t Hits@10 Raw Tail \t Hits@10 Filter Tail \t Hits@1 Raw Tail \t Hits@1 Filter Tail")
print(str(tail_ma_raw) + "\t" + str(tail_ma_filter) + "\t" + str(tail_hits10_raw) + "\t" + str(tail_hits10_filter)
      + "\t" + str(tail_hits1_raw) + "\t" + str(tail_hits1_filter))


print("MA Raw Head \t MA Filter Head \t Hits@10 Raw Head \t Hits@10 Filter Head \t Hits@1 Raw Head \t Hits@1 Filter Head")
print(str(head_ma_raw) + "\t" + str(head_ma_filter) + "\t" + str(head_hits10_raw) + "\t" + str(head_hits10_filter)
      + "\t" + str(head_hits1_raw) + "\t" + str(head_hits1_filter))


print("MA Raw AVG \t MA Filter AVG \t Hits@10 Raw AVG \t Hits@10 Filter AVG \t Hits@1 Raw AVG \t Hits@1 Filter AVG")
avg_ma_raw = (head_ma_raw + tail_ma_raw) / 2
avg_ma_filter = (head_ma_filter + tail_ma_filter) / 2
avg_hits10_raw = (head_hits10_raw + tail_hits10_raw) / 2
avg_hits10_filter = (head_hits10_filter + tail_hits10_filter) / 2
avg_hits1_raw = (head_hits1_raw + tail_hits1_raw) / 2
avg_hits1_filter = (head_hits1_filter + tail_hits1_filter) / 2

print(str(avg_ma_raw) + "\t" + str(avg_ma_filter) + "\t" + str(avg_hits10_raw) + "\t" + str(avg_hits10_filter) + "\t"
      + str(avg_hits1_raw) + "\t" + str(avg_hits1_filter))
