from MKGC.Add_des_str.triple_classification import tc_dataset_vreator as tc_dataset_creator
from MKGC.Add_des_str.triple_classification import triple_scorer as triple_scorer
from MKGC.Add_des_str.triple_classification import tc_evlauator as tc_evlauator
from MKGC.Add_des_str.triple_classification import parmeters as p


f = open(p.triple_classification_results_file, "w")
for i in range(10):

    tc_dataset_creator.main()
    triple_scorer.main()
    test_avg_acc, avg_acc = tc_evlauator.main()
    f.write(str(test_avg_acc)+"\n")
    f.flush()

f.close()

