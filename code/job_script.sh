
# creata a sparse cbm with sparsity level at 0.5 to 0.7

python cmb_sparse_joint_wnb.py --model_name distilbert-base-uncased --term 'sparsity' --INIT_SPARSITY 0.5 --FINAL_SPARSITY 0.7 > log/goemo/joint/sparsity/1000_5-7_distillbert.txt