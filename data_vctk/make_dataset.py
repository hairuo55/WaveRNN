import os
import pickle
train_path = "/data/wangtao/tacotron2/VCTK_file/train_real.txt"
test_seen_path = "/data/wangtao/tacotron2/VCTK_file/test_seen.txt"
test_unseen_path = "/data/wangtao/tacotron2/VCTK_file/test.txt"
f_train = open(train_path)
f_seen_test = open(test_seen_path)
f_unseen_test = open(test_unseen_path)

dataset_train_ids = []
for line in f_train:
    dataset_train_ids += [os.path.basename(line.split("|")[0])[:-4]]
with open(f'dataset_train_ids.pkl', 'wb+') as f:
    pickle.dump(dataset_train_ids, f)

print("train_ids competed")

dataset_test_seen_ids = []
for line in f_seen_test:
    dataset_test_seen_ids += [os.path.basename(line.split("|")[0])[:-4]]
with open(f'dataset_test_seen_ids.pkl', 'wb+') as f:
    pickle.dump(dataset_test_seen_ids, f)
print("test_seen_ids competed")


dataset_test_unseen_ids = []
for line in f_unseen_test:
    dataset_test_unseen_ids += [os.path.basename(line.split("|")[0])[:-4]]
with open(f'dataset_test_unseen_ids.pkl', 'wb+') as f:
    pickle.dump(dataset_test_unseen_ids, f)
print("test_unseen_ids competed")
