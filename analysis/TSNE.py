import argparse
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import auc,roc_curve, precision_recall_curve, roc_auc_score,confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import Counter
import glob
import os
import numpy as np
from PIL import Image
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets #手写数据集要用到
from sklearn.manifold import TSNE

with open("../../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
class_to_idx = config["PACS_class_to_idx"]
domain_to_idx = config["PACS_domain_to_idx"]

args = argparse.ArgumentParser()
args.add_argument("--selected_epochs", type=int, default=0)
args = args.parse_args()
selected_epochs = args.selected_epochs

print(selected_epochs)
train_class_result_2D_list = []
train_domain_result_2D_list = []

# 训练集

intermediate_train = np.load(f'../results/intermediate_results/epoch={selected_epochs}.npz', allow_pickle=True)

perplexity=300
early_exaggeration=12
tsne_2D = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, verbose=1, early_exaggeration=early_exaggeration) #调用TSNE
class_result_2D = tsne_2D.fit_transform(np.concatenate([intermediate_train["class_feature"], intermediate_train["center"].reshape(1, -1)]))
train_class_result_2D_list.append(class_result_2D)

colors = ['#999999', '#e41a1c', '#ff7f00']
custom_cmap = ListedColormap(colors)

x_min, x_max = np.min(class_result_2D, 0), np.max(class_result_2D, 0)
data = (class_result_2D - x_min) / (x_max - x_min)
fig, ax  = plt.subplots(1,1,figsize=(6,6))

size = np.zeros_like(intermediate_train["target_list"], dtype=int) + 20
size = np.append(size, 100)
scatter = ax.scatter(data[:, 0], data[:, 1], c=np.append(intermediate_train["target_list"], 2), cmap=custom_cmap, s=size)
ax.legend(*(scatter.legend_elements()[0],["normal", "anomaly", "center"]))
plt.savefig(f"../results/intermediate_results/epoch={selected_epochs},class_embedding.pdf", format="pdf")

print("train class embedding finish")

perplexity=200
early_exaggeration=12
tsne_2D = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, verbose=1, early_exaggeration=early_exaggeration) #调用TSNE
domain_result_2D = tsne_2D.fit_transform(np.concatenate([intermediate_train["texture_feature"], intermediate_train["domain_prototype"]]))
train_domain_result_2D_list.append(domain_result_2D)

colors = ['#999999', '#e41a1c', '#ff7f00']
custom_cmap = ListedColormap(colors)

x_min, x_max = np.min(domain_result_2D, 0), np.max(domain_result_2D, 0)
data = (domain_result_2D - x_min) / (x_max - x_min)
fig, ax  = plt.subplots(1,1,figsize=(6,6))

size = np.zeros_like(intermediate_train["target_list"], dtype=int) + 20
size = np.concatenate([size, np.array([100, 100, 100])])
scatter = ax.scatter(data[:, 0], data[:, 1], c=np.concatenate([intermediate_train["domain_label_list"], np.array([0,1,2])]), cmap=plt.cm.Set1, s=size)
ax.legend(*(scatter.legend_elements()[0], ['art_painting', 'cartoon', 'photo', 'sketch']))
plt.savefig(f"../results/intermediate_results/epoch={selected_epochs},domain_embedding.pdf", format="pdf")
print("train domain embedding finish")

# 验证集

intermediate_val = np.load(f'../results/intermediate_results/epoch={selected_epochs},val.npz', allow_pickle=True)
intermediate_val

perplexity=150
early_exaggeration=12
tsne_2D = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, verbose=1, early_exaggeration=early_exaggeration) #调用TSNE
val_class_result_2D = tsne_2D.fit_transform(np.concatenate([intermediate_val["class_feature_list"], intermediate_train["center"].reshape(1, -1)]))

colors = ['#999999', '#e41a1c', '#ff7f00']
custom_cmap = ListedColormap(colors)

x_min, x_max = np.min(val_class_result_2D, 0), np.max(val_class_result_2D, 0)
data = (val_class_result_2D - x_min) / (x_max - x_min)
fig, ax  = plt.subplots(1,1,figsize=(6,6))

size = np.zeros_like(intermediate_val["target_list"], dtype=int) + 20
size = np.append(size, 100)
scatter = ax.scatter(data[:, 0], data[:, 1], c=np.append(intermediate_val["target_list"], 2), cmap=custom_cmap, s=size)
ax.legend(*(scatter.legend_elements()[0],["normal", "anomaly", "center"]))
plt.savefig(f"../results/intermediate_results/epoch={selected_epochs},val_class_embedding.pdf", format="pdf")

perplexity=60
early_exaggeration=12
tsne_2D = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, verbose=1, early_exaggeration=early_exaggeration) #调用TSNE
domain_result_2D = tsne_2D.fit_transform(np.concatenate([intermediate_val["texture_feature_list"], intermediate_train["domain_prototype"]]))

colors = ['#999999', '#e41a1c', '#ff7f00']
custom_cmap = ListedColormap(colors)

x_min, x_max = np.min(domain_result_2D, 0), np.max(domain_result_2D, 0)
data = (domain_result_2D - x_min) / (x_max - x_min)
fig, ax  = plt.subplots(1,1,figsize=(6,6))

size = np.zeros_like(intermediate_val["target_list"], dtype=int) + 20
size = np.concatenate([size, np.array([100, 100, 100])])
scatter = ax.scatter(data[:, 0], data[:, 1], c=np.concatenate([intermediate_val["domain_label_list"], np.array([0,1,2])]), cmap=plt.cm.Set1, s=size)
ax.legend(*(scatter.legend_elements()[0], ['art_painting', 'cartoon', 'photo']))
plt.savefig(f"../results/intermediate_results/epoch={selected_epochs},val_domain_embedding.pdf", format="pdf")


# 测试集


intermediate_test = dict()
for item in domain_to_idx.keys():
    intermediate_test[item] = np.load(f'../results/intermediate_results/epoch={selected_epochs},{item}.npz', allow_pickle=True)
intermediate_test

fig, ax  = plt.subplots(1,4,figsize=(16,4))

test_class_result_2D_dict = dict()

for idx, selected_domain in enumerate(domain_to_idx.keys()):
    print(Counter(intermediate_test[selected_domain]["target_list"]))
    perplexity=Counter(intermediate_test[selected_domain]["target_list"])[1]
    early_exaggeration=12
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, verbose=1, early_exaggeration=early_exaggeration) #调用TSNE
    class_result_2D = tsne_2D.fit_transform(np.concatenate([intermediate_test[selected_domain]["class_feature_list"], intermediate_train["center"].reshape(1, -1)]))
    test_class_result_2D_dict[selected_domain] = class_result_2D
    
    colors = ['#999999', '#e41a1c', '#ff7f00']
    custom_cmap = ListedColormap(colors)

    x_min, x_max = np.min(class_result_2D, 0), np.max(class_result_2D, 0)
    data = (class_result_2D - x_min) / (x_max - x_min)

    size = np.zeros_like(intermediate_test[selected_domain]["target_list"], dtype=int) + 20
    size = np.append(size, 100)
    scatter = ax[idx].scatter(data[:, 0], data[:, 1], c=np.append(intermediate_test[selected_domain]["target_list"], 2), cmap=custom_cmap, s=size, label = "")
    ax[idx].legend(*(scatter.legend_elements()[0],["normal", "anomaly", "center"]))
    ax[idx].set_title(selected_domain)

plt.savefig(f"../results/intermediate_results/epoch={selected_epochs},test_class_embedding.pdf", format="pdf")
print("test class embedding finish")

intermediate_test_domain_embedding = []
intermediate_test_domain_labels = []
for item in domain_to_idx.keys():
    intermediate_test_domain_embedding.append(intermediate_test[item]["texture_feature_list"])
    intermediate_test_domain_labels.append(intermediate_test[item]["domain_label_list"])
intermediate_test_domain_embedding = np.concatenate(intermediate_test_domain_embedding)
intermediate_test_domain_labels = np.concatenate(intermediate_test_domain_labels)

perplexity=400
early_exaggeration=12
tsne_2D = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity, verbose=1, early_exaggeration=early_exaggeration) #调用TSNE
test_domain_result_2D = tsne_2D.fit_transform(np.concatenate([intermediate_test_domain_embedding, intermediate_train["domain_prototype"]]))

colors = ['#999999', '#e41a1c', '#ff7f00']
custom_cmap = ListedColormap(colors)

x_min, x_max = np.min(test_domain_result_2D, 0), np.max(test_domain_result_2D, 0)
data = (test_domain_result_2D - x_min) / (x_max - x_min)
fig, ax  = plt.subplots(1,1,figsize=(6,6))

size = np.zeros_like(intermediate_test_domain_labels, dtype=int) + 20
size = np.concatenate([size, np.array([100, 100, 100])])
edgecolors = ["none"] * size.shape[0]
edgecolors[-3:] = ["black"] * 3
scatter = ax.scatter(data[:, 0], data[:, 1], c=np.concatenate([intermediate_test_domain_labels, np.array([0,1,2])]), cmap=plt.cm.Set1, s=size, edgecolors=edgecolors)
ax.legend(*(scatter.legend_elements()[0], ['art_painting', 'cartoon', 'photo', 'sketch']))
plt.savefig(f"../results/intermediate_results/epoch={selected_epochs},test_domain_embedding.pdf", format="pdf")

print("test domain embedding finish")

np.savez(f'../results/intermediate_results/t-nse,epochs={selected_epochs}.npz',
         train_class_result_2D_list=np.array(train_class_result_2D_list),
         train_domain_result_2D_list=np.array(train_domain_result_2D_list),
         test_class_result_2D_dict=np.array(test_class_result_2D_dict),
         test_domain_result_2D = np.array(test_domain_result_2D))