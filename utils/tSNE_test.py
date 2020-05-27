import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import offsetbox
from tqdm import tqdm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class tSNE_generator():
    def __init__(self, data_loader, model, shape = (7,32,32)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shape = shape
        self.data_loader = data_loader
        model.fc7 = Identity()
        self.model = model
        self.classes = ['s1', 's2', 'tal', 'dct', 'cd','cd45', 'nestin', 'cd31glom', 'cd31inter']
        self.DISTANCE_THREHOLD = 1e-2
        
    def get_features(self):
        self.features = []
        self.labels = []
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            for i, row in enumerate(output.cpu()):
                self.features.append(row.data.numpy())
                self.labels.append(target.cpu().data.numpy()[i])
                
    def _tSNE(self, num_imgs = 'all'):
        tsne = TSNE(init = 'random', verbose = 1, perplexity = 100)
        print("NOW FITTING")
        if num_imgs != 'all':
            input_features = self.features[num_imgs]
        else:
            input_features = self.features
            
        pca = PCA(n_components = 0.9, svd_solver = 'full') #select number of feature to get 90% of variance
        input_features = pca.fit_transform(input_features)
        print("PCA COMPLETE: using " + str(len(input_features[0])) + " features")
        y = tsne.fit_transform(input_features)
        print("FITTING COMPLETE")
        #tsne.plot_embedding(y, title = "tSNE embedding")
        '''
        fig, ax = plt.figure()
        ax.scatter(y[:,0], y[:,1])
        fig.show()
        '''
        return y

    def plot_embedding(self, X, title=None):
        ml = self.labels
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure()
        ax = plt.subplot(111)
        plt.scatter(X[:,0], X[:,1], c = [plt.cm.Set1(self.labels[i]/9.) for i in range(X.shape[0])], s = 0.75)
        cbar = ax.figure.colorbar(cm.ScalarMappable(cmap='Set1'), ax=ax, ticks = np.linspace(0,1,9))
        cbar.ax.set_yticklabels(self.classes)
        #plt.scatter(X[:,0], X[:,1], cmap = plt.cm.Set1, s = 0.5)
        for i in range(X.shape[0]):
            continue
            '''
            plt.text(X[i, 0], X[i, 1], str(self.labels[i]),
                     color=plt.cm.Set1(self.labels[i] / 9.),
                     fontdict={'weight': 'bold', 'size': 5})
            '''
            #plt.scatter(X[i,0], X[i, 1], c = [plt.cm.Set1(self.labels[i]/9.)], s = 0.5, label = str(self.labels[i]))
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < self.DISTANCE_THREHOLD:
                    # don't show points that are too close
                     continue
                shown_images = np.r_[shown_images, [X[i]]]
                img = self.data_loader.dataset.__getitem__(i)
                img = np.squeeze(img[0])
                #print(img.size())
                img = img[int(self.shape[0]/2)]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(img, cmap=plt.cm.gray_r),
                    X[i], bboxprops = dict(edgecolor=plt.cm.Set1(self.labels[i]/9.)))
                #ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
        print("FINISHED PLOTTING")
        
    def show_examples(self, X):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        #plt.figure()
        img_counter = 0
        pw = 5
        pl = 10
        #fig, axs = plt.subplots(pl, pw, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig, axs = plt.subplots(pl, pw, figsize=(12,12),facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = 0.25, wspace= -0.75)

        axs = axs.ravel()
        shown_images = np.array([[1., 1.]])
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < self.DISTANCE_THREHOLD:
                # don't show points that are too close
                 continue
            shown_images = np.r_[shown_images, [X[i]]]
            img = self.data_loader.dataset.__getitem__(i)
            img = np.squeeze(img[0])
            img = img[int(self.shape[0]/2)]
            if img_counter < pw*pl:
                axs[img_counter].imshow(img, cmap = plt.cm.gray_r)
                axs[img_counter].set_title(self.classes[self.labels[i]], fontsize = 5)
                axs[img_counter].set_xticks([])
                axs[img_counter].set_yticks([])
                img_counter+=1
        #plt.tight_layout()    
        