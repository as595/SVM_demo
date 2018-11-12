import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
import sklearn.datasets as ds
import ipywidgets as ipw

s1 = ipw.IntSlider(description='No. Samples',value=200,min=2,max=5000,step=2)
s2 = ipw.FloatSlider(description='Cluster S.D.',value = .6, min=.1, max=2,step=.05)
s3 = ipw.FloatSlider(description='C',min=-8.0,max=2)
s4 = ipw.FloatSlider(description=r'$\gamma$',min=-5.0,max=2)
dd1 = ipw.Dropdown(options=['linear','rbf'])


def sim(numSamples, clusterSD, cost, gamma, kernel):
    # we'll make 2 classes with information in every class
    blob_centers = ([-1]*2,[1]*2)

    # now we generate some gaussian blobs around the centers
    blobs,b_labels = ds.make_blobs(n_samples=numSamples, n_features=numSamples,
                                centers=blob_centers,
                                cluster_std=clusterSD,shuffle=True)
    # define a classifier and fit to the blob data
    clf = svm.SVC(kernel=kernel,C=np.exp(cost),gamma=np.exp(gamma))
    clf.fit(blobs,b_labels)

    # plot it to illustrate
    plt.figure(figsize = (9,8))
    plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=60,c='limegreen')

    for cls in np.unique(b_labels):
        blob = blobs[b_labels==cls,:]
        plt.scatter(blob[:,0],blob[:,1],s=20)

    plt.xlabel(r'$X_{0}$')
    plt.ylabel(r'$X_{1}$')

    # create grid to evaluate model
    xx = yy = np.linspace(-3,3, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    plt.contour(XX[:,:], YY[:,:], Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '-.'])
    
    # accuracy score
    score = clf.score(blobs,b_labels)
    # plot the params
    plt.annotate(r'$\gamma=10$^{:03.1f}'.format(gamma),(.88,.9),xycoords='axes fraction')
    plt.annotate(r'$C=10$^{:03.1f}'.format(cost),(.88,.95),xycoords='axes fraction')
    plt.annotate('Acc. = {:03.2f}'.format(score),(.88,.85),xycoords='axes fraction')
    if kernel == 'linear':
        plt.title('Linear SVC for {} training samples'.format(numSamples))
        plt.annotate(r'$\beta_0={:3.2f}$'.format(clf.coef_[0][0]),(.88,.80),xycoords='axes fraction')
        plt.annotate(r'$\beta_1={:3.2f}$'.format(clf.coef_[0][1]),(.88,.75),xycoords='axes fraction')
    
    plt.show()
