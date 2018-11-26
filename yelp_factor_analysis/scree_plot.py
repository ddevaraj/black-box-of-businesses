#!/usr/bin/env python
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

all_features_business = pd.read_csv("/Users/s0v005x/Desktop/MyWork/Courses/DataMining/Project/yelp_dataset_csv/yelp__business.csv")
all_features_business.dropna(inplace=True)
all_features_business = all_features_business[['latitude', 'longitude', 'stars', 'review_count', 'is_open']][0:100000]
#A = np.random.randn(100000, 8)
#print(A)
A = np.asmatrix(all_features_business, dtype='float') #* np.asmatrix(A.T)
print(A.shape)
U, S, V = np.linalg.svd(A)
eigvals = S**2 / np.cumsum(S)[-1]
eigvals2 = S**2 / np.sum(S)
#assert (eigvals == eigvals2).all()

fig = plt.figure()
sing_vals = np.arange(len(eigvals)) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
#I don't like the default legend so I typically make mine like below, e.g.
#with smaller fonts and a bit transparent so I do not cover up data, and make
#it moveable by the viewer in case upper-right is a bad place for it
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.set_draggable(state=True)
plt.show()

def main():
    """Run main."""

    return 0

if __name__ == '__main__':
    main()