import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Read example files
samples = np.load('samples_2.npy')
labels = np.load('labels_2.npy')

# Setting uninitialized models
scaler = StandardScaler()
f_selection = SelectKBest()
pca = PCA()
lda = LDA()

# Setting scaler parameters
scaler.set_params(
    with_mean=True,
    with_std=True,
)


# Splitting data to training/test sets using a 70/30 split
samples_train, samples_test, labels_train, labels_test = train_test_split(
    samples,
    labels,
    test_size=0.3,
    train_size=0.7,
    random_state=42,
)

# Fitting scaler and scaling features
scaler.fit(samples_train)
samples_train = scaler.transform(samples_train)



# Running feature selection and PCA for the optimal number features k=20
# Users can use different values for k starting at 3 to see how it affects the model's prediction accuracy
f_selection.set_params(
    score_func=f_classif,
    k=20,                 
)


selec_features = f_selection.fit_transform(samples_train, labels_train)
selec_feat_indices = f_selection.get_support(indices=True) # Getting the indices for the selected features

# Setting up pca parameters. Caution: Number of components cannot exceed number of samples!
pca.set_params(
    n_components=selec_features.shape[1]
)
    
pca.fit(selec_features) # Fitting PCA model
scores = pca.transform(selec_features) # Calculating PCA scores


# Training the LDA model for classification using PC1 and PC2 scores from PCA
lda.fit(scores[:, [0,1]], labels_train)


# Scaling and transforming test sets
samples_test = scaler.transform(samples_test)
samples_test = samples_test[:, selec_feat_indices]
scores_test = pca.transform(samples_test)

# Calculating mean accuracy of LDA predictions as (# of correct predictions / Total # of predictions)
mean_accuracy = lda.score(scores_test[:, [0,1]], labels_test)

print("Model accuracy is {}%".format(round(mean_accuracy * 100), 1))

