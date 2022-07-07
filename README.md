# Cryptocurrencies


In this module challenge we had a CSV file of cryptocurrencies that we wanted to cluster and visualize.

In order to be able to fit it with a clustering algorithm (KMeans) and do a scatter plot and scatter-3d, we first needed to do some preprocessing.



## Preprocessing

Preprocessing is an essential step before using any clustering algorithm, which could be feature elimination, encoding, scaling and normalizing...

We first looked at the data and realized that since we do not need to visualize the cryptocurrencies that are not traded, we dropped the ones that had a IsTrading=False, then we also cleaned the dataset from any row that had a NaN value. We, then removed the rows where no coins have been mined!
```
crypto_df.drop(crypto_df[ crypto_df['IsTrading'] == False ].index, inplace = True)
crypto_df = crypto_df.dropna()
crypto_df = crypto_df[crypto_df['TotalCoinsMined'] > 0]
```

For unsupervised learning algorithm to work properly, we need to have numerical data in our dataset, but some columns have text values, so we used get_dummies method to create numerical columns that carry the same info as the text columns which changed the number of columns (features) drastically! the dataframe shape was (532, 4) and it was converted to (532,98)
```
X = pd.get_dummies(crypto_df, columns=['Algorithm','ProofType'])
```

One more step that needs to be done on the dataset to help the clustering algorithms do a better job, is to scale / standardize the values as the values in this dataframe were in very different ranges (some 0 and 1 and others in the millions). for this we used the StandardScaler() model then fit_transformed the data.
```
X_scaled = StandardScaler().fit_transform(X)
```

## Dimentionality Reduction

With so many features in the dataset, algorithms' performance and quality in clustering will be greatly impacted. So we will use PCA method to reduce the 90+ features in the dataframe to 3 principal components. We do this by initialising the pca model and fit it with the data.

```
X_pca = PCA(n_components=3).fit_transform(X_scaled)
```

## Clustering

Now with only 3 principal components we can begin clustering. We will use KMeans algorithm. but how many clusters can we make that best classifies the data? In order to find the best K, we will use the elbow_curve plot, which essentially is try & error with the number of clusters. We try the KMeans with 1 to 10 number of clusters and calculate the inertia, then we will plot it and by looking at the plot and finding at which K does the curve change most to a horizontal line. and that is how we determine how many clusters is optimum. Our elbow curve as below, shows K=4 is the best K in our situation:

![elbow_curve](/images/elbow_curve.png "Elbow Curve")

Now that we know our K, we use KMeans algorithm and fit it and make our predictions that each row of the dataframe belongs to which class.

## Visualisation

We can now visualize the clusters on a 3d-plot, using plotly.express with each PCA as one the 3dimensions, figure as below:

![3d_scatter](/images/3d_scatter.png "3d Scatter")

We can also, show the clusters based on the 2 dimensions of "number of coins mined", "total supplied coins" and their cluster(class) is shown by the color of datapoints. figure as below:

![hvplot_scatter](/images/hvplot_scatter.png "hvPolt Scatter")




