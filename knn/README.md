## KNN

KNN - K Nearest Neighbours

#### Properties

1. non-parametric algorithm -> knn assume nothing regarding the data distribution. so knn is often the first option when there is no prior knowledge about the data

2. Instance based Learning -> knn doesn't require an explicit training step because there is no model to build.

3. Flexible for distance metric -> we can use any distance metric like Euclidean, manhattan or hamming etc

#### Limitations

1. Large Space Complexity -> as it is instance based learning so the calculated distance metrics between data points should be loaded into ram

2. Test Time -> In knn there is less training time but while test the we hace to loop through all the datapoints for getting the k nearest point. so high testing time

3. Random Data or Imbalance data-> It doesnot perform well if the data is random

4. Outliers -> if k values if low the model is effected by outliers else the model is robust to outliers

5. Scale -> in knn we use distance metric so all the features should be scaled

6. High Dimention Data -> For high dimention data the knn fails because of curse of dimentionality.
