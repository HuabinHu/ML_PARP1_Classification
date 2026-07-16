# Random Forest PARP1 Classification Model

The code employs the Random Forest (RF) algorithm to predict the activity class of small molecules based on their molecular structures represented by SMILES strings.

To generate molecular representations, the SMILES strings are first converted into RDKit molecular objects, followed by calculation of Morgan fingerprints (ECFP4). The Morgan fingerprints are generated using a radius of 2 and a fingerprint size of 1024 bits, resulting in binary molecular descriptors that capture local chemical environments. These fingerprint features are subsequently used as input variables for training the Random Forest classifier.

Prior to model development, the dataset is randomly divided into training and testing sets with a ratio of 7:3. To optimize model performance, hyperparameter tuning is performed using GridSearchCV with a stratified 5-fold cross-validation strategy. The optimized Random Forest models are evaluated through 10 independent train-test splits using several performance metrics, including the Matthews correlation coefficient (MCC), area under the receiver operating characteristic curve (AUC), balanced accuracy, and F1-score.

Here, we provide our curated molecular dataset used for model development. The dataset contains SMILES representations of molecules and their corresponding activity annotations. The column named "Class" indicates the binary classification labels, where "1" represents highly potent molecules and "0" represents weakly potent molecules.
