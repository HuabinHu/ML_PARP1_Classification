# ============================================================
# Random Forest PARP Classification Model
#
# Workflow:
# SMILES
#   ↓
# RDKit molecular structure
#   ↓
# Morgan fingerprint (radius=2, 1024 bits)
#   ↓
# Random Forest classifier
#   ↓
# GridSearchCV optimization
#   ↓
# Performance evaluation
# ============================
# 1. Import libraries
# ============================
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
# Evaluation metrics
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
    f1_score
)
# ============================
# 2. Load dataset
# ============================
# Change this path according to your dataset location
DATA_PATH = "../data/PARP1_Dataset.csv"
# Dataset should contain:
# SMILE  : molecular SMILES string
# Class  : binary activity label (0/1)
df = pd.read_csv(DATA_PATH,sep="\t")
print(df.head())
print("\nClass distribution:")
print(df["Class"].value_counts())
# SMILES column name
SMILES_FIELD = "SMILE"
# ============================
# 3. Generate Morgan fingerprint
# ============================
# Morgan fingerprint parameters
# radius=2 corresponds to ECFP4
# fpSize=1024 means each molecule is represented by 1024 bits
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
def fingerprint_to_array(molecule):
    """
    Convert RDKit fingerprint object
    into numpy array.
    """
    fingerprint = morgan_generator.GetFingerprint(molecule)
    array = np.zeros((1024,),dtype=int)
    DataStructs.ConvertToNumpyArray(fingerprint,array)
    return array
# Convert SMILES into RDKit molecule object
df["Mol"] = [Chem.MolFromSmiles(smiles) for smiles in df[SMILES_FIELD]]
# Generate fingerprints
df["Fingerprint"] = [fingerprint_to_array(mol) for mol in df["Mol"]]
# ============================
# 4. Model parameter optimization
# ============================
# Internal cross validation
cv_strategy = StratifiedKFold(n_splits=5,shuffle=True,random_state=56)
# Random Forest hyperparameter space
RF_parameters = {
    "param_grid": {
        "n_estimators":[100,200,300,400],
        "bootstrap":[False],
        "max_features":["sqrt","log2",0.7],
        "max_depth":[7,10,12,None],
        "class_weight":["balanced"],
        "min_samples_leaf":[1,3,5,10]
    },
    "cv":cv_strategy,
    "scoring":"balanced_accuracy",
    "refit":True,
    "n_jobs":-1
}
# ============================
# 5. Evaluation helper function
# ============================
def calculate_metrics(y_true,y_pred,y_probability):
    """
    Calculate model performance metrics.
    """
    return {
        "MCC":matthews_corrcoef(y_true,y_pred),
        "AUC":roc_auc_score(y_true,y_probability),
        "Balanced_accuracy":balanced_accuracy_score(y_true,y_pred),
        "F1_score":f1_score(y_true,y_pred)
    }
# ============================
# 6. Model training loop
# ============================
all_results = []
best_parameters = []
prediction_details = []
# Repeat 10 independent train-test splits
for seed in tqdm(range(10),desc="Random Forest runs"):
    print(
        f"\nRunning seed {seed}"
    )
    # Train/test split
    train, test = train_test_split(df,test_size=0.3,random_state=seed,stratify=df["Class"])
    X_train = list(train["Fingerprint"])
    X_test = list(test["Fingerprint"])
    y_train = train["Class"]
    y_test = test["Class"]
    # ============================
    # Random Forest + GridSearch
    # ============================
    RF_model = GridSearchCV(estimator=RandomForestClassifier(random_state=seed),**RF_parameters,verbose=1)
    # Train model
    RF_model.fit(X_train,y_train)
    # Prediction
    prediction = RF_model.predict(X_test)
    probability = RF_model.predict_proba(X_test)[:,1]
    # Save best parameters
    best_parameters.append(RF_model.best_params_)
    # Metrics
    metrics = calculate_metrics(y_test,prediction,probability)
    metrics["seed"] = seed
    all_results.append(metrics)
    # Save molecule level prediction
    for smiles,pred,true in zip(test[SMILES_FIELD],prediction,y_test):
        prediction_details.append([seed,smiles,pred,true])
# ============================
# 7. Results summary
# ============================
results_df = pd.DataFrame(all_results)
print("\nModel performance:")
print(results_df)
# Confusion matrix
confusion_results = []
for row in prediction_details:
    seed = row[0]
    pred = row[2]
    true = row[3]
    tn,fp,fn,tp = confusion_matrix(
        [true],
        [pred],
        labels=[0,1]
    ).ravel()
    confusion_results.append([seed,tn,fp,fn,tp])
confusion_df = pd.DataFrame(
    confusion_results,
    columns=["Seed","TN","FP","FN","TP"]
)
print("\nConfusion matrix:")
print(confusion_df)
# Save results
results_df.to_csv(
    "../results/RF_performance.csv",
    index=False
)
confusion_df.to_csv("../results/RF_confusion_matrix.csv",index=False)
print("\nFinished!")