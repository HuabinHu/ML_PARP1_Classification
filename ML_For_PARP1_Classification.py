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
from tqdm.notebook import tqdm
# Machine Learning & Evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    matthews_corrcoef, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve, 
    balanced_accuracy_score, 
    f1_score
)

# Cheminformatics (RDKit)
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
# ==========================================
# 1. Configuration and Helper Functions
# ==========================================
# Initialize Morgan fingerprint generator (Radius=2, Size=1024)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def fp_as_array(mol):
    """Convert RDKit molecule object to Morgan fingerprint as a numpy array."""
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def label_results(truth_list, pred_list):
    """Classify prediction results into TN, FN, FP, TP based on truth and predicted labels."""
    label_list = [["TN", "FN"], ["FP", "TP"]]
    res = [] 
    for truth, pred in zip(pred_list, truth_list):
        res.append(label_list[truth][pred])
    return res

def safe_ratio(x, y):
    """Safely calculate ratio to avoid division by zero, rounding to 2 decimal places."""
    if (x + y) == 0:
        return 0.0
    return round(x / (x + y), 2)
# ==========================================
# 2. Data Loading and Molecular Fingerprint Calculation
# ==========================================
# Load dataset
df = pd.read_csv("C:/HHB_project_ICR/PARP1_Perspective_Screening/ML_PARP1_projecrt_SPECS_ChemDiv/Manuscript_version/CCB/CCB_version/Submitted_version/Git_hub/PARP1_Dataset.csv", sep="\t")
smiles_col = "SMILE"  # Column name containing SMILES

# Print class distribution
print("Class value counts:")
print(df["Class"].value_counts())

# Convert SMILES to RDKit molecule objects and compute fingerprints
df['Mol'] = [Chem.MolFromSmiles(x) for x in df[smiles_col]]
df['fp'] = [fp_as_array(x) for x in df['Mol']]
# ==========================================
# 3. Cross-Validation and Hyperparameter Grid Search Configuration
# ==========================================
# 5-fold stratified cross-validation
data_splitter_intern = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)

# Hyperparameter grid for Grid Search
model_param_dict = {
    'param_grid': {
        'n_estimators': [100, 200, 300, 400], 
        'bootstrap': [False],
        'max_features': ["sqrt", "log2", 0.7], 
        'max_depth': [7, 10, 12, None], 
        'class_weight': ['balanced'],
        'min_samples_leaf': [1, 3, 5, 10]
    },
    'cv': data_splitter_intern,
    'scoring': "balanced_accuracy",
    'refit': "balanced_accuracy",
    'n_jobs': -1
}
# ==========================================
# 4. Model Training Loop
# ==========================================
stat_list = []
detail_list = []
pred_list = []
prob_list = {}
best_params_list = [] 

for cycle in tqdm(range(0, 10)):  # Run model evaluation with specified random state(s)
    # Split dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.3, random_state=cycle)
    train_x, test_x = list(train.fp), list(test.fp)
    train_y, test_y = train.Class, test.Class
    
    # Random Forest classifier with nested Grid Search
    rf = GridSearchCV(
        estimator=RandomForestClassifier(random_state=cycle), 
        **model_param_dict,
        verbose=10  # Set to 10 to output detailed search logs
    )
    
    rf.fit(train_x, train_y)
    best_params_list.append(rf.best_params_) 
    
    # Predict classes and get probabilities
    pred = rf.predict(test_x) 
    pred_list.append([pred, test_y])
    
    result_list = label_results(test_y, pred)
    prob = rf.predict_proba(test_x)
    
    # Record predicted probabilities for Class 1 (positive)
    prob_list[cycle] = [mol_prob[1] for mol_prob in prob]
    
    # Store detailed prediction details
    for smiles, pred_val, true_val, result in zip(test[smiles_col], pred, test_y, result_list):
        detail_list.append([cycle, smiles, pred_val, true_val, result])
        
    # Calculate evaluation metrics for the current run
    stat_list.append({
        'mcc': matthews_corrcoef(test_y, pred),  # Note: standardized as (y_true, y_pred)
        'auc': roc_auc_score(test_y, prob[:, 1]), 
        'bac': balanced_accuracy_score(test_y, pred),
        'f1_score': f1_score(test_y, pred)
    })
# ==========================================
# 5. Evaluation Metrics Summarization and Display
# ==========================================
row_list = []
for p, t in pred_list:
    # Reshape confusion matrix values (using predictions as rows, truth as columns to match original custom logic)
    row_list.append(confusion_matrix(p, t).flatten())

# Construct confusion matrix DataFrame
confusion_df = pd.DataFrame(row_list, columns=["tn", "fn", "fp", "tp"])

# Populate performance metrics
confusion_df["MCC"] = [x['mcc'] for x in stat_list]
confusion_df["AUC"] = [x['auc'] for x in stat_list]
confusion_df["F1 score"] = [x['f1_score'] for x in stat_list]
confusion_df["Balanced accuracy"] = [x['bac'] for x in stat_list]

# Calculate Sensitivity (Recall) and Specificity (Selectivity)
confusion_df["Recall_Sensitivity"] = [
    safe_ratio(confusion_df.loc[i, "tp"], confusion_df.loc[i, "fn"]) 
    for i in range(confusion_df.shape[0])
]
confusion_df["Specificity_Selectivity"] = [
    safe_ratio(confusion_df.loc[i, "tn"], confusion_df.loc[i, "fp"]) 
    for i in range(confusion_df.shape[0])
]

# Display the final performance table
confusion_df
