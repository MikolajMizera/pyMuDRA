from os.path import join
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys
from mordred import Calculator, descriptors
from pymudra.mudra import MUDRAEstimator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def Q2(y, y_pred):
    press = np.sum((y_pred-y)**2)
    tss=np.sum((y-y.mean())**2)
    return 1-(press/tss)

def clean_descriptors(X, idx):
    
    X=X[:,np.all(np.isfinite(X), axis=0)]
    X = X[:, ~(X.std(axis=0)<0.02)]
    
    corrs = np.array([1])
    while np.any(corrs>0.95):
        corrs = np.corrcoef(X.T)**2
        corrs[np.triu_indices_from(corrs)]=0
        idx1, idx2 = np.unravel_index(np.argmax(corrs), corrs.shape)
        X = X[:,~(np.arange(len(X.T))==idx1)]
    
    print('Descriptor set %d dimensions: %d'%(idx, X.shape[1]))
    
    return X

# functions for calculating fingerprints
morgan_fp = lambda m: rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)
maccs_fp = lambda m: MACCSkeys.GenMACCSKeys(m) 

sdf_filename = join('datasets', 'D4_mols_confident.sdf')
mols = [m for m in Chem.SDMolSupplier(sdf_filename, removeHs=False)]

print('\nCalculating fingerprints and descriptors...')
X = [np.array([np.array(d(mol)).astype(float) for mol in tqdm(mols)])
                                                for d in [morgan_fp, maccs_fp]]

print('\nCleaning up variables...')
X = [clean_descriptors(x, i) for i, x in tqdm(list(enumerate(X)))]

y = np.array([mol.GetPropsAsDict()['pki'] for mol in mols])

quantiles = np.quantile(y, np.linspace(0,1,8))
y_classes = np.digitize(y, quantiles[1:], right=True)

# 5-fold cross-validation
y_preds, y_tests = [], []
skf = StratifiedKFold(n_splits=5)
for fold_id, (train_index, test_index) in enumerate(skf.split(X[0], y_classes)):
    
    X_train, y_train = [x[train_index] for x in X], y[train_index]
    X_test, y_test = [x[test_index] for x in X], y[test_index]
    
    mudra_est = MUDRAEstimator('regressor', n_neighbors = 5, metric = 'jaccard')
    mudra_est.fit(X_train, y_train)
    
    y_pred, _, _ = mudra_est.predict(X_test)
    y_preds.append(y_pred)
    y_tests.append(y_test)
   
print('Q2 score: %.3f'%(Q2(np.concatenate(y_tests), np.concatenate(y_preds))))