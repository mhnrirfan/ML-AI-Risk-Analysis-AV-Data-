# %%
# %% [markdown]
# # Complete Clustering Notebook (step-by-step with progress prints)
# This notebook:
# - Loads datasets
# - Samples UK data (stratified)
# - Builds an adaptive unsupervised feature pipeline
# - Finds k via elbow + silhouette
# - Searches KMeans hyperparameters (randomized)
# - Trains final KMeans, plots distributions, PCA 2D, t-SNE 3D
# - Builds a Decision Tree to explain cluster assignments
# - Computes silhouette on train & test

# %%
# %% 
# 0. Imports & global settings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import HashingEncoder

import random
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# %%

# %% [markdown]
# ## 1. Load datasets (print shapes)

# %%
print("Loading datasets...")
UK_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/UK-cleaned_data.csv"
US_path = "/Users/mahnooriqbal/COMP702 Project/ML-AI-Risk-Analysis-AV-Data-/Datasets/US_imputed_data.csv"

UK_data = pd.read_csv(UK_path)
US_data = pd.read_csv(US_path)

print("Initial shapes:")
print(" UK:", UK_data.shape)
print(" US:", US_data.shape)

# drop obvious unnecessary columns safely
for df,name in [(UK_data,"UK"), (US_data,"US")]:
    for col in ['latitude','longitude','Report ID','Report Version']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
print("Dropped common meta columns (if present).")
print("Shapes after drops:")
print(" UK:", UK_data.shape)
print(" US:", US_data.shape)

# %%
# ## 2. Stratified sampling of UK dataset (with validation)

# %%
def enhanced_stratified_sample(data, sample_frac=0.2, stratify_col='Highest Injury Severity Alleged'):
    print("\nPerforming stratified sampling...")
    if stratify_col not in data.columns:
        # fallback to first categorical
        possible = data.select_dtypes(include=['object','category']).columns
        if len(possible)==0:
            raise ValueError("No categorical column available for stratification.")
        stratify_col = possible[0]
        print(f"Requested stratify column not found; falling back to {stratify_col}")
    print(f"Stratifying on: {stratify_col}")
    _, sample = train_test_split(
        data,
        test_size=sample_frac,
        stratify=data[stratify_col],
        random_state=RANDOM_STATE
    )
    print("Original distribution (top levels):")
    print(data[stratify_col].value_counts(normalize=True).head(10))
    print("Sample distribution (top levels):")
    print(sample[stratify_col].value_counts(normalize=True).head(10))
    return sample

UK_sampled = enhanced_stratified_sample(UK_data, sample_frac=0.2)
print("UK sampled shape:", UK_sampled.shape)

# We'll keep full US as US_full
US_full = US_data.copy()

# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_feature_pipeline(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# %%
# ## 3. Adaptive unsupervised feature pipeline builder
# The pipeline will only include transformers for columns that actually exist in the dataframe passed.

# %%
def create_adaptive_pipeline(df):
    """
    Build a ColumnTransformer + Pipeline that only uses columns present in df.
    Returns an sklearn Pipeline.
    """
    # Candidate column groups (adjust if your dataset has different names)
    time_cols = [c for c in ['Incident Date', 'Incident Time (24:00)'] if c in df.columns]
    high_card = [c for c in ['City', 'Model'] if c in df.columns]
    med_card = [c for c in ['Make', 'State'] if c in df.columns]
    low_card = [c for c in ['Roadway Type', 'Lighting', 'Weather'] if c in df.columns]
    text_cols = [c for c in ['SV Pre-Crash Movement'] if c in df.columns]
    numeric_cols = [c for c in ['Model Year', 'Posted Speed Limit (MPH)'] if c in df.columns]

    transformers = []

    # time transformer (if both present)
    if len(time_cols) == 2:
        time_transformer = FunctionTransformer(lambda df_in: pd.DataFrame({
            'Year': pd.to_datetime(df_in['Incident Date'], errors='coerce').dt.year.fillna(0).astype(int),
            'Month': pd.to_datetime(df_in['Incident Date'], errors='coerce').dt.month.fillna(0).astype(int),
            'Hour': pd.to_datetime(df_in['Incident Time (24:00)'], errors='coerce').dt.hour.fillna(0).astype(int),
            'Hour_sin': np.sin(2*np.pi*(pd.to_datetime(df_in['Incident Time (24:00)'], errors='coerce').dt.hour.fillna(0))/24),
            'Hour_cos': np.cos(2*np.pi*(pd.to_datetime(df_in['Incident Time (24:00)'], errors='coerce').dt.hour.fillna(0))/24)
        }))
        transformers.append(('time', time_transformer, time_cols))

    # hashing encoders for high & med card
    if high_card:
        transformers.append(('hash_high', HashingEncoder(n_components=12), high_card))
    if med_card:
        # using hashing to stay unsupervised and compact
        transformers.append(('hash_med', HashingEncoder(n_components=6), med_card))

    # onehot for low cardinality
    if low_card:
        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False), low_card))

    # text column(s)
    if text_cols:
        # TfidfVectorizer expects 1D input; wrap it so ColumnTransformer passes appropriate series
        transformers.append(('text', TfidfVectorizer(max_features=50), text_cols[0]))

    # numeric scaling
    if numeric_cols:
        transformers.append(('numeric', StandardScaler(), numeric_cols))

    if len(transformers) == 0:
        raise ValueError("No recognized columns present to build a feature pipeline.")

    preprocessor = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),  # small threshold
        ('dim_reduce', TruncatedSVD(n_components=min(30, max(2, len(numeric_cols)+10))))  # reduce to reasonable dim
    ])

    return pipeline

# Quick test of pipeline creation on US and UK sample
print("\nTesting pipeline creation on US and UK sampled frames:")
try:
    p_us = create_adaptive_pipeline(US_full)
    print("US pipeline created.")
except Exception as e:
    print("US pipeline creation error:", e)

try:
    p_uk = create_adaptive_pipeline(UK_sampled)
    print("UK pipeline created.")
except Exception as e:
    print("UK pipeline creation error:", e)


# %%
# ## 4. Helper: elbow + silhouette to select candidate k

# %%
def find_optimal_k(X, k_min=2, k_max=10, random_state=RANDOM_STATE):
    print("\nComputing elbow and silhouette for k in range({}, {})".format(k_min, k_max+1))
    inertias = []
    sils = []
    ks = list(range(k_min, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))
        print(f" k={k:2d}  inertia={km.inertia_:,.2f}  silhouette={sils[-1]:.4f}")
    # plot
    fig, axes = plt.subplots(1,2,figsize=(14,4))
    axes[0].plot(ks, inertias, '-o')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow (Inertia)')
    axes[1].plot(ks, sils, '-o')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette vs k')
    plt.show()
    best_k = ks[int(np.argmax(sils))]
    print(f"Selected k (by max silhouette) = {best_k}")
    return best_k, ks, inertias, sils



# %%
# ## 5. Helper: randomized hyperparameter search for KMeans (small/light-weight)

# %%
def km_random_search(X, k, n_trials=12, random_state=RANDOM_STATE):
    """
    Random search over some KMeans hyperparameters while keeping n_clusters=k.
    Returns best params and best model (fitted).
    """
    print(f"\nRunning randomized hyperparam search for KMeans with k={k} (trials={n_trials})")
    inits = ['k-means++', 'random']
    n_init_choices = [5, 10, 20, 30]
    max_iter_choices = [100, 200, 500]
    best_score = -999
    best_cfg = None
    best_model = None

    for i in range(n_trials):
        init = random.choice(inits)
        n_init = random.choice(n_init_choices)
        max_iter = random.choice(max_iter_choices)
        km = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f" trial {i+1:02d}: init={init}, n_init={n_init}, max_iter={max_iter} -> silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_cfg = dict(init=init, n_init=n_init, max_iter=max_iter)
            best_model = km
    print("Best config:", best_cfg, "best silhouette:", best_score)
    return best_cfg, best_model, best_score

# %%
# ## 6. Full analysis function for one dataframe (includes train/test silhouette evaluation)

# %%
def full_clustering_workflow(df, dataset_name="Dataset", k_min=2, k_max=8):
    print("\n" + "="*60)
    print(f"RUNNING WORKFLOW FOR: {dataset_name}")
    print("="*60)

    # Build pipeline adaptively
    print("Building adaptive pipeline (based on available columns)...")
    pipeline = create_adaptive_pipeline(df)
    print("Pipeline created.")

    # Drop any explicit known label columns if present
    labels_to_drop = ['Highest Injury Severity Alleged']
    df_safe = df.drop(columns=[c for c in labels_to_drop if c in df.columns], errors='ignore').reset_index(drop=True)

    # Fit-transform features
    print("Fitting pipeline & transforming features...")
    X_all = pipeline.fit_transform(df_safe)
    print("Transformed feature shape:", X_all.shape)

    # Split train/test for validation of clustering stability
    X_train, X_test = train_test_split(X_all, test_size=0.3, random_state=RANDOM_STATE)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Find candidate k using elbow/silhouette on full feature set
    best_k, ks, inertias, sils = find_optimal_k(X_train, k_min=k_min, k_max=k_max)

    # Random search for best KMeans hyperparams using selected k (on TRAIN set)
    best_cfg, best_km_model, best_train_sil = km_random_search(X_train, best_k, n_trials=12)

    # Evaluate on train and test
    print("\nEvaluating final model on train and test splits...")
    train_labels = best_km_model.labels_
    test_labels = best_km_model.predict(X_test)
    train_sil = silhouette_score(X_train, train_labels)
    test_sil = silhouette_score(X_test, test_labels)
    print(f"Silhouette - Train: {train_sil:.4f}, Test: {test_sil:.4f}")

    # attach clusters back to a copy of original df (for distribution plots)
    df_out = df_safe.copy()
    # predict clusters for entire set
    full_labels = best_km_model.predict(X_all)
    df_out['Cluster'] = full_labels

    # Plots: distributions for each (original) column grouped by cluster
    print("\nPlotting distributions per column by cluster (this can take a while)...")
    cols_to_plot = [c for c in df.columns if c != 'Cluster']  # original columns
    # limit plotting to reasonable number of columns to avoid massive outputs
    max_plots = 20
    cnt = 0
    for col in cols_to_plot:
        if cnt >= max_plots:
            print(f"...stopping plots after {max_plots} columns (reduce max_plots to see more).")
            break
        plt.figure(figsize=(6,3))
        try:
            if pd.api.types.is_numeric_dtype(df_out[col]):
                sns.boxplot(x='Cluster', y=col, data=df_out)
            else:
                # show value counts top categories as stacked bar
                topn = df_out[col].value_counts().index[:6]
                pivot = (df_out[df_out[col].isin(topn)]
                         .groupby(['Cluster', col]).size().unstack(fill_value=0))
                pivot.plot(kind='bar', stacked=True, legend=False, figsize=(6,3))
                plt.legend(loc='upper right', fontsize='small')
            plt.title(f"{dataset_name} - {col} by Cluster")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Skipping plot for {col} due to error: {e}")
        cnt += 1

    # Decision tree explainability (train a tree to predict cluster from X)
    print("\nTraining decision tree to explain cluster assignments (limited depth)...")
    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    tree.fit(X_train, train_labels)
    plt.figure(figsize=(16,8))
    plot_tree(tree, filled=True, feature_names=[f"f{i}" for i in range(X_train.shape[1])],
              class_names=[f"Cluster {i}" for i in sorted(np.unique(train_labels))], rounded=True)
    plt.title(f"{dataset_name} - Decision Tree explaining clusters (trained on train split)")
    plt.show()

    # PCA 2D plot
    print("\nPlotting PCA 2D (colored by cluster)...")
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_all)
    plt.figure(figsize=(7,5))
    plt.scatter(pca2[:,0], pca2[:,1], c=full_labels, cmap='tab10', alpha=0.7)
    plt.title(f"{dataset_name} - PCA 2D (k={best_k})")
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.show()

    # t-SNE 3D plot
    print("\nComputing t-SNE 3D (this may take some time)...")
    tsne3 = TSNE(n_components=3, random_state=RANDOM_STATE, perplexity=30, n_iter=1000).fit_transform(X_all)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(tsne3[:,0], tsne3[:,1], tsne3[:,2], c=full_labels, cmap='tab10', alpha=0.7)
    plt.title(f"{dataset_name} - t-SNE 3D (k={best_k})")
    plt.show()

    # Final summary prints
    print("\nFinal summary:")
    print(f" Dataset: {dataset_name}")
    print(f" Selected k: {best_k}")
    print(f" Best KMeans config: {best_cfg}")
    print(f" Train Silhouette: {train_sil:.4f}")
    print(f" Test Silhouette: {test_sil:.4f}")
    print(f" Cluster counts:\n{df_out['Cluster'].value_counts().sort_index()}")

    return {
        'pipeline': pipeline,
        'kmeans': best_km_model,
        'df_with_clusters': df_out,
        'train_silhouette': train_sil,
        'test_silhouette': test_sil,
        'best_k': best_k
    }


# %%
# %% [markdown]
# ## 7. Run workflow for US and UK datasets

# %%
# You can tune k_min/k_max as desired. Keep k_max moderate to keep compute time reasonable.
us_results = full_clustering_workflow(US_full, dataset_name="US Data", k_min=2, k_max=8)
uk_results = full_clustering_workflow(UK_sampled, dataset_name="UK Sampled", k_min=2, k_max=8)


# %%
# %% [markdown]
# ## 8. Quick results summary (prints)

# %%
print("\nUS Results summary:")
print(" Selected k:", us_results['best_k'])
print(" Train silhouette:", us_results['train_silhouette'])
print(" Test silhouette:", us_results['test_silhouette'])
print(" Cluster counts:\n", us_results['df_with_clusters']['Cluster'].value_counts().sort_index())

print("\nUK Results summary:")
print(" Selected k:", uk_results['best_k'])
print(" Train silhouette:", uk_results['train_silhouette'])
print(" Test silhouette:", uk_results['test_silhouette'])
print(" Cluster counts:\n", uk_results['df_with_clusters']['Cluster'].value_counts().sort_index())

# %%
print("\nNotebook run complete.")


