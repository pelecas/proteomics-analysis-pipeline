#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:42:12 2026

@author: Eliel
"""

# =========================================================
# APP_PROTEOMICS.PY
# =========================================================

import streamlit as st
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import gseapy as gp

from mygene import MyGeneInfo

from itertools import combinations

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import (
    IterativeImputer,
    KNNImputer,
    SimpleImputer
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    PowerTransformer
)

from sklearn.decomposition import PCA

from scipy.stats import (
    shapiro,
    f_oneway,
    kruskal
)

from statsmodels.stats.multitest import multipletests




# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Proteomics Pipeline",

    layout="wide"
)

st.title("Proteomics Analysis Pipeline")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Pipeline Settings")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.sidebar.file_uploader(

    "Upload Proteomics CSV",

    type=["csv"]
)

# =====================================================
# DATA SETTINGS
# =====================================================

already_log2 = st.sidebar.checkbox(

    "Data already log2 transformed?",

    value=False
)

# =====================================================
# IMPUTATION SETTINGS
# =====================================================

st.sidebar.subheader("Imputation")

imputation_method = st.sidebar.selectbox(

    "Imputation Method",

    [
        "RF",
        "KNN",
        "LR",
        "Mean"
    ],

    index=0
)

# =====================================================
# NORMALIZATION SETTINGS
# =====================================================

st.sidebar.subheader("Normalization")

normalization_method = st.sidebar.selectbox(

    "Normalization Method",

    [
        "Imputed",
        "StandardScaler",
        "MinMaxScaler",
        "QuantileTransformer",
        "RobustScaler",
        "PowerTrans_YJ",
        "PowerTrans_BoxCox",
        "SQRT"
    ],

    index=5
)

# =====================================================
# SIGNIFICANCE SETTINGS
# =====================================================

st.sidebar.subheader("Statistics")

significance_column = st.sidebar.selectbox(

    "Significance Metric",

    [
        "pvalue",
        "BH_FDR",
        "BY_FDR",
        "Bonferroni"
    ],

    index=1
)

# =====================================================
# THRESHOLDS
# =====================================================

fc_threshold = st.sidebar.slider(

    "Log2FC Threshold",

    0.1,
    5.0,
    0.5
)

pval_threshold = st.sidebar.slider(

    "P-value / FDR Threshold",

    0.0001,
    0.2,
    0.05
)

cv_threshold = st.sidebar.slider(

    "CV Threshold (%)",

    5,
    100,
    30
)

# =====================================================
# VISUALIZATION OPTIONS
# =====================================================

st.sidebar.subheader("Visualization")

show_volcano_labels = st.sidebar.checkbox(

    "Show Volcano Labels",

    value=True
)

show_pca_grid = st.sidebar.checkbox(

    "Show PCA Grid",

    value=True
)

export_svg = st.sidebar.checkbox(

    "Export SVG Figures",

    value=True
)

# =========================================================
# GO SETTINGS
# =========================================================

st.sidebar.header("GO Enrichment")

GO_DATABASE = st.sidebar.selectbox(

    "GO Database",

    [
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "GO_Cellular_Component_2023"
    ]
)

GO_SIGNIFICANCE = st.sidebar.selectbox(

    "GO Significance Column",

    [
        "pvalue",
        "BH_FDR",
        "BY_FDR",
        "Bonferroni"
    ],

    index=0
)

GO_FC_THRESHOLD = st.sidebar.slider(

    "GO FC Threshold",

    0.1,
    5.0,
    0.5
)

GO_PVALUE_THRESHOLD = st.sidebar.slider(

    "GO P-value Threshold",

    0.0001,
    0.2,
    0.05
)

TOP_GO_TERMS = st.sidebar.slider(

    "Top GO Terms",

    5,
    50,
    20
)

# =========================================================
# FUNCTIONS
# =========================================================

@st.cache_data

def impute_data(

    X,

    method="rf",

    random_state=42,

    max_iter=20,

    tol=1e-3
):

    method = method.lower()

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    if method == "rf":

        imputer = IterativeImputer(

            estimator=RandomForestRegressor(

                n_estimators=100,

                random_state=random_state,

                n_jobs=-1
            ),

            max_iter=max_iter,

            tol=tol,

            initial_strategy="mean",

            random_state=random_state
        )

    # =====================================================
    # LINEAR REGRESSION
    # =====================================================

    elif method == "lr":

        imputer = IterativeImputer(

            estimator=LinearRegression(),

            max_iter=max_iter,

            tol=tol,

            initial_strategy="mean",

            random_state=random_state
        )

    # =====================================================
    # KNN
    # =====================================================

    elif method == "knn":

        imputer = KNNImputer(

            n_neighbors=5,

            weights="distance"
        )

    # =====================================================
    # MEAN
    # =====================================================

    elif method == "mean":

        imputer = SimpleImputer(

            strategy="mean"
        )

    else:

        raise ValueError(
            "Invalid imputation method"
        )

    imputed_array = imputer.fit_transform(X)

    return pd.DataFrame(

        imputed_array,

        columns=X.columns,

        index=X.index
    )

# =========================================================
# TRANSFORMATION FUNCTION
# =========================================================

@st.cache_data

def apply_transform(

    name,

    _transformer,

    data
):

    transformed = _transformer.fit_transform(data)

    result = pd.DataFrame(

        transformed,

        columns=data.columns,

        index=data.index
    )

    # =====================================================
    # FORCE POSITIVE
    # =====================================================

    min_val = result.min().min()

    if min_val <= 0:

        shift = abs(min_val) + 1

        result += shift

    return result

# =========================================================
# CV FUNCTION
# =========================================================

@st.cache_data

def compute_group_cv(df, groups):

    cv_results = []

    for group_name, cols in groups.items():

        values = df[cols].to_numpy()

        means = values.mean(axis=1)

        stds = values.std(axis=1)

        cv = np.where(

            means > 1e-8,

            (stds / means) * 100,

            np.nan
        )

        cv_results.extend(cv)

    return np.nanmedian(cv_results)

# =========================================================
# VOLCANO FUNCTION
# =========================================================

def volcano_plot(

    ax,

    df,

    fc_col,

    pval_col="minuslog10_BH_FDR"
):

    data = df.copy()

    data["significance"] = "Not Significant"

    # =====================================================
    # UPREGULATED
    # =====================================================

    data.loc[
        (
            data[fc_col] >= fc_threshold
        ) &
        (
            data[pval_col] >= 1.3
        ),

        "significance"

    ] = "Upregulated"

    # =====================================================
    # DOWNREGULATED
    # =====================================================

    data.loc[
        (
            data[fc_col] <= -fc_threshold
        ) &
        (
            data[pval_col] >= 1.3
        ),

        "significance"

    ] = "Downregulated"

    # =====================================================
    # COLORS
    # =====================================================

    palette = {

        "Not Significant":
            "lightgray",

        "Upregulated":
            "#D62728",

        "Downregulated":
            "#1F77B4"
    }

    # =====================================================
    # PLOT
    # =====================================================

    for category in palette.keys():

        subset = data[
            data["significance"] == category
        ]

        ax.scatter(

            subset[fc_col],

            subset[pval_col],

            color=palette[category],

            alpha=0.7,

            s=30,

            edgecolor="black",

            linewidth=0.3
        )

    # =====================================================
    # LABELS
    # =====================================================

    if show_volcano_labels:

        for idx, row in data.iterrows():

            if (
                abs(row[fc_col]) >= fc_threshold
                and
                row[pval_col] >= 3
            ):

                ax.text(

                    row[fc_col],

                    row[pval_col],

                    str(idx),

                    fontsize=11,
                    
                    weight="bold"
                )

    # =====================================================
    # THRESHOLDS
    # =====================================================

    ax.axvline(

        fc_threshold,

        linestyle="--",

        color="black"
    )

    ax.axvline(

        -fc_threshold,

        linestyle="--",

        color="black"
    )

    ax.axhline(

        1.3,

        linestyle="--",

        color="black"
    )

    display_title = (
        fc_col
        .replace("log2FC_", "")
        .replace("_vs_", " vs ")
        .replace("_", " ")
    )
    
    ax.set_title(
        display_title,
        fontsize=17,
        weight="bold"
    )
    
    ax.set_xlabel(
        "log2 Fold Change",
        fontsize=15,
        weight="bold"
    )
    
    ax.set_ylabel(
        "-log10 adjusted p-value",
        fontsize=15,
        weight="bold"
    )
    
    # Increase x-axis and y-axis number sizes
    ax.tick_params(
        axis="both",
        labelsize=13
    )

    ax.grid(alpha=0.2)

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv(

        uploaded_file,

        index_col=0
    )

    st.subheader("Raw Data")

    st.dataframe(df.head())

    st.write(f"Shape: {df.shape}")

    # =====================================================
    # GROUP DETECTION
    # =====================================================

    groups = {}

    for col in df.columns:

        if "_" in col:

            group, num = col.rsplit("_", 1)

            if num.isdigit():

                groups.setdefault(group, []).append(col)

    st.subheader("Detected Groups")

    st.write(groups)

    # =====================================================
    # INTENSITY COLUMNS
    # =====================================================

    intensity_cols = [

        col

        for cols in groups.values()

        for col in cols
    ]

    X = df[intensity_cols].copy()

    # =====================================================
    # REPLACE ZEROS
    # =====================================================

    X = X.replace(0, np.nan)

    # =====================================================
    # LOG2 TRANSFORMATION
    # =====================================================

    if already_log2:

        X_log2 = X.copy()

    else:

        X_log2 = np.log2(X)

    # =====================================================
    # REMOVE INFINITE VALUES
    # =====================================================

    X_log2 = X_log2.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =====================================================
    # MISSING VALUES
    # =====================================================

    st.subheader("Missing Values")

    fig, ax = plt.subplots(

        figsize=(10, 4)
    )

    sns.heatmap(

        X_log2.isnull(),

        cmap="viridis",

        cbar=False,

        ax=ax
    )

    st.pyplot(fig)

    # =====================================================
    # IMPUTATION
    # =====================================================

    st.subheader("Imputation")

    imputed_df = impute_data(

        X_log2,

        method=imputation_method.lower()
    )

    st.success(

        f"Imputation completed using {imputation_method}"
    )

    # =====================================================
    # TRANSFORMATIONS
    # =====================================================

    scalers = {

        "StandardScaler":
            StandardScaler(),

        "MinMaxScaler":
            MinMaxScaler(feature_range=(1, 2)),

        "QuantileTransformer":
            QuantileTransformer(

                n_quantiles=min(
                    1000,
                    imputed_df.shape[0]
                ),

                output_distribution="normal",

                random_state=0
            ),

        "RobustScaler":
            RobustScaler(),

        "PowerTrans_YJ":
            PowerTransformer(
                method="yeo-johnson"
            ),

        "PowerTrans_BoxCox":
            PowerTransformer(
                method="box-cox"
            )
    }

    transformed_dfs = {

        "Imputed":
            imputed_df
    }

    # =====================================================
    # APPLY NORMALIZATIONS
    # =====================================================

    for scaler_name, scaler in scalers.items():

        try:

            transformed_dfs[scaler_name] = apply_transform(

                scaler_name,

                scaler,

                imputed_df
            )

        except Exception as e:

            st.warning(

                f"{scaler_name} failed: {e}"
            )

    # =====================================================
    # SQRT
    # =====================================================

    sqrt_df = imputed_df.copy()

    min_val = sqrt_df.min().min()

    if min_val <= 0:

        sqrt_df += abs(min_val) + 1

    transformed_dfs["SQRT"] = np.sqrt(sqrt_df)

    # =====================================================
    # NORMALIZATION QC
    # =====================================================

    st.subheader("Normalization QC")

    cv_summary = []

    for name, df_trans in transformed_dfs.items():

        median_cv = compute_group_cv(

            df_trans,

            groups
        )

        cv_summary.append({

            "Transformation":
                name,

            "Median CV":
                median_cv,

            "Pass":
                median_cv < cv_threshold
        })

    cv_summary = pd.DataFrame(cv_summary)

    cv_summary = cv_summary.sort_values(
        "Median CV"
    )

    st.dataframe(cv_summary)

    # =====================================================
    # SUGGESTED NORMALIZATION
    # =====================================================

    suggested = cv_summary.iloc[0]["Transformation"]

    st.success(

        f"Suggested normalization: {suggested}"
    )
    

    # =====================================================
    # ANALYSIS DATA
    # =====================================================
    
    analysis_df = transformed_dfs[
        normalization_method
    ]
    
    # =====================================================
    # PCA GRID ACROSS NORMALIZATIONS
    # =====================================================
    
    if show_pca_grid:
    
        st.subheader(
            "PCA Grid Across Normalizations"
        )
    
        n_methods = len(transformed_dfs)
    
        n_cols = 3
    
        n_rows = int(
            np.ceil(n_methods / n_cols)
        )
    
        fig, axes = plt.subplots(
    
            n_rows,
    
            n_cols,
    
            figsize=(7 * n_cols, 6 * n_rows)
        )
    
        axes = np.array(axes).flatten()
    
        for ax, (method_name, df_trans) in zip(
    
            axes,
    
            transformed_dfs.items()
        ):
    
            # =================================================
            # SCALE
            # =================================================
    
            scaled_pca = StandardScaler().fit_transform(
    
                df_trans.T
            )
    
            # =================================================
            # PCA
            # =================================================
    
            pca = PCA(n_components=2)
    
            pcs = pca.fit_transform(
                scaled_pca
            )
    
            # =================================================
            # PCA DF
            # =================================================
    
            sample_info = pd.DataFrame({
    
                "Sample": intensity_cols,
    
                "Group": [
    
                    col.rsplit("_", 1)[0]
    
                    for col in intensity_cols
                ]
            })
    
            pca_df = pd.DataFrame({
    
                "PC1": pcs[:, 0],
    
                "PC2": pcs[:, 1],
    
                "Group": sample_info["Group"]
            })
    
            # =================================================
            # PLOT
            # =================================================
    
            sns.scatterplot(
    
                data=pca_df,
    
                x="PC1",
    
                y="PC2",
    
                hue="Group",
    
                s=150,
    
                ax=ax
            )
    
            # =================================================
            # VARIANCE
            # =================================================
    
            explained_var = (
    
                pca.explained_variance_ratio_
                * 100
            )
    
            ax.set_title(
    
                f"{method_name}\n"
                f"PC1: {explained_var[0]:.1f}% | "
                f"PC2: {explained_var[1]:.1f}%",
    
                fontsize=12,
    
                weight="bold"
            )
    
            ax.grid(alpha=0.2)
    
            ax.legend(
                fontsize=8
            )
    
        # =====================================================
        # REMOVE EMPTY AXES
        # =====================================================
    
        for i in range(
    
            len(transformed_dfs),
    
            len(axes)
        ):
    
            fig.delaxes(axes[i])
    
        plt.tight_layout()
    
        st.pyplot(fig)
        
        
        # =====================================================
        # PCA GRID SVG EXPORT
        # =====================================================
        
        pca_svg = io.BytesIO()
        
        fig.savefig(
        
            pca_svg,
        
            format="svg",
        
            bbox_inches="tight"
        )
        
        st.download_button(
        
            label="Download PCA Grid SVG",
        
            data=pca_svg.getvalue(),
        
            file_name="PCA_Grid.svg",
        
            mime="image/svg+xml"
        )   
                
        
    
    
    # =====================================================
    # SINGLE PCA
    # =====================================================
    
    st.subheader("Selected Normalization PCA")
    
    scaled_pca = StandardScaler().fit_transform(
        analysis_df.T
    )
    
    pca = PCA(n_components=2)
    
    pcs = pca.fit_transform(scaled_pca)
    
    sample_info = pd.DataFrame({
    
        "Sample": intensity_cols,
    
        "Group": [
            col.rsplit("_", 1)[0]
            for col in intensity_cols
        ]
    })
    
    pca_df = pd.DataFrame({
    
        "PC1": pcs[:, 0],
    
        "PC2": pcs[:, 1],
    
        "Group": sample_info["Group"]
    })
    
    fig, ax = plt.subplots(
    
        figsize=(8, 6)
    )
    
    sns.scatterplot(
    
        data=pca_df,
    
        x="PC1",
    
        y="PC2",
    
        hue="Group",
    
        s=180,
    
        ax=ax
    )
    
    explained_var = (
        pca.explained_variance_ratio_
        * 100
    )
    
    ax.set_title(
    
        f"{normalization_method}\n"
        f"PC1: {explained_var[0]:.1f}% | "
        f"PC2: {explained_var[1]:.1f}%",
    
        fontsize=14,
    
        weight="bold"
    )
    
    ax.grid(alpha=0.2)
    
    st.pyplot(fig)
    
    
    
    pca_svg = io.BytesIO()

    fig.savefig(
    
        pca_svg,
    
        format="svg",
    
        bbox_inches="tight"
    )
    
    st.download_button(
    
        label="Download PCA SVG",
    
        data=pca_svg.getvalue(),
    
        file_name="PCA_Grid.svg",
    
        mime="image/svg+xml"
    )
    
    # =====================================================
    # DIFFERENTIAL ANALYSIS
    # =====================================================
    
    st.subheader("Differential Analysis")
    
    group_pairs = list(
        combinations(groups.keys(), 2)
    )
    
    results = pd.DataFrame(
        index=analysis_df.index
    )
    
    # =====================================================
    # GROUP MEANS
    # =====================================================
    
    group_means = {}
    
    for group_name, cols in groups.items():
    
        group_means[group_name] = (
    
            analysis_df[cols]
            .mean(axis=1)
        )
    
    # =====================================================
    # LOG2FC
    # =====================================================
    
    for g1, g2 in group_pairs:
    
        fc_name = f"log2FC_{g2}_vs_{g1}"
    
        results[fc_name] = (
    
            group_means[g2]
    
            -
    
            group_means[g1]
        )
    
    # =====================================================
    # PVALUES
    # =====================================================
    
    pvals = []
    
    selected_tests = []
    
    for protein in analysis_df.index:
    
        group_values = []
    
        normality_pass = True
    
        # =================================================
        # NORMALITY TEST
        # =================================================
    
        for group_name, cols in groups.items():
    
            vals = (
    
                analysis_df
    
                .loc[protein, cols]
    
                .dropna()
    
                .values
            )
    
            group_values.append(vals)
    
            try:
    
                if len(vals) >= 3:
    
                    _, p = shapiro(vals)
    
                    if p < 0.05:
    
                        normality_pass = False
    
            except:
    
                normality_pass = False
    
        # =================================================
        # TEST SELECTION
        # =================================================
    
        try:
    
            if normality_pass:
    
                test_result = f_oneway(
                    *group_values
                )
    
                selected_tests.append(
                    "ANOVA"
                )
    
            else:
    
                test_result = kruskal(
                    *group_values
                )
    
                selected_tests.append(
                    "Kruskal"
                )
    
            pvals.append(
                test_result.pvalue
            )
    
        except:
    
            pvals.append(np.nan)
    
            selected_tests.append(
                "Failed"
            )
    
    # =====================================================
    # STORE RESULTS
    # =====================================================
    
    results["pvalue"] = pvals
    
    results["Test"] = selected_tests
    
    # =====================================================
    # MULTIPLE TESTING
    # =====================================================
    
    results["BH_FDR"] = multipletests(
    
        results["pvalue"].fillna(1),
    
        method="fdr_bh"
    
    )[1]
    
    results["BY_FDR"] = multipletests(
    
        results["pvalue"].fillna(1),
    
        method="fdr_by"
    
    )[1]
    
    results["Bonferroni"] = multipletests(
    
        results["pvalue"].fillna(1),
    
        method="bonferroni"
    
    )[1]
    
    # =====================================================
    # -LOG10
    # =====================================================
    
    results["minuslog10_significance"] = (
    
        -np.log10(
            results[significance_column]
        )
    )
    
    # =====================================================
    # RESULTS TABLE
    # =====================================================
    
    st.dataframe(
        results.head()
    )
    
    # =====================================================
    # VOLCANO GRID
    # =====================================================
    
    st.subheader("Volcano Plot Grid")
    
    fc_columns = [
    
        col
    
        for col in results.columns
    
        if col.startswith("log2FC")
    ]
    
    n_plots = len(fc_columns)
    
    n_cols = 3
    
    n_rows = int(
        np.ceil(n_plots / n_cols)
    )
    
    fig, axes = plt.subplots(
    
        n_rows,
    
        n_cols,
    
        figsize=(7 * n_cols, 6 * n_rows)
    )
    
    axes = np.array(axes).flatten()
    
    # =====================================================
    # VOLCANO LOOP
    # =====================================================
    
    for ax, fc_col in zip(axes, fc_columns):
    
        volcano_plot(
    
            ax,
    
            results,
    
            fc_col,
    
            pval_col="minuslog10_significance"
        )
    
        # =================================================
        # LABELS
        # =================================================
    
        if show_volcano_labels:
    
            for idx, row in results.iterrows():
    
                if (
                    abs(row[fc_col]) >= fc_threshold
                    and
                    row["minuslog10_significance"] >= 3
                ):
    
                    ax.text(
    
                        row[fc_col],
    
                        row["minuslog10_significance"],
    
                        str(idx),
    
                        fontsize=7
                    )
    
    # =====================================================
    # REMOVE EMPTY AXES
    # =====================================================
    
    for i in range(
    
        len(fc_columns),
    
        len(axes)
    ):
    
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    st.pyplot(fig)
    
    
    # =====================================================
    # VOLCANO SVG EXPORT
    # =====================================================
    
    volcano_svg = io.BytesIO()
    
    fig.savefig(
    
        volcano_svg,
    
        format="svg",
    
        bbox_inches="tight"
    )
    
    st.download_button(
    
        label="Download Volcano Grid SVG",
    
        data=volcano_svg.getvalue(),
    
        file_name="Volcano_Grid.svg",
    
        mime="image/svg+xml"
    )
    
    
    

    
    # =====================================================
    # GO ENRICHMENT
    # =====================================================
    
    st.subheader("GO Enrichment")
    
    # =====================================================
    # SETTINGS
    # =====================================================
    
    TOP_TERMS_PER_COMPARISON = 10
    
    TOP_TERMS_TOTAL = 40
    
    MIN_COUNTS = 2
    
    bubble_scale = 75
    
    # =====================================================
    # SIGNIFICANT IDS
    # =====================================================
    
    significant_ids = results[
    
        results[GO_SIGNIFICANCE]
        < GO_PVALUE_THRESHOLD
    
    ].index.astype(str).unique().tolist()
    
    significant_ids = list(
        set(significant_ids)
    )
    
    # =====================================================
    # CONVERT IDS
    # =====================================================
    
    with st.spinner(
        "Converting UniProt IDs..."
    ):
    
        mg = MyGeneInfo()
    
        query_results = mg.querymany(
    
            significant_ids,
    
            scopes="uniprot",
    
            fields="symbol",
    
            species=9606,
    
            verbose=False,
    
            returnall=False
        )
    
    # =====================================================
    # BUILD MAPPING
    # =====================================================
    
    mapping = {
    
        r["query"]: r.get("symbol")
    
        for r in query_results
    
        if (
            "symbol" in r
            and
            r.get("symbol") is not None
        )
    }
    
    results["GeneSymbol"] = (
    
        results.index
        .map(mapping)
    )
    
    st.write(
        f"Mapped genes: {len(mapping)}"
    )
    
    # =====================================================
    # STORAGE
    # =====================================================
    
    all_go_up = []
    
    all_go_down = []
    
    # =====================================================
    # GO LOOP
    # =====================================================
    
    for fc_col in fc_columns:
    
        comparison_name = fc_col.replace(
    
            "log2FC_",
    
            ""
        )
    
        st.write(
            f"Running enrichment: {comparison_name}"
        )
    
        # =================================================
        # UPREGULATED
        # =================================================
    
        sig_up = results[
    
            (
                results[GO_SIGNIFICANCE]
                < GO_PVALUE_THRESHOLD
            ) &
    
            (
                results[fc_col]
                > GO_FC_THRESHOLD
            )
        ]
    
        up_genes = (
    
            sig_up["GeneSymbol"]
    
            .dropna()
    
            .unique()
    
            .tolist()
        )
    
        if len(up_genes) >= 2:
    
            try:
    
                enr_up = gp.enrichr(
    
                    gene_list=up_genes,
    
                    gene_sets=GO_DATABASE,
    
                    organism="human",
    
                    outdir=None,
    
                    no_plot=True
                )
    
                go_up = enr_up.results.copy()
    
                if not go_up.empty:
    
                    go_up["Counts"] = [
    
                        int(x.split("/")[0])
    
                        for x in go_up["Overlap"]
                    ]
    
                    # =========================================
                    # FILTER COUNTS
                    # =========================================
    
                    go_up = go_up[
    
                        go_up["Counts"]
                        >= MIN_COUNTS
                    ]
    
                    go_up = go_up.sort_values(
    
                        "Adjusted P-value"
    
                    ).head(
                        TOP_TERMS_PER_COMPARISON
                    )
    
                    go_up["Comparison"] = comparison_name
    
                    go_up["Direction"] = "Upregulated"
    
                    go_up["minuslog10FDR"] = (
    
                        -np.log10(
                            go_up["Adjusted P-value"]
                        )
                    )
    
                    all_go_up.append(go_up)
    
            except Exception as e:
    
                st.warning(
                    f"UP enrichment failed: {e}"
                )
    
        # =================================================
        # DOWNREGULATED
        # =================================================
    
        sig_down = results[
    
            (
                results[GO_SIGNIFICANCE]
                < GO_PVALUE_THRESHOLD
            ) &
    
            (
                results[fc_col]
                < -GO_FC_THRESHOLD
            )
        ]
    
        down_genes = (
    
            sig_down["GeneSymbol"]
    
            .dropna()
    
            .unique()
    
            .tolist()
        )
    
        if len(down_genes) >= 2:
    
            try:
    
                enr_down = gp.enrichr(
    
                    gene_list=down_genes,
    
                    gene_sets=GO_DATABASE,
    
                    organism="human",
    
                    outdir=None,
    
                    no_plot=True
                )
    
                go_down = enr_down.results.copy()
    
                if not go_down.empty:
    
                    go_down["Counts"] = [
    
                        int(x.split("/")[0])
    
                        for x in go_down["Overlap"]
                    ]
    
                    # =========================================
                    # FILTER COUNTS
                    # =========================================
    
                    go_down = go_down[
    
                        go_down["Counts"]
                        >= MIN_COUNTS
                    ]
    
                    go_down = go_down.sort_values(
    
                        "Adjusted P-value"
    
                    ).head(
                        TOP_TERMS_PER_COMPARISON
                    )
    
                    go_down["Comparison"] = comparison_name
    
                    go_down["Direction"] = "Downregulated"
    
                    go_down["minuslog10FDR"] = (
    
                        -np.log10(
                            go_down["Adjusted P-value"]
                        )
                    )
    
                    all_go_down.append(go_down)
    
            except Exception as e:
    
                st.warning(
                    f"DOWN enrichment failed: {e}"
                )
    
    # =====================================================
    # CONCAT
    # =====================================================
    
    if len(all_go_up) > 0:
    
        go_up_df = pd.concat(
    
            all_go_up,
    
            ignore_index=True
        )
    
    else:
    
        go_up_df = pd.DataFrame()
    
    if len(all_go_down) > 0:
    
        go_down_df = pd.concat(
    
            all_go_down,
    
            ignore_index=True
        )
    
    else:
    
        go_down_df = pd.DataFrame()
    
    # =====================================================
    # LIMIT TOTAL TERMS
    # =====================================================
    
    if not go_up_df.empty:
    
        top_up_terms = (
    
            go_up_df
    
            .groupby("Term")["minuslog10FDR"]
    
            .max()
    
            .sort_values(
                ascending=False
            )
    
            .head(TOP_TERMS_TOTAL)
    
            .index
        )
    
        go_up_df = go_up_df[
    
            go_up_df["Term"].isin(top_up_terms)
        ]
    
    if not go_down_df.empty:
    
        top_down_terms = (
    
            go_down_df
    
            .groupby("Term")["minuslog10FDR"]
    
            .max()
    
            .sort_values(
                ascending=False
            )
    
            .head(TOP_TERMS_TOTAL)
    
            .index
        )
    
        go_down_df = go_down_df[
    
            go_down_df["Term"].isin(top_down_terms)
        ]
    
    # =====================================================
    # STOP IF EMPTY
    # =====================================================
    
    if go_up_df.empty and go_down_df.empty:
    
        st.warning(
            "No GO enrichment results generated."
        )
    
    else:
    
        # =================================================
        # REMOVE GO IDS
        # =================================================
    
        for df_go in [go_up_df, go_down_df]:
    
            if not df_go.empty:
    
                df_go["Term"] = (
    
                    df_go["Term"]
    
                    .astype(str)
    
                    .str.replace(
    
                        r"\s*\(GO:\d+\)",
    
                        "",
    
                        regex=True
                    )
                )
    
        # =================================================
        # ABBREVIATIONS
        # =================================================
    
        abbreviations = {
    
            "Positive Regulation Of": "+ Reg. Of",
    
            "Negative Regulation Of": "- Reg. Of",
    
            "Regulation Of": "Reg. Of",
    
            "Cellular": "Cell",
    
            "Organization": "Org.",
    
            "Transport": "Trans.",
    
            "Metabolic Process": "Metab. Proc.",
    
            "Process": "Proc.",
    
            "Catabolic Process": "Catab. Proc.",
    
            "Biosynthetic Process": "Biosynth. Proc.",
    
            "Homeostasis": "Homeo.",
    
            "Differentiation": "Diff.",
    
            "Apoptotic": "Apop.",
    
            "Endothelial": "Endoth.",
    
            "Epithelial": "Epithel.",
    
            "Lipoprotein": "Lipoprot.",
    
            "Signaling": "Signal.",
    
            "Pathway": "Path.",
    
            "Receptor": "Rec.",
    
            "Receptors": "Recs.",
    
            "Chemotaxis": "Chemotax.",
    
            "Migration": "Migr.",
    
            "High-Density": "HDL",
    
            "Low-Density": "LDL",
    
            "Cholesterol": "Chol.",
    
            "Inflammatory Response": "Inflamm. Resp.",
    
            "Intermediate Filament": "Int. Filament",
    
            "Supramolecular Fiber": "Supramol. Fiber"
        }
    
        # =================================================
        # ABBREVIATION FUNCTION
        # =================================================
    
        def abbreviate_term(term):
    
            for old, new in abbreviations.items():
    
                term = term.replace(old, new)
    
            return term
    
        # =================================================
        # APPLY ABBREVIATIONS
        # =================================================
    
        for df_go in [go_up_df, go_down_df]:
    
            if not df_go.empty:
    
                df_go["Term"] = [
    
                    abbreviate_term(term)
    
                    for term in df_go["Term"]
                ]
    
                df_go["Term"] = [
    
                    "\n".join(
    
                        textwrap.wrap(
                            term,
                            width=28
                        )
                    )
    
                    for term in df_go["Term"]
                ]
    
        # =====================================================
        # FIGURE GRID
        # =====================================================
    
        fig, axes = plt.subplots(
    
            ncols=2,
    
            figsize=(24, 18),
    
            sharey=False
        )
    
        plot_configs = [
    
            (
                go_up_df,
                axes[0],
                "UPREGULATED PATHWAYS"
            ),
    
            (
                go_down_df,
                axes[1],
                "DOWNREGULATED PATHWAYS"
            )
        ]
    
        # =====================================================
        # PLOT LOOP
        # =====================================================
    
        for df_plot, ax, title in plot_configs:
    
            if df_plot.empty:
    
                ax.set_visible(False)
    
                continue
    
            scatter = ax.scatter(
    
                x=df_plot["Comparison"],
    
                y=df_plot["Term"],
    
                s=df_plot["Counts"] * bubble_scale,
    
                c=df_plot["minuslog10FDR"],
    
                cmap="cividis",
    
                edgecolor="black",
    
                linewidth=0.5,
    
                alpha=0.9
            )
    
            ax.set_title(
    
                title,
    
                fontsize=25,
    
                weight="bold"
            )
    
            ax.set_xlabel(
    
                "Comparison",
    
                fontsize=21,
    
                weight="bold"
            )
    
            ax.tick_params(
    
                axis="x",
    
                rotation=45,
    
                labelsize=15
            )
    
            ax.tick_params(
    
                axis="y",
    
                labelsize=14
            )
    
            ax.grid(
    
                alpha=0.2,
    
                linestyle="--"
            )
    
            cbar = fig.colorbar(
    
                scatter,
    
                ax=ax,
    
                shrink=0.55,
    
                pad=0.02
            )
    
            cbar.set_label(
    
                "-log10(FDR)",
    
                fontsize=16,
    
                weight="bold"
            )
    
        # =====================================================
        # Y LABELS
        # =====================================================
    
        axes[0].set_ylabel(
    
            "GO Biological Process",
    
            fontsize=18,
    
            weight="bold"
        )
    
        axes[1].set_ylabel(
    
            "GO Biological Process",
    
            fontsize=19,
    
            weight="bold"
        )
    
        # =====================================================
        # LEGEND
        # =====================================================
    
        all_counts = pd.concat([
    
            go_up_df["Counts"],
    
            go_down_df["Counts"]
        ])
    
        min_count = int(all_counts.min())
    
        max_count = int(all_counts.max())
    
        mid_count = int(
            (min_count + max_count) / 2
        )
    
        legend_sizes = [
    
            min_count,
    
            mid_count,
    
            max_count
        ]
    
        for size in legend_sizes:
    
            axes[1].scatter(
    
                [],
    
                [],
    
                s=size * bubble_scale,
    
                color="gray",
    
                alpha=0.5,
    
                edgecolor="black",
    
                label=f"{size}"
            )
    
        axes[1].legend(
    
            title="Protein Counts",
    
            title_fontsize=16,
    
            fontsize=15,
    
            bbox_to_anchor=(1.28, 1),
    
            loc="upper left"
        )
    
        # =====================================================
        # MAIN TITLE
        # =====================================================
    
        fig.suptitle(
    
            "GO Enrichment Comparison Grid",
    
            fontsize=27,
    
            weight="bold",
    
            y=1.02
        )
    
        plt.tight_layout()
    
        # =====================================================
        # STREAMLIT DISPLAY
        # =====================================================
        
        st.pyplot(fig)
        
        # =====================================================
        # SVG EXPORT
        # =====================================================
        
        svg_buffer = io.BytesIO()

        fig.savefig(
            svg_buffer,
            format="svg",
            bbox_inches="tight"
        )
        
        st.download_button(
            label="Download SVG",
            data=svg_buffer.getvalue(),
            file_name="figure.svg",
            mime="image/svg+xml"
        )
