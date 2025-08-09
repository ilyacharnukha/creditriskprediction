import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix
import gender_guesser.detector as gender
import gc
from itertools import combinations
from sklearn.metrics import roc_auc_score
import shap
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
import dcor

from itertools import combinations
from sklearn.metrics import roc_auc_score
import shap
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
import dcor


plt.rcParams['font.family'] = 'DejaVu Serif'

def find_outliers(var_name, threshold, df):
    df = df.copy()
    df["z_score"] = np.abs(stats.zscore(df[var_name]))
    outliers = df[var_name][df["z_score"] > threshold]
    print(f'Outliers for the "{var_name}" column:\n\n{outliers}')

def predict_in_batches(model, data, batch_size=512):
    outputs = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        out = model.predict(batch, verbose=0)
        outputs.append(out)
        print(f"Processed batch {i // batch_size + 1}/{(len(data) // batch_size) + 1}")
    return np.vstack(outputs)

def predict_in_batches(model, data, batch_size=512):
    outputs = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        out = model.predict(batch, verbose=0)
        outputs.append(out)
        print(f"Processed batch {i // batch_size + 1}/{(len(data) // batch_size) + 1}")
    return np.vstack(outputs)

def find_outliers_iqr(var_name, threshold, df):
    Q1 = df[var_name].quantile(0.25)
    Q3 = df[var_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    print(df[(df[var_name] < lower) | (df[var_name] > upper)])

def add_model_results(results_df, name, best_score):
    new_result = pd.DataFrame([[name, best_score]], columns=["Model", "Best Score"])
    return pd.concat([results_df, new_result], ignore_index=True)

def add_results(results_df, name, best_score):
    new_result = pd.DataFrame([[name, best_score]], columns=["Feature", "Score"])
    return pd.concat([results_df, new_result], ignore_index=True)

def compare_mi(df, col):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_rows', None)
    df = df.copy()
    results = pd.DataFrame(columns=["Feature", "Score"])
    
    for feature in df.columns:
        if feature == col:
            continue
        try:
            score = mi_test(df, feature, col)
            results = add_results(results, feature, score)
        except Exception as e:
            print(f"Skipped {feature}: {e}")
    
    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)

def compare_mi_three_way(df, target):
    features = [col for col in df.columns if col != target]
    summary = []
    pair_scores = {}

    for f1, f2 in combinations(features, 2):
        try:
            score = mi_test_three_way(df, f1, f2, target)
            pair_key = f"{f1} + {f2}"
            pair_scores[pair_key] = score
        except Exception as e:
            pair_scores[f"{f1} + {f2}"] = np.nan

    feature_stats = []
    for feature in features:
        relevant_scores = [score for pair, score in pair_scores.items() if feature in pair]
        relevant_scores = [s for s in relevant_scores if not pd.isna(s)]
        if relevant_scores:
            feature_stats.append({
                "Name": feature,
                "Average": round(np.mean(relevant_scores), 3),
                "Median": round(np.median(relevant_scores), 3),
                "Max": round(np.max(relevant_scores), 3)
            })

    pair_df = pd.DataFrame(pair_scores.items(), columns=["Pair", "Score"]).dropna().sort_values(by="Score", ascending=False)
    stats_df = pd.DataFrame(feature_stats).sort_values(by="Average", ascending=False)
    return stats_df

def encode_categoricals(df):

    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include='category').columns:

        le = LabelEncoder()
        col_data = df[col].astype(str)
        all_classes = list(col_data.unique())

        if 'Unknown' not in all_classes:
            all_classes.append('Unknown')

        le.fit(all_classes) 
        df[col] = le.transform(col_data.fillna('Unknown'))
        encoders[col] = le
    return df, encoders

def apply_encoders(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            if 'Unknown' not in le.classes_:
                raise ValueError(
                    f"Encoder for column '{col}' was not trained with "
                    f"an 'Unknown' category. Cannot handle unseen values."
                )
            col_data = df[col].astype(str).fillna('Unknown').map(
                lambda x: x if x in le.classes_ else 'Unknown'
            )
            df[col] = le.transform(col_data)
    return df


def mi_test_three_way(df, feature1, feature2, target):
    df = df.copy()
    df['temp1'] = LabelEncoder().fit_transform(df[feature1])
    df['temp2'] = LabelEncoder().fit_transform(df[feature2])
    df['temp'] = df['temp1'].astype(str) + "_" + df['temp2'].astype(str)
    mi = mutual_info_classif(df[['temp']], df[target], discrete_features=True, random_state=42)
    df.drop(columns=["temp", "temp1", "temp2"], inplace=True)
    return mi[0]


def compare_mi_three_way(df, target):
    features = [col for col in df.columns if col != target]
    summary = []
    pair_scores = {}

    for f1, f2 in combinations(features, 2):
        try:
            score = mi_test_three_way(df, f1, f2, target)
            pair_key = f"{f1} + {f2}"
            pair_scores[pair_key] = score
        except Exception as e:
            pair_scores[f"{f1} + {f2}"] = np.nan

    feature_stats = []
    for feature in features:
        relevant_scores = [score for pair, score in pair_scores.items() if feature in pair]
        relevant_scores = [s for s in relevant_scores if not pd.isna(s)]
        if relevant_scores:
            feature_stats.append({
                "Name": feature,
                "Average": round(np.mean(relevant_scores), 3),
                "Median": round(np.median(relevant_scores), 3),
                "Max": round(np.max(relevant_scores), 3)
            })

    pair_df = pd.DataFrame(pair_scores.items(), columns=["Pair", "Score"]).dropna().sort_values(by="Score", ascending=False)
    stats_df = pd.DataFrame(feature_stats).sort_values(by="Average", ascending=False)
    return stats_df

def encode_categoricals(df):

    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include='category').columns:

        le = LabelEncoder()
        col_data = df[col].astype(str)
        all_classes = list(col_data.unique())

        if 'Unknown' not in all_classes:
            all_classes.append('Unknown')

        le.fit(all_classes) 
        df[col] = le.transform(col_data.fillna('Unknown'))
        encoders[col] = le
    return df, encoders

def apply_encoders(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            if 'Unknown' not in le.classes_:
                raise ValueError(
                    f"Encoder for column '{col}' was not trained with "
                    f"an 'Unknown' category. Cannot handle unseen values."
                )
            col_data = df[col].astype(str).fillna('Unknown').map(
                lambda x: x if x in le.classes_ else 'Unknown'
            )
            df[col] = le.transform(col_data)
    return df


def mi_test_three_way(df, feature1, feature2, target):
    df = df.copy()
    df['temp1'] = LabelEncoder().fit_transform(df[feature1])
    df['temp2'] = LabelEncoder().fit_transform(df[feature2])
    df['temp'] = df['temp1'].astype(str) + "_" + df['temp2'].astype(str)
    mi = mutual_info_classif(df[['temp']], df[target], discrete_features=True, random_state=42)
    df.drop(columns=["temp", "temp1", "temp2"], inplace=True)
    return mi[0]

def mi_test(df, feature, target):
    df = df.copy()
    df['temp'] = LabelEncoder().fit_transform(df[feature])
    mi = mutual_info_classif(df[['temp']], df[target], discrete_features=True, random_state=42)
    df.drop(columns=["temp"], inplace=True)
    return mi[0]

def auc_test(df, feature, target):
    df = df.copy()
    x = df[feature].dropna()
    y = df[target].loc[x.index]
    
    if y.nunique() < 2:
        raise ValueError("Target must have at least two classes")

    return roc_auc_score(y, x)

def compare_auc(df, col):
    df = df.copy()
    results = pd.DataFrame(columns=["Feature", "Score"])
    
    for feature in df.columns:
        if feature == col:
            continue
        try:
            score = auc_test(df, feature, col)
            results = add_results(results, feature, score)
        except Exception as e:
            print(f"Skipped {feature}: {e}")
    
    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)

def pi_test(df, feature, target, model=None):
    df = df.copy()
    x = df[[feature]].fillna(0)
    y = df[target]
    
    if model is None:
        model = GradientBoostingClassifier()
        model.fit(x, y)

    result = permutation_importance(model, x, y, n_repeats=5, random_state=42)
    return result.importances_mean[0]

def compare_pi(df, target, model=None):
    results = pd.DataFrame(columns=["Feature", "Score"])
    df = df.copy()

    for feature in df.columns:
        if feature == target:
            continue
        try:
            if df[feature].dtype == 'object' or df[feature].dtype.name == 'category':
                df.loc[:, feature] = LabelEncoder().fit_transform(df[feature].astype(str))
            score = pi_test(df, feature, target, model)
            results = add_results(results, feature, score)
        except Exception as e:
            print(f"Skipped {feature}: {e}")
    
    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)


def shap_test(df, feature, target, model=None):
    df = df.copy()
    x = df[[feature]].fillna(0)
    y = df[target]
    
    if model is None:
        model = GradientBoostingClassifier()
        model.fit(x, y)

    explainer = shap.Explainer(model, x)
    shap_values = explainer(x)
    return np.abs(shap_values.values).mean()

def compare_shap(df, target):
    df = df.copy()
    X = df.drop(columns=[target]).fillna(0)
    y = df[target]

    model = GradientBoostingClassifier()
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap_mean = np.abs(shap_values.values).mean(axis=0)

    results = pd.DataFrame({
        "Feature": X.columns,
        "Score": shap_mean
    })

    return results.sort_values("Score", ascending=False).reset_index(drop=True)

def corr_test(df, feature, target):
    df = df.copy()
    y = df[target].dropna()
    x = df[feature].loc[y.index].dropna()
    y = y.loc[x.index]

    mean_total = np.mean(x)
    ss_total = np.sum((x - mean_total) ** 2)
    ss_between = sum([
        len(x[y == cls]) * (np.mean(x[y == cls]) - mean_total) ** 2
        for cls in np.unique(y)
    ])
    return ss_between / ss_total if ss_total != 0 else 0

def compare_corr(df, target):
    results = pd.DataFrame(columns=["Feature", "Score"])

    for feature in df.columns:
        if feature == target:
            continue
        try:
            score = corr_test(df, feature, target)
            results = add_results(results, feature, score)
        except Exception as e:
            print(f"Skipped {feature}: {e}")

    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)

def compare_xgb(df, target):
    df = df.copy()
    X = df.drop(columns=[target]).fillna(0)
    y = df[target]

    model = GradientBoostingClassifier()
    model.fit(X, y)

    results = pd.DataFrame(columns=["Feature", "Score"])
    for feature, score in zip(X.columns, model.feature_importances_):
        results = add_results(results, feature, score)

    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)



def auc_test(df, feature, target):
    df = df.copy()
    x = df[feature].dropna()
    y = df[target].loc[x.index]
    
    if y.nunique() < 2:
        raise ValueError("Target must have at least two classes")

    return roc_auc_score(y, x)

def compare_auc(df, col):
    df = df.copy()
    results = pd.DataFrame(columns=["Feature", "Score"])
    
    for feature in df.columns:
        if feature == col:
            continue
        try:
            score = auc_test(df, feature, col)
            results = add_results(results, feature, score)
        except Exception as e:
            print(f"Skipped {feature}: {e}")
    
    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)

def pi_test(df, feature, target, model=None):
    df = df.copy()
    x = df[[feature]].fillna(0)
    y = df[target]
    
    if model is None:
        model = GradientBoostingClassifier()
        model.fit(x, y)

    result = permutation_importance(model, x, y, n_repeats=5, random_state=42)
    return result.importances_mean[0]

def compare_pi(df, target, model=None):
    results = pd.DataFrame(columns=["Feature", "Score"])
    df = df.copy()

    for feature in df.columns:
        if feature == target:
            continue
        try:
            if df[feature].dtype == 'object' or df[feature].dtype.name == 'category':
                df.loc[:, feature] = LabelEncoder().fit_transform(df[feature].astype(str))
            score = pi_test(df, feature, target, model)
            results = add_results(results, feature, score)
        except Exception as e:
            print(f"Skipped {feature}: {e}")
    
    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)


def shap_test(df, feature, target, model=None):
    df = df.copy()
    x = df[[feature]].fillna(0)
    y = df[target]
    
    if model is None:
        model = GradientBoostingClassifier()
        model.fit(x, y)

    explainer = shap.Explainer(model, x)
    shap_values = explainer(x)
    return np.abs(shap_values.values).mean()

def compare_shap(df, target):
    df = df.copy()
    X = df.drop(columns=[target]).fillna(0)
    y = df[target]

    model = GradientBoostingClassifier()
    model.fit(X, y)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap_mean = np.abs(shap_values.values).mean(axis=0)

    results = pd.DataFrame({
        "Feature": X.columns,
        "Score": shap_mean
    })

    return results.sort_values("Score", ascending=False).reset_index(drop=True)

def corr_test(df, feature, target):
    df = df.copy()
    y = df[target].dropna()
    x = df[feature].loc[y.index].dropna()
    y = y.loc[x.index]

    mean_total = np.mean(x)
    ss_total = np.sum((x - mean_total) ** 2)
    ss_between = sum([
        len(x[y == cls]) * (np.mean(x[y == cls]) - mean_total) ** 2
        for cls in np.unique(y)
    ])
    return ss_between / ss_total if ss_total != 0 else 0

def compare_corr(df, target):
    results = pd.DataFrame(columns=["Feature", "Score"])

    for feature in df.columns:
        if feature == target:
            continue
        try:
            score = corr_test(df, feature, target)
            results = add_results(results, feature, score)
        except Exception as e:
            print(f"Skipped {feature}: {e}")

    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)

def compare_xgb(df, target):
    df = df.copy()
    X = df.drop(columns=[target]).fillna(0)
    y = df[target]

    model = GradientBoostingClassifier()
    model.fit(X, y)

    results = pd.DataFrame(columns=["Feature", "Score"])
    for feature, score in zip(X.columns, model.feature_importances_):
        results = add_results(results, feature, score)

    return results.sort_values(by="Score", ascending=False).reset_index(drop=True)



def chi_square_test(df, feature, target):
    cont = pd.crosstab(df[feature], df[target], margins=True, margins_name="Total")
    print("Contingency Table:\n\n", cont, "\n\n")
    chi, p, dof, expected = chi2_contingency(cont)
    print(f"H0: Variable {feature} and {target} are independent\n", f"H1: Variable {feature} and {target} are NOT independent\n\n", f"P-value: {p:.4f}", sep="")

def compare_chi2(df, target_col):
    df = df.copy()
    results = pd.DataFrame(columns=["Feature", "Score"])

    for feature in df.columns:
        if feature == target_col:
            continue
        try:
            p_val = chi_square_test(df, feature, target_col)
            results = add_results(results, feature, p_val)
        except Exception as e:
            print(f"Skipped {feature}: {e}")

    return results.sort_values(by="Score").reset_index(drop=True)

def bin_difference(diff):
    if pd.isna(diff):
        return np.nan
    elif diff < 0.10:
        return "<110% difference"
    elif diff < 3.00:
        return "difference between 110% and 400%"
    else:
        return ">400% difference"

def draw_count(df, palette, big_label, xcol, add_to_height):
    df_temp = pd.DataFrame(df[xcol].value_counts(sort=False)).reset_index()
    df_temp.columns = ["feature", "count"]
    df_temp = df_temp.sort_values(by="count", ascending=False).reset_index(drop=True)

    total = df_temp["count"].sum()

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        x="feature",
        y="count",
        data=df_temp,
        hue="feature",
        palette=palette,
        width=0.8,
        legend=False,
        order=df_temp["feature"]
    )

    if ax.legend_ is not None:
        ax.legend_.remove()

    for p in ax.patches:
        height = p.get_height()
        percent = height / total * 100 if total > 0 else 0
        bar_label = f"{int(height)} ({percent:.0f}%)"

        ax.text(
            p.get_x() + p.get_width() / 2,
            height + add_to_height,
            bar_label,
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )

    ax.set_title(
        big_label,
        fontdict={"size": 16, "name": "DejaVu Serif"},
    )

    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)

    ax.set(
        yticks=[],
        ylabel="",
        xlabel="",
    )    


def draw_hist(df, big_label, bins, xcol, xlabel, color="#ffac1c"):
    plt.figure(figsize=(8, 6))
    ax = df[xcol].plot(kind="hist", bins=bins, color=color, edgecolor="black")
    ax.set_title(
        big_label,
        fontdict={"size": 16, "name": "DejaVu Serif"},
    )
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(xlabel)


def draw_lines(big_label, xcol, xlabel, ycol, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(xcol, ycol, "o-", color="red")
    ax = plt.gca()
    ax.set_title(
        big_label,
        fontdict={"size": 16, "name": "DejaVu Serif"},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    plt.show()


def draw_scatter(big_label, xcol, xlabel, ycol, ylabel, line=True):
    plt.figure(figsize=(8, 6))
    if line:
        sns.regplot(x=xcol, y=ycol, scatter=True, fit_reg=True)
    else:
        plt.scatter(xcol, ycol, color="blue")
    ax = plt.gca()
    ax.set_title(big_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    plt.grid(True)
    plt.show()

def draw_catplot(
    x,
    xlabel,
    y,
    ylabel,
    biglabel,
    kind,
    data,
    hue,
    palette=["#ffac1c"],
    legend=False,
):
    plot = sns.catplot(
        x=x,
        y=y,
        kind=kind,
        data=data,
        palette=palette,
        hue=hue,
        legend=legend,
        height=6,
        aspect=1.3
    )
    plot.set_axis_labels(xlabel, ylabel)
    plt.title(biglabel)
    for ax in plot.axes.flat:
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
    for ax in plot.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    plt.show()


def is_feature_useful(feature_number, X_train, X_train_pca):
    _ = feature_number - 1
    feature = X_train.iloc[:, _]
    component = X_train_pca[:, 0]
    draw_scatter(
        f"Component 1 vs. {X_train.columns[_].title()}",
        feature,
        f"{X_train.columns[_].title()}",
        component,
        f"Component 1",
    )


def draw_boxplot(df, palette, big_label, xcol, ycol, x_label, y_label):
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(
        x=xcol,
        y=ycol,
        hue=xcol,
        palette=palette,
        data=df,
        legend=False
    )

    plt.xlabel(x_label, fontsize=14, fontname="DejaVu Serif")
    plt.ylabel(y_label, fontsize=14, fontname="DejaVu Serif")
    plt.title(big_label, fontsize=16, fontname="DejaVu Serif")

    plt.xticks(fontsize=12, fontname="DejaVu Serif")
    plt.yticks(fontsize=12, fontname="DejaVu Serif")

    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()

def classify_bureau_balance(row):
    if row['pct_x'] >= 0.8:
        return 'unknown'
    if row['n'] == 0:
        return 'unknown'
    
    statuses = [s for s in ['0', '1', '2', '3', '4', '5'] if row[s] > 0]
    
    if set(statuses).issubset({'0'}):
        return 'perfect'
    elif set(statuses).issubset({'0', '1'}) and row['pct_1'] < 0.15:
        return 'good'
    elif set(statuses).issubset({'0', '1', '2'}) and row['pct_12'] < 0.20:
        return 'okay'
    else:
        return 'risky'
    
def classify_loan_timing(row):
    est = row['DAYS_CREDIT_ENDDATE']
    fact = row['DAYS_ENDDATE_FACT']
    
    if pd.isna(est) or pd.isna(fact):
        return 'unknown'
    if est == fact:
        return 'on time'
    elif fact + 30 < est:
        return 'early repay'
    elif fact > est + 30:
        return 'late'
    else:
        return 'fair'

def classify_sum_overdue(value):
    if pd.isna(value):
        return 'unknown'
    if value == 0:
        return 'perfect'
    elif value < np.exp(5):
        return 'good'
    elif value < np.exp(7):
        return 'fair'
    else:
        return 'risky'
    
def split_and_merge(left_df, right_df, on_col, how='left', nparts=16):
    split_dfs = np.array_split(right_df, nparts)
    merged_chunks = []
    for chunk in split_dfs:
        merged = left_df.merge(chunk, on=on_col, how=how)
        merged_chunks.append(merged)
        del merged, chunk
        gc.collect()
    return pd.concat(merged_chunks, ignore_index=True)

    
def classify_max_overdue(value):
    if pd.isna(value):
        return 'unknown'
    if value == 0:
        return 'perfect'
    elif value < np.exp(5):
        return 'good'
    elif value < np.exp(7):
        return 'fair'
    else:
        return 'risky'

def classify_overdue(group):
    overdue = group['CREDIT_DAY_OVERDUE'].dropna()
    
    if (overdue == 0).all():
        return 'perfect'
    
    total = len(overdue)
    if total == 0:
        return 'risky'
    
    below_14 = ((overdue > 0) & (overdue < 14)).sum() / total
    below_35 = ((overdue >= 14) & (overdue < 35)).sum() / total
    above_35 = (overdue >= 35).sum()
    
    if below_14 < 0.15 and above_35 == 0:
        return 'good'
    elif (below_14 + below_35) < 0.15 and above_35 == 0:
        return 'okay'
    else:
        return 'risky'


def bootstrap(array1, array2, n_iterations=5000):
    means = []
    for _ in range(n_iterations):
        sample1 = np.random.choice(array1, size=len(array1), replace=True)
        sample2 = np.random.choice(array2, size=len(array2), replace=True)
        mean_diff = np.mean(sample1) - np.mean(sample2)
        means.append(mean_diff)
        ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    
    print(f"Median difference in means: {np.median(means):.2f}")
    print(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}]")

def compute_distance_correlation_matrix(df):
    df = df.copy().fillna(df.median())
    cols = df.columns
    n = len(cols)
    
    dcor_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    
    for i in range(n):
        for j in range(i, n):
            val = dcor.distance_correlation(df.iloc[:, i], df.iloc[:, j])
            dcor_matrix.iloc[i, j] = val
            dcor_matrix.iloc[j, i] = val
    
    return dcor_matrix

def plot_correlation_heatmap(corr_matrix, title="Distance Correlation Matrix"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='viridis', annot=False, square=True,
                cbar_kws={'label': 'Distance Correlation'})
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()