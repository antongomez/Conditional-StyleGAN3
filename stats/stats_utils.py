import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import dunnett, ttest_1samp, ttest_rel, wilcoxon
from statsmodels.formula.api import ols

# ---------------------------------------------------------------


def calculate_anova(df, model_vars=["config", "seed"], dependent_var="aa_test", typ=2):
    """
    Performs an ANOVA analysis on the given DataFrame, grouped by dataset.
    """
    results = {}

    for dataset, group in df.groupby("dataset"):
        # Create the formula string for the OLS model
        formula = f"{dependent_var} ~ " + " + ".join([f"C({var})" for var in model_vars])

        # Fit the OLS model
        model = ols(formula, data=group).fit()

        # Get the ANOVA table
        anova_table = sm.stats.anova_lm(model, typ=typ)

        # Extract the results for this dataset
        dataset_results = {}
        for var in model_vars:
            f_statistic = anova_table.loc[f"C({var})", "F"]
            p_value = anova_table.loc[f"C({var})", "PR(>F)"]
            dataset_results[var] = (f_statistic, p_value)
        results[dataset] = dataset_results

    return results


# ---------------------------------------------------------------


def calculate_group_diffs(group, ref_config, dataset_type="test"):
    """
    Calculates the difference in metrics between experiments in a group and the baseline configuration.
    This function is intended to be used with groupby().apply().
    """
    # Find the baseline row for this group (config '00')
    baseline = group[group["config"] == ref_config]

    # If there's no baseline for this seed/group, we can't calculate differences
    if baseline.empty:
        # Return an empty DataFrame with the expected difference columns
        return pd.DataFrame(columns=["oa_diff", "aa_diff"])

    # Get baseline metrics
    baseline_oa = baseline[f"oa_{dataset_type}"].iloc[0]
    baseline_aa = baseline[f"aa_{dataset_type}"].iloc[0]

    # Calculate differences for the rest of the group
    group[f"oa_{dataset_type}_diff"] = group[f"oa_{dataset_type}"] - baseline_oa
    group[f"aa_{dataset_type}_diff"] = group[f"aa_{dataset_type}"] - baseline_aa

    return group


# ---------------------------------------------------------------


def calculate_ttest_significance(metric, group1, group2=None):
    """
    Performs a paired t-test to determine if the difference between two groups is statistically significant.
    """
    if group2 is not None:
        # Ensure the groups are aligned by seed
        merged = pd.merge(group1, group2, on="seed", suffixes=("_g1", "_g2"))
        if len(merged) < 2:
            return 1.0  # Not enough data to compute significance

        stat, p_value = ttest_rel(merged[f"{metric}_g1"], merged[f"{metric}_g2"])
    else:
        if len(group1) < 2:
            return 1.0, 0
        stat, p_value = ttest_1samp(group1[metric], 0)

    return stat, p_value


def calculate_wilcoxon_significance(metric, group1, group2=None):
    """
    Performs a Wilcoxon signed-rank test to determine if the difference between two groups is statistically significant.
    """
    if group2 is not None:
        merged = pd.merge(group1, group2, on="seed", suffixes=("_g1", "_g2"))
        if len(merged) < 2:
            return 1.0

        # Wilcoxon test requires the differences
        diff = merged[f"{metric}_g1"] - merged[f"{metric}_g2"]
    else:
        if len(group1) < 2:
            return 1.0, 0

        # In this case we have already computed the differences
        diff = group1[metric]

    if np.all(diff == 0):
        return 1.0

    stat, p_value = wilcoxon(diff)
    return stat, p_value


def calculate_dunnett_significance(raw_data, metric, ref_config, other_configs):
    """
    Performs Dunnett's test to compare multiple configurations against a control (reference configuration).
    """
    control_group = raw_data[ref_config][metric].values
    samples = [raw_data[cfg][metric].values for cfg in other_configs]

    if len(control_group) < 2 or any(len(sample) < 2 for sample in samples):
        return [0] * len(other_configs), [1.0] * len(other_configs)

    try:
        result = dunnett(*samples, control=control_group)
        return result.statistic, result.pvalue
    except Exception as e:
        warnings.warn(f"Dunnett's test failed with error: {e}")
        return [0] * len(other_configs), [1.0] * len(other_configs)


def calculate_significance(metric, group1, group2=None, test_type="ttest"):
    """
    Performs a statistical test to determine if the difference between two groups is statistically significant.
    If only one group is specified,
    """
    if test_type == "ttest":
        return calculate_ttest_significance(metric, group1, group2)
    elif test_type == "wilcoxon":
        return calculate_wilcoxon_significance(metric, group1, group2)
    else:
        raise ValueError(f"Unknown test type: {test_type}")


# ---------------------------------------------------------------


def calculate_stats(grouped_results, dataset_type="test", ref_config=None, test_type="ttest"):
    """Computes mean and std for each dataset and configuration."""
    stats = defaultdict(lambda: defaultdict(dict))
    raw_data = defaultdict(lambda: defaultdict(dict))

    metric_suffixes = [f"_{dataset_type}"]
    if ref_config is not None:
        metric_suffixes.append(f"_{dataset_type}_diff")

    for (dataset, config), group in grouped_results:
        n_experiments = len(group)
        if n_experiments < 5:
            warnings.warn(
                f"Warning: Dataset '{dataset}' with uniform_class={group['uniform_class'].iloc[0]}, "
                f"disc_on_gen={group['disc_on_gen'].iloc[0]} has only {n_experiments} experiments (expected 5)"
            )

        raw_data[dataset][config] = group
        stats[dataset][config] = {}

        for metric_suffix in metric_suffixes:
            oa_mean = group[f"oa{metric_suffix}"].mean() * 100
            oa_std = group[f"oa{metric_suffix}"].std(ddof=1) * 100
            aa_mean = group[f"aa{metric_suffix}"].mean() * 100
            aa_std = group[f"aa{metric_suffix}"].std(ddof=1) * 100

            stats[dataset][config][f"oa{metric_suffix}"] = (oa_mean, oa_std)
            stats[dataset][config][f"aa{metric_suffix}"] = (aa_mean, aa_std)

        # Compute significance for t-test and Wilcoxon
        if ref_config is not None and config != ref_config and test_type in ["ttest", "wilcoxon"]:
            # We can make directly a 1-sample test against 0 since we have already computed differences
            stat_oa, pvalue_oa = calculate_significance(f"oa_{dataset_type}_diff", group, test_type=test_type)
            stat_aa, pvalue_aa = calculate_significance(f"aa_{dataset_type}_diff", group, test_type=test_type)

            stats[dataset][config][f"oa_{dataset_type}_statistic"] = stat_oa
            stats[dataset][config][f"aa_{dataset_type}_statistic"] = stat_aa
            stats[dataset][config][f"oa_{dataset_type}_pvalue"] = pvalue_oa
            stats[dataset][config][f"aa_{dataset_type}_pvalue"] = pvalue_aa

    # Compute significance for Dunnett's test
    if ref_config is not None and test_type == "dunnett":
        for dataset, configs_data in raw_data.items():
            other_configs = [cfg for cfg in configs_data.keys() if cfg != ref_config]
            if not other_configs:
                continue

            for metric_prefix in ["oa", "aa"]:
                metric = f"{metric_prefix}_{dataset_type}"
                try:
                    stats_dunnett, pvalues_dunnett = calculate_dunnett_significance(
                        configs_data, metric, ref_config, other_configs
                    )
                    for i, config in enumerate(other_configs):
                        stats[dataset][config][f"{metric_prefix}_{dataset_type}_statistic"] = stats_dunnett[i]
                        stats[dataset][config][f"{metric_prefix}_{dataset_type}_pvalue"] = pvalues_dunnett[i]
                except KeyError:
                    warnings.warn(
                        f"Skipping Dunnett's test for metric {metric} in dataset {dataset} due to missing data."
                    )
                    continue

    return stats
