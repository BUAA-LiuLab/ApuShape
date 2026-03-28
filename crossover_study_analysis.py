"""
Two-way Fixed Effects Model for User Study Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def load_data(csv_path):
    """Load CSV data file."""
    return pd.read_csv(csv_path)


def run_two_way_fe(df):
    """
    Two-way fixed effects model: T ~ condition + phase + annotator + image
    Returns regression coefficients and statistics.
    """
    model = smf.ols(
        'correction_time_sec ~ C(condition) + phase + C(annotator_id) + C(image_id)',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['annotator_id']})

    cond_key = [k for k in model.params.index if k.startswith('C(condition)')][0]
    beta1 = model.params[cond_key]
    beta1_p = model.pvalues[cond_key]
    beta1_ci = model.conf_int().loc[cond_key]

    return {
        'coefficient': beta1,
        'p_value': beta1_p,
        'ci_lower': beta1_ci[0],
        'ci_upper': beta1_ci[1],
        'model': model
    }


def paired_ttest(df):
    """Paired t-test between ShapeActive and No-assist conditions."""
    paired = df.pivot_table(
        index=['annotator_id', 'image_id'],
        columns='condition',
        values='correction_time_sec'
    )
    diff = paired['ShapeActive'] - paired['No-assist']
    diff = diff.dropna()

    t_stat, p_val = stats.ttest_1samp(diff, popmean=0)
    mean_diff = diff.mean()
    se = diff.std() / np.sqrt(len(diff))
    ci = stats.t.interval(0.95, len(diff) - 1, loc=mean_diff, scale=se)

    return {
        'mean_diff': mean_diff,
        't_statistic': t_stat,
        'p_value': p_val,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_pairs': len(diff)
    }


