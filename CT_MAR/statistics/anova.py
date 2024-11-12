import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc
from statsmodels.formula.api import ols
from itertools import combinations
import matplotlib.pyplot as plt
import os

#파일위치
file = 'c:\\Users\\tlab\\Desktop\\SSIM.xlsx'

#파일경로설정
df = pd.read_excel(file)
file_path = os.path.dirname(file)
independent_name = df.columns[2]
dependent_name = df.columns[4]
independent_var = df.iloc[:, 2]  # C 행의 값이 독립변수
dependent_var_data = df.iloc[:, 4]  # e 열의 첫 번째 값이 종속변수 이름

# 고유한 독립변수 개수 감지 (중복 제거)
unique_independent_vars = independent_var.unique()

# 동질성 검정
# Levene 검정
levene_stat, levene_pvalue = stats.levene(*[
    dependent_var_data[independent_var == category] 
    for category in unique_independent_vars
])
# Bartlett 검정
bartlett_stat, bartlett_pvalue = stats.bartlett(*[
    dependent_var_data[independent_var == category] 
    for category in unique_independent_vars
])

# 결과 저장
homogeneity_results_df = pd.DataFrame({
    'Test': ['Levene', 'Bartlett'],
    'Statistic': [levene_stat, bartlett_stat],
    'p-value': [levene_pvalue, bartlett_pvalue]
})
homogeneity_results_df.to_csv(os.path.join(file_path, f'{dependent_name}_homogeneity_test_results.csv'), index=False)

df_anova = pd.DataFrame({
    independent_name: independent_var,
    dependent_name: dependent_var_data
})

# 독립 변수를 범주형으로 명시적으로 변환
df_anova[independent_name] = df_anova[independent_name].astype('category')

# 결측치 제거 (선택적)
df_anova.dropna(subset=[independent_name, dependent_name], inplace=True)

# ANOVA 모델 생성 및 수행
model = ols(f'{dependent_name} ~ C({independent_name})', data=df_anova).fit()

# ANOVA 검정 - typ=3으로 수정
anova_table = sm.stats.anova_lm(model, typ=3)
pd.set_option('display.float_format', lambda x: '%.200f' % x)

# ANOVA 테이블 결과를 CSV 파일로 저장
anova_file_path = os.path.join(file_path, f'{dependent_name}_anova_results.csv')
anova_table.to_csv(anova_file_path, float_format='%.200f')

# ANOVA 결과 출력
print(anova_table)
# ANOVA 결과
data_count = len(dependent_var_data)
height = 6 
width = max(10, data_count / 10)  # 데이터 개수에 비례하여 너비 조정
plt.figure(figsize=(width, height))
df_anova.boxplot(column=dependent_name, by=independent_name, grid=False, showmeans=True)
plt.title(f'{dependent_name}_Boxplot')
plt.suptitle('')
plt.xlabel(independent_name)
plt.ylabel(dependent_name)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(file_path,f'{dependent_name}_boxplot.png'))

# Tukey HSD
tukey_data = mc.pairwise_tukeyhsd(dependent_var_data, independent_var, alpha=0.05)
tukey_cnr_summary = tukey_data.summary()
tukey_df = pd.DataFrame(tukey_cnr_summary.data[1:], columns=tukey_cnr_summary.data[0])
tukey_df.to_csv(os.path.join(file_path,f'{dependent_name}_tukey_hsd_results.csv'), index=False)

# Tukey HSD 시각화
plt.figure(figsize=(width, height))
tukey_data.plot_simultaneous()  # comparison_name 제거
plt.title(f'Tukey HSD - {dependent_name}')
plt.grid(True)
plt.savefig(os.path.join(file_path, f'{dependent_name}_tukey_hsd_plot.png'))

# Bonferroni
comp = mc.MultiComparison(dependent_var_data, independent_var)
bonferroni_data = comp.allpairtest(stats.ttest_ind, method='bonf')[0]
bonferroni_data_results = bonferroni_data
bonferroni_df = pd.DataFrame(bonferroni_data_results.data[1:], columns=bonferroni_data_results.data[0])
bonferroni_df.to_csv(os.path.join(file_path,f'{dependent_name}_bonferroni_results.csv'), index=False)

# Bonferroni 검정 시각화
bonferroni_plot_data = bonferroni_df[['group1', 'group2', 'pval']].copy()
bonferroni_plot_data['comparison'] = bonferroni_plot_data['group1'] + ' vs ' + bonferroni_plot_data['group2']

plt.figure(figsize=(width, height))
plt.barh(bonferroni_plot_data['comparison'], bonferroni_plot_data['pval'], color='skyblue')
plt.axvline(x=0.05, color='red', linestyle='--')  # 유의수준 0.05 기준선
plt.xlabel('p-value')
plt.ylabel('Group Comparison')
plt.title(f'Bonferroni - {dependent_name}')
plt.grid(True)
plt.savefig(os.path.join(file_path, f'{dependent_name}_bonferroni_plot.png'))

# Scheffé method
def scheffe_posthoc(data, group_col, value_col):
    groups = data[group_col].unique()
    group_combinations = list(combinations(groups, 2))
    results = []
    
    for group1, group2 in group_combinations:
        group1_data = data[data[group_col] == group1][value_col]
        group2_data = data[data[group_col] == group2][value_col]
        stat, pval = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        results.append([group1, group2, stat, pval])
    
    return pd.DataFrame(results, columns=['Group 1', 'Group 2', 't-statistic', 'p-value'])

# Applying Scheffé
scheffe_results = scheffe_posthoc(df, independent_name, dependent_name)
scheffe_results.to_csv(os.path.join(file_path,f'{dependent_name}_scheffe_results.csv'), index=False)