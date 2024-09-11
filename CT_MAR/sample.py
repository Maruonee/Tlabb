import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# 파일 위치
file = 'c:\\Users\\tlab\\Desktop\\chest_button_snr_cnr.xlsx'

# 파일 경로 설정
df = pd.read_excel(file)
file_path = os.path.dirname(file)
independent_name = df.columns[2]  # 독립변수
dependent_name = df.columns[3]  # 종속변수
independent_var = df.iloc[:, 2]  # C 행의 값이 독립변수
dependent_var_data = df.iloc[:, 3]  # e 열의 첫 번째 값이 종속변수 이름

# 고유한 독립변수 개수 감지 (중복 제거)
unique_independent_vars = independent_var.unique()

# 동질성 검정
# Levene 검정
levene_stat, levene_pvalue = stats.levene(*[
    dependent_var_data[independent_var == category] 
    for category in unique_independent_vars
])

# 결과 저장
homogeneity_results_df = pd.DataFrame({
    'Test': ['Levene'],
    'Statistic': [levene_stat],
    'p-value': [levene_pvalue]
})
homogeneity_results_df.to_csv(os.path.join(file_path, f'{dependent_name}_homogeneity_test_results.csv'), index=False)

# Welch ANOVA 수행
welch_anova_result = stats.ttest_ind(
    *[dependent_var_data[independent_var == category] for category in unique_independent_vars], 
    equal_var=False  # Welch ANOVA의 핵심 옵션
)

# Welch ANOVA 결과 저장
welch_anova_results_df = pd.DataFrame({
    'F-statistic': [welch_anova_result.statistic],
    'p-value': [welch_anova_result.pvalue]
})
welch_anova_file_path = os.path.join(file_path, f'{dependent_name}_welch_anova_results.csv')
welch_anova_results_df.to_csv(welch_anova_file_path, index=False)

# ANOVA 결과 시각화
data_count = len(dependent_var_data)
height = 6 
width = max(10, data_count / 10)  # 데이터 개수에 비례하여 너비 조정
plt.figure(figsize=(width, height))
df.boxplot(column=dependent_name, by=independent_name, grid=False, showmeans=True)
plt.title(f'{dependent_name}_Boxplot')
plt.suptitle('')
plt.xlabel(independent_name)
plt.ylabel(dependent_name)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(file_path,f'{dependent_name}_boxplot_welch_anova.png'))

plt.show()
