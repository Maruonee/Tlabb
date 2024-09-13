import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# 파일 위치
file = 'c:\\Users\\tlab\\Desktop\\abd.xlsx'

# 파일 경로 설정
df = pd.read_excel(file)
file_path = os.path.dirname(file)
independent_name = df.columns[2]
dependent_name = df.columns[3]
independent_var = df.iloc[:, 2]  # 독립변수
dependent_var_data = df.iloc[:, 3]  # 종속변수

# 고유한 독립변수 감지 (중복 제거)
unique_independent_vars = independent_var.unique()

# 맨-휘트니 U 검정
mannwhitney_results = []
for i, group1 in enumerate(unique_independent_vars):
    for group2 in unique_independent_vars[i+1:]:
        group1_data = dependent_var_data[independent_var == group1]
        group2_data = dependent_var_data[independent_var == group2]
        stat, pval = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        mannwhitney_results.append([group1, group2, stat, pval])

# 결과 저장
mannwhitney_df = pd.DataFrame(mannwhitney_results, columns=['Group 1', 'Group 2', 'U Statistic', 'p-value'])
mannwhitney_df.to_csv(os.path.join(file_path, f'{dependent_name}_mannwhitney_results.csv'), index=False)

# 맨-휘트니 U 검정 시각화
mannwhitney_plot_data = mannwhitney_df[['Group 1', 'Group 2', 'p-value']].copy()
mannwhitney_plot_data['Comparison'] = mannwhitney_plot_data['Group 1'] + ' vs ' + mannwhitney_plot_data['Group 2']

# 그래프 그리기
plt.figure(figsize=(10, 6))
bars = plt.barh(mannwhitney_plot_data['Comparison'], mannwhitney_plot_data['p-value'], color='skyblue')

# 0.05 기준선을 그리고 해당 값을 명확하게 표기
plt.axvline(x=0.05, color='red', linestyle='--', label='0.05 Significance Level')
plt.text(0.05, len(mannwhitney_plot_data) - 0.5, 'p = 0.05', color='red', va='center')
plt.xlabel('p-value')
plt.ylabel('Group Comparison')
plt.title(f'Mann-Whitney U Test - {dependent_name}')
plt.legend()
plt.grid(True)

# 그래프 저장
plt.savefig(os.path.join(file_path, f'{dependent_name}_mannwhitney_plot.png'))

# 동질성 검정 (Levene, Bartlett)
levene_stat, levene_pvalue = stats.levene(*[dependent_var_data[independent_var == category] for category in unique_independent_vars])
bartlett_stat, bartlett_pvalue = stats.bartlett(*[dependent_var_data[independent_var == category] for category in unique_independent_vars])

# 결과 저장
homogeneity_results_df = pd.DataFrame({
    'Test': ['Levene', 'Bartlett'],
    'Statistic': [levene_stat, bartlett_stat],
    'p-value': [levene_pvalue, bartlett_pvalue]
})
homogeneity_results_df.to_csv(os.path.join(file_path, f'{dependent_name}_homogeneity_test_results.csv'), index=False)
