import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 'Mode' 값을 활동 이름으로 매핑
mode_names = {
    1: "01_walk",
    2: "02_run",
    3: "03_Jump",
    4: "04_sit",
    5: "05_lie",
    6: "06_upstairs",
    7: "07_downstairs",
    8: "08_up_ramp",
    9: "09_down_ramp",
    10: "10_sit_chair",
    11: "11_sit_up",
}

base_folder_path = 'C:\\Users\\tlab\Desktop\\낙상실험데이터정리\\데이터\\edit\\' 

#폴더별 받기
folder_names = ['김건남',
                '김길자',
                '김정순',
                '서순복',
                '서정승',
                '여신자',
                '오정순',
                '이재빈',
                '장근자',
                '조갑남',
                '조정희',
                '조진숙',
                '한화조',
                '허봉자',
                '황옥금']
#or glob으로 받아도 됨
# folder_names = [item for item in os.listdir(base_folder_path) 
                    # if os.path.isdir(os.path.join(base_folder_path, item))]
                    
for folder_name in folder_names:
    folder_path = os.path.join(base_folder_path, folder_name)
    print(f"{folder_path}폴더 시작")
    xlsx_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    for file in xlsx_files:
        df = pd.read_excel(file)
        
        # 모드별 엑셀 저장
        for mode, activity_name in mode_names.items():
            mode_df = df[df['Mode'] == mode]

            if not mode_df.empty:
                activity_prefix = activity_name.replace(" ", "_")
                new_file_name = f"{os.path.splitext(os.path.basename(file))[0]}_{activity_prefix}.xlsx"
                new_file_path = os.path.join(folder_path, new_file_name) #엑셀파일 이름
                print(f"{new_file_path} 저장")
                mode_df.to_excel(new_file_path, index=False)

                # 각 열에 대해 그래프
                for column in mode_df.columns[1:]:
                    plt.figure(figsize=(10, 6))
                    plt.plot(mode_df[column])
                    plt.title(f"{activity_name} - {column}")
                    plt.xlabel('Index')
                    plt.ylabel('Value')
                    graph_path = os.path.join(folder_path, f"{os.path.splitext(os.path.basename(file))[0]}_{activity_prefix}_{column}.png") #그래프 이름
                    plt.savefig(graph_path)
                    plt.close()

                    print(f"{graph_path} 저장")
    print(f" {folder_path}폴더 완료")