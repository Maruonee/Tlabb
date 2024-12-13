import os
import pydicom
import pandas as pd

# DICOM 파일이 있는 폴더 경로 설정
input_folder = 'C:\\Users\\tlab\\Desktop\\cobb_dicom'
output_excel = 'C:\\Users\\tlab\\Desktop\\dicom_patient_info.xlsx'

# 데이터를 저장할 리스트 초기화
data = []

# A 폴더의 모든 DICOM 파일을 순회하며 나이와 성별 추출
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.dcm'):
        # DICOM 파일 경로 지정
        dicom_path = os.path.join(input_folder, filename)
        
        # DICOM 파일 읽기
        dicom_data = pydicom.dcmread(dicom_path)
        
        # 환자 정보 추출
        try:
            # 나이와 성별 정보를 DICOM 태그에서 추출
            patient_age = dicom_data.PatientAge if 'PatientAge' in dicom_data else '정보 없음'
            patient_gender = dicom_data.PatientSex if 'PatientSex' in dicom_data else '정보 없음'
            
            # 추출한 정보를 리스트에 추가
            data.append({
                '파일명': filename,
                '나이': patient_age,
                '성별': patient_gender
            })
        
        except Exception as e:
            print(f"{filename}에서 정보를 추출하는 중 오류 발생: {e}")

# 추출한 정보를 DataFrame으로 변환
df = pd.DataFrame(data)

# DataFrame을 엑셀 파일로 저장
df.to_excel(output_excel, index=False)
print(f"모든 DICOM 파일에서 나이와 성별 정보가 엑셀 파일로 저장되었습니다: {output_excel}")
