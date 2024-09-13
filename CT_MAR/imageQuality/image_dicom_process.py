import os
import numpy as np
import pydicom
import csv
from skimage.metrics import structural_similarity

# 폴더내 파일 이름 동일하게 설정 (1:1대칭)
# 원본폴더 - 001.dcm
#         - 002.dcm
################################
# 비교폴더 - 001.dcm
#         - 002.dcm
################################

original_folder = 'C:\\Users\\tlab\\Documents\\ct_metal\\ori'  # 원본 DICOM 파일이 있는 폴더
compare_folder = 'C:\\Users\\tlab\\Documents\\ct_metal\\com'   # 비교할 DICOM 파일이 있는 폴더
processed_folder = 'C:\\Users\\tlab\\Documents\\ct_metal\\proc' # 처리된 DICOM 파일이 있는 폴더

# ROI 설정(실제 분석에 맞게 수정 필요)
signal_roi_coords =  (218, 196, 307, 291) # 예: (x1, y1, x2, y2)
noise_roi_coords =  (7, 26, 90, 105)  # 예: (x1, y1, x2, y2)
            
def calculate_snr(signal_roi, noise_roi):
    signal_mean = np.mean(signal_roi)
    noise_std = np.std(noise_roi)
    if noise_std == 0:
        return np.inf
    else:
        snr = signal_mean / noise_std
        return snr

def calculate_cnr(signal_roi, noise_roi):
    mean_signal = np.mean(signal_roi)
    mean_noise = np.mean(noise_roi)
    noise_std = np.std(noise_roi)
    if noise_std == 0:
        return np.inf
    else:
        cnr = np.abs(mean_signal - mean_noise) / noise_std
        return cnr

def calculate_psnr(original_image, compared_image):
    mse = np.mean((original_image - compared_image) ** 2)
    if mse == 0:
        return np.inf
    max_pixel = np.max(original_image)
    psnr = (max_pixel ** 2) / mse
    return psnr

def calculate_ssim(original_image, compared_image):
    ssim_value = structural_similarity(original_image, compared_image, data_range=compared_image.max() - compared_image.min())
    return ssim_value

def process_files(original_folder, compare_folder, processed_folder,signal_roi_coords,noise_roi_coords):
    original_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.dcm')])
    compare_files = sorted([f for f in os.listdir(compare_folder) if f.endswith('.dcm')])
    processed_files = sorted([f for f in os.listdir(processed_folder) if f.endswith('.dcm')])

    # CSV 파일 경로 설정
    output_csv = os.path.join(original_folder, 'output_results.csv')

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['file_name', 'SNR (Original)', 'SNR (Compare)', 'SNR (Processed)',
                      'CNR (Original)', 'CNR (Compare)', 'CNR (Processed)', 'PSNR (Compare)', 'PSNR (Processed)',
                      'SSIM (Compare)', 'SSIM (Processed)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for orig_file, comp_file, proc_file in zip(original_files, compare_files, processed_files):
            original_path = os.path.join(original_folder, orig_file)
            compare_path = os.path.join(compare_folder, comp_file)
            processed_path = os.path.join(processed_folder, proc_file)

            original_image = pydicom.dcmread(original_path).pixel_array
            compare_image = pydicom.dcmread(compare_path).pixel_array
            processed_image = pydicom.dcmread(processed_path).pixel_array

            signal_roi = original_image[signal_roi_coords[1]:signal_roi_coords[3], signal_roi_coords[0]:signal_roi_coords[2]]
            noise_roi = original_image[noise_roi_coords[1]:noise_roi_coords[3], noise_roi_coords[0]:noise_roi_coords[2]]

            # SNR 계산
            snr_original = calculate_snr(signal_roi, noise_roi)
            snr_compare = calculate_snr(compare_image[signal_roi_coords[1]:signal_roi_coords[3], signal_roi_coords[0]:signal_roi_coords[2]],
                                        compare_image[noise_roi_coords[1]:noise_roi_coords[3], noise_roi_coords[0]:noise_roi_coords[2]])
            snr_processed = calculate_snr(processed_image[signal_roi_coords[1]:signal_roi_coords[3], signal_roi_coords[0]:signal_roi_coords[2]],
                                          processed_image[noise_roi_coords[1]:noise_roi_coords[3], noise_roi_coords[0]:noise_roi_coords[2]])

            # CNR 계산
            cnr_original = calculate_cnr(signal_roi, noise_roi)
            cnr_compare = calculate_cnr(compare_image[signal_roi_coords[1]:signal_roi_coords[3], signal_roi_coords[0]:signal_roi_coords[2]],
                                        compare_image[noise_roi_coords[1]:noise_roi_coords[3], noise_roi_coords[0]:noise_roi_coords[2]])
            cnr_processed = calculate_cnr(processed_image[signal_roi_coords[1]:signal_roi_coords[3], signal_roi_coords[0]:signal_roi_coords[2]],
                                          processed_image[noise_roi_coords[1]:noise_roi_coords[3], noise_roi_coords[0]:noise_roi_coords[2]])

            # PSNR 계산
            psnr_compare = calculate_psnr(original_image, compare_image)
            psnr_processed = calculate_psnr(original_image, processed_image)

            # SSIM 계산
            ssim_compare = calculate_ssim(original_image, compare_image)
            ssim_processed = calculate_ssim(original_image, processed_image)

            # ROI 좌표를 000,000,000,000 형식으로 변환
            signal_roi_str = f"({signal_roi_coords[0]:03},{signal_roi_coords[1]:03},{signal_roi_coords[2]:03},{signal_roi_coords[3]:03})"
            noise_roi_str = f"({noise_roi_coords[0]:03},{noise_roi_coords[1]:03},{noise_roi_coords[2]:03},{noise_roi_coords[3]:03})"

            # 결과 텍스트 작성
            result_text = (f'결과\n신호 ROI 위치 {signal_roi_str}\n'
                           f'노이즈 ROI 위치: {noise_roi_str}\n'
                           f'SNR (원본): {snr_original:.4f}\n'
                           f'SNR (비교): {snr_compare:.4f}\n'
                           f'SNR (처리): {snr_processed:.4f}\n'
                           f'CNR (원본): {cnr_original:.4f}\n'
                           f'CNR (비교): {cnr_compare:.4f}\n' 
                           f'CNR (처리): {cnr_processed:.4f}\n'
                           f'PSNR (비교): {psnr_compare:.4f}\n'
                           f'PSNR (처리): {psnr_processed:.4f}\n'
                           f'SSIM (비교): {ssim_compare:.4f}\n'
                           f'SSIM (처리): {ssim_processed:.4f}\n')
            # 결과 파일 저장
            result_filename = os.path.splitext(orig_file)[0] + '.txt'
            result_filepath = os.path.join(original_folder, result_filename)
            with open(result_filepath, 'w') as result_file:
                result_file.write(result_text)
                
            # CSV에 결과 추가
            writer.writerow({
                'file_name': orig_file,
                'SNR (Original)': f'{snr_original:.4f}',
                'SNR (Compare)': f'{snr_compare:.4f}',
                'SNR (Processed)': f'{snr_processed:.4f}',
                'CNR (Original)': f'{cnr_original:.4f}',
                'CNR (Compare)': f'{cnr_compare:.4f}',
                'CNR (Processed)': f'{cnr_processed:.4f}',
                'PSNR (Compare)': f'{psnr_compare:.4f}',
                'PSNR (Processed)': f'{psnr_processed:.4f}',
                'SSIM (Compare)': f'{ssim_compare:.4f}',
                'SSIM (Processed)': f'{ssim_processed:.4f}'
            })

            print(f'결과 저장됨: {result_filepath}')

    print(f'CSV 파일 저장됨: {output_csv}')
# 파일 처리
process_files(original_folder, compare_folder, processed_folder, signal_roi_coords, noise_roi_coords)
