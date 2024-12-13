"""
Siemens의 NMAR(Normalized Metal Artifact Reduction)
금속 아티팩트로 인한 왜곡을 보정하기 위해 복잡한 보간 및 정규화 기술을 사용
Siemens의 NMAR 알고리즘은 여러 각도에서 수집된 데이터를 기반으로 금속 물체 주변의 이미지를 개선하며, 특히 복잡한 해부학적 구조에서의 정확성을 높이기 위해 설계됨

Philips의 OMAR(Orthopedic Metal Artifact Reduction)
Philips는 OMAR 알고리즘을 사용하며, 이는 주로 정형외과적 금속 임플란트 주변의 이미지를 개선하는 데 최적화됨
금속 인공물의 고정된 패턴을 인식하고 제거하는 방법을 사용

GE Healthcare의 Smart MAR(Smart Metal Artifact Reduction)
세 단계의 투영 기반 방식을 적용하여 아티팩트를 효과적으로 줄임
GE의 Smart MAR은 Philips나 Siemens의 NMAR와 달리 투영 데이터에 중점을 두고, CT 영상 복원 이전 단계에서 아티팩트를 처리하는 방식이 특징

====================================================================
metal_segmentation: 이 함수는 주어진 CT 이미지에서 금속 영역을 세그먼트화합니다. 임계값을 사용하여 금속 영역을 감지합니다.
inpainting: 금속 마스크가 적용된 영역을 채우기 위해 OpenCV의 두 가지 보간 방법(텔레아(Telea) 및 나비에-스토크스(Navier-Stokes))를 사용합니다.
nmar_algorithm: 회사별로 NMAR 알고리즘을 적용합니다.
Siemens: 기본 NMAR 알고리즘으로 Gaussian 필터를 사용하여 금속 영역의 이미지를 부드럽게 보정합니다.
Philips: Navier-Stokes 기반 보간 방법을 사용하여 보정 후 부드럽게 처리합니다.
GE: 더 강력한 필터링을 사용하여 GE의 투영 기반 보정을 간략화하여 구현했습니다.
====================================================================
"""
import cv2
import numpy as np
import os

# 폴더설정
folder_path = 'C:\\Users\\tlab\\Desktop\\ct_mar\\opendata\\tissue\\tissue_origin'


def metal_segmentation(ct_image, threshold=2000):
    segmented = ct_image > threshold
    return segmented

def inpainting(ct_image, metal_mask, method='telea'):
    inpainted_image = ct_image.copy()
    inpainted_image[metal_mask] = 0
    if method == 'telea':
        return cv2.inpaint(ct_image, metal_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    elif method == 'ns':
        return cv2.inpaint(ct_image, metal_mask.astype(np.uint8), 3, cv2.INPAINT_NS)
    else:
        raise ValueError("Unsupported inpainting method")

def nmar_algorithm(ct_image, company="siemens"):
    metal_mask = metal_segmentation(ct_image)
    
    if company == "siemens":
        inpainted_image = inpainting(ct_image, metal_mask, method='telea')
        corrected_image = cv2.GaussianBlur(inpainted_image, (5, 5), 1)  
    # elif company == "philips":
    #     inpainted_image = inpainting(ct_image, metal_mask, method='ns')
    #     corrected_image = cv2.GaussianBlur(inpainted_image, (7, 7), 1.5)
    # elif company == "ge":
    #     inpainted_image = inpainting(ct_image, metal_mask, method='telea')
    #     corrected_image = cv2.GaussianBlur(inpainted_image, (9, 9), 2)
    else:
        raise ValueError("Unsupported company")

    return corrected_image

if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            ct_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if ct_image is not None:
                corrected_siemens = nmar_algorithm(ct_image, company="siemens")
                # corrected_philips = nmar_algorithm(ct_image, company="philips")
                # corrected_ge = nmar_algorithm(ct_image, company="ge")

                # Save the corrected images to the same folder
                corrected_siemens_path = os.path.join(folder_path, f'{filename}_nmar.png')
                # corrected_philips_path = os.path.join(folder_path, f'{filename}_nmar_philips.png')
                # corrected_ge_path = os.path.join(folder_path, f'{filename}_nmar_ge.png')
                
                cv2.imwrite(corrected_siemens_path, corrected_siemens)
                # cv2.imwrite(corrected_philips_path, corrected_philips)
                # cv2.imwrite(corrected_ge_path, corrected_ge)
                print(f'{filename}_done.')
            else:
                print(f"Error: Could not load image from {image_path}.")
else:
    print(f"Error: Folder {folder_path} does not exist.")
