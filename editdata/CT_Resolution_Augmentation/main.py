import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
import imageio
import scipy, scipy.misc, scipy.signal
import sys
from skimage import exposure as ex
import PIL
from PIL import Image

def CreateDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def ImageProcess(Path):
    Root_Dir = Path
    Img_Path_List = []
    Possible_Img_Extension = ['.png']
    for (Root, Dirs, Files) in os.walk(Root_Dir):
        if len(Files) > 0:
            for File_Name in Files:
                if os.path.splitext(File_Name)[1] in Possible_Img_Extension:
                    Img_Path = Root + '/' + File_Name
                    Img_Path = Img_Path.replace('\\', '/')
                    Img_Path_List.append(Img_Path)
                    print(Img_Path)

                    # Erosion & Dilation
                    Erosion_Folder_Path = Root + '/Erosion'
                    Dilation_Folder_Path = Root + '/Dilation'
                    CreateDirectory(Erosion_Folder_Path)
                    CreateDirectory(Dilation_Folder_Path)
                    
                    for i in range(2):
                        Kernel_Size = i * 2 + 3
                        print(f"Kernel_Size = {Kernel_Size}")
                        Erosion_Image = Erosion_Process(Img_Path, Kernel_Size)
                        Dilation_Image = Dilation_Process(Img_Path, Kernel_Size)
                        Erosion_Image_Path = Root + '/Erosion/Erosion_' + str(Kernel_Size) + "_" + File_Name
                        Dilation_Image_Path = Root + '/Dilation/Dilation_' + str(Kernel_Size) + "_" + File_Name
                        cv2.imwrite(Erosion_Image_Path, Erosion_Image)
                        cv2.imwrite(Dilation_Image_Path, Dilation_Image)

                    # Histogram Equalization (he)
                    he_Folder_Path = Root + '/he'
                    CreateDirectory(he_Folder_Path)
                    Input_he_Image = imageio.imread(Img_Path)
                    he_Image = he(Input_he_Image)
                    he_Image_Path = Root + '/he/he_' + File_Name
                    plt.imsave(he_Image_Path,he_Image)
                    
                    # Dynamic Histogram Equalization (dhe)
                    dhe_Folder_Path = Root + '/dhe'
                    CreateDirectory(dhe_Folder_Path)
                    Input_dhe_Image = imageio.imread(Img_Path)
                    dhe_Image = dhe(Input_dhe_Image)
                    dhe_Image_Path = Root + '/dhe/dhe_' + File_Name
                    plt.imsave(dhe_Image_Path,dhe_Image)
                    
                    # A New Image Contrast Enhancement Algorithm Using Exposure Fusion Framework (ying)
                    ying_Folder_Path = Root + '/ying'
                    CreateDirectory(ying_Folder_Path)
                    Input_ying_Image = imageio.imread(Img_Path)
                    ying_Image = Ying_2017_CAIP(Input_ying_Image)
                    ying_Image_Path = Root + '/ying/ying_' + File_Name
                    plt.imsave(ying_Image_Path,ying_Image)

                    # Blur
                    Blur_Folder_Path = Root + '/Blur'
                    CreateDirectory(Blur_Folder_Path)
                    
                    for i in range(2):
                        Kernel_Size = i * 2 + 3
                        print(f"Kernel_Size = {Kernel_Size}")
                        Blur_Image = Blur_Process(Img_Path, Kernel_Size)
                        Blur_Image_Path = Root + '/Blur/Blur_' + str(Kernel_Size) + "_" + File_Name
                        cv2.imwrite(Blur_Image_Path, Blur_Image)

                    # GaussianBlur
                    GaussianBlur_Folder_Path = Root + '/GaussianBlur'
                    CreateDirectory(GaussianBlur_Folder_Path)
                    
                    for i in range(2):
                        for j in range(3):
                            Kernel_Size = i * 2 + 3
                            Sigma_Size = j
                            print(f"Kernel_Size = {Kernel_Size}")
                            print(f"Sigma_Size = {Sigma_Size}")
                            GaussianBlur_Image = GaussianBlur_Process(Img_Path, Kernel_Size, Sigma_Size)
                            GaussianBlur_Image_Path = Root + '/GaussianBlur/GaussianBlur_' + str(Kernel_Size) + "_" + str(Sigma_Size) + "_" + File_Name
                            cv2.imwrite(GaussianBlur_Image_Path, GaussianBlur_Image)

                    # MedianBlur
                    MedianBlur_Folder_Path = Root + '/MedianBlur'
                    CreateDirectory(MedianBlur_Folder_Path)
                    
                    for i in range(2):
                        Kernel_Size = i * 2 + 3
                        print(f"Kernel_Size = {Kernel_Size}")
                        MedianBlur_Image = MedianBlur_Process(Img_Path, Kernel_Size)
                        MedianBlur_Image_Path = Root + '/MedianBlur/MedianBlur_' + str(Kernel_Size) + "_" + File_Name
                        cv2.imwrite(MedianBlur_Image_Path, MedianBlur_Image)

                    # BilateralFilter
                    BilateralFilter_Folder_Path = Root + '/BilateralFilter'
                    CreateDirectory(BilateralFilter_Folder_Path)
                    
                    for i in range(2):
                        for j in range(15):
                            Kernel_Size = i * 2 + 3
                            SigmaColor_Size = j * 10 + 10
                            SigmaSpac_Size = j * 10 + 10
                            print(f"Kernel_Size = {Kernel_Size}")
                            print(f"SigmaColor_Size = {SigmaColor_Size}")
                            print(f"SigmaSpac_Size = {SigmaSpac_Size}")
                            BilateralFilter_Image = BilateralFilter_Process(Img_Path, Kernel_Size, SigmaColor_Size, SigmaSpac_Size)
                            BilateralFilter_Image_Path = Root + '/BilateralFilter/BilateralFilter_' + str(Kernel_Size) + "_" + str(SigmaColor_Size) + "_" + str(SigmaSpac_Size) + "_" + File_Name
                            cv2.imwrite(BilateralFilter_Image_Path, BilateralFilter_Image)
                        

def Erosion_Process(Path, Kernel_Size):
    Input_Image = cv2.imread(Path)
    Kernel = np.ones((Kernel_Size,Kernel_Size), np.uint8)
    Erosion_Image = cv2.erode(Input_Image, Kernel)
    return(Erosion_Image)

def Dilation_Process(Path, Kernel_Size):
    Input_Image = cv2.imread(Path)
    Kernel = np.ones((Kernel_Size,Kernel_Size), np.uint8)
    Dilation_Image = cv2.dilate(Input_Image, Kernel)
    return(Dilation_Image)

def build_is_hist(img):
    hei = img.shape[0]
    wid = img.shape[1]
    ch = img.shape[2]
    Img = np.zeros((hei+4, wid+4, ch))
    for i in range(ch):
        Img[:,:,i] = np.pad(img[:,:,i], (2,2), 'edge')
    hsv = (matplotlib.colors.rgb_to_hsv(Img))
    hsv[:,:,0] = hsv[:,:,0] * 255
    hsv[:,:,1] = hsv[:,:,1] * 255
    hsv[hsv>255] = 255
    hsv[hsv<0] = 0
    hsv = hsv.astype(np.uint8).astype(np.float64)
    fh = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
    fv = fh.conj().T
    
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    I = hsv[:,:,2]

    dIh = scipy.signal.convolve2d(I, np.rot90(fh, 2), mode='same')
    dIv = scipy.signal.convolve2d(I, np.rot90(fv, 2), mode='same')
    dIh[dIh==0] = 0.00001
    dIv[dIv==0] = 0.00001
    dI = np.sqrt(dIh**2+dIv**2).astype(np.uint32)
    di = dI[2:hei+2,2:wid+2]
    
    dSh = scipy.signal.convolve2d(S, np.rot90(fh, 2), mode='same')
    dSv = scipy.signal.convolve2d(S, np.rot90(fv, 2), mode='same')
    dSh[dSh==0] = 0.00001
    dSv[dSv==0] = 0.00001
    dS = np.sqrt(dSh**2+dSv**2).astype(np.uint32)
    ds = dS[2:hei+2,2:wid+2]

    
    h = H[2:hei+2,2:wid+2]
    s = S[2:hei+2,2:wid+2]
    i = I[2:hei+2,2:wid+2].astype(np.uint8)
    
    Imean = scipy.signal.convolve2d(I,np.ones((5,5))/25, mode='same')
    Smean = scipy.signal.convolve2d(S,np.ones((5,5))/25, mode='same')
    
    Rho = np.zeros((hei+4,wid+4))
    for p in range(2,hei+2):
        for q in range(2,wid+2):
            tmpi = I[p-2:p+3,q-2:q+3]
            tmps = S[p-2:p+3,q-2:q+3]
            corre = np.corrcoef(tmpi.flatten('F'),tmps.flatten('F'))
            Rho[p,q] = corre[0,1]
    
    rho = np.abs(Rho[2:hei+2,2:wid+2])
    rho[np.isnan(rho)] = 0
    rd = (rho*ds).astype(np.uint32)
    Hist_I = np.zeros((256,1))
    Hist_S = np.zeros((256,1))
    
    for n in range(0,255):
        temp = np.zeros(di.shape)
        temp[i==n] = di[i==n]
        Hist_I[n+1] = np.sum(temp.flatten('F'))
        temp = np.zeros(di.shape)
        temp[i==n] = rd[i==n]
        Hist_S[n+1] = np.sum(temp.flatten('F'))

    return Hist_I, Hist_S

def dhe(img, alpha=0.5):
    
    hist_i, hist_s = build_is_hist(img)
    hist_c = alpha*hist_s + (1-alpha)*hist_i
    hist_sum = np.sum(hist_c)
    hist_cum = hist_c.cumsum(axis=0)
    
    hsv = matplotlib.colors.rgb_to_hsv(img)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    i = hsv[:,:,2].astype(np.uint8)
    
    c = hist_cum / hist_sum
    s_r = (c * 255)
    i_s = np.zeros(i.shape)
    for n in range(0,255):
        i_s[i==n] = s_r[n+1]/255.0
    i_s[i==255] = 1
    hsi_o = np.stack((h,s,i_s), axis=2)
    result = matplotlib.colors.hsv_to_rgb(hsi_o)
    
    result = result * 255
    result[result>255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def he(img):
    if(len(img.shape)==2):      #gray
        outImg = ex.equalize_hist(img[:,:])*255 
    elif(len(img.shape)==3):    #RGB
        outImg = np.zeros((img.shape[0],img.shape[1],3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

    outImg[outImg>255] = 255
    outImg[outImg<0] = 0
    return outImg.astype(np.uint8)

def computeTextureWeights(fin, sigma, sharpness):
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')

    W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    return  W_h, W_v
    
def solveLinearEquation(IN, wx, wy, lamda):
    [r, c] = IN.shape
    k = r * c
    dx =  -lamda * wx.flatten('F')
    dy =  -lamda * wy.flatten('F')
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda *tempx.flatten('F')
    dya = -lamda *tempy.flatten('F')
    tmp = wx[:,-1]
    tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
    tmp = wy[-1,:]
    tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')
    
    wx[:,-1] = 0
    wy[-1,:] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')
    
    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
    D = 1 - ( dx + dy + dxa + dya)
    A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T
    
    tin = IN[:,:]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')
    
    return OUT
    

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S

def rgb2gm(I):
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = np.abs((I[:,:,0]*I[:,:,1]*I[:,:,2]))**(1/3)

    return I

def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma)*beta
    return J

def entropy(X):
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp<0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0*pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Esatimate k
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    tmp[tmp<0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)
    
    isBad = isBad * 1
    isBad = scipy.misc.imresize(isBad, (50,50), interp='bicubic', mode='F')
    isBad[isBad<0.5] = 0
    isBad[isBad>=0.5] = 1
    Y = Y[isBad==1]
    
    if Y.size == 0:
       J = I
       return J
    
    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)
    
    # Apply k
    J = applyK(I, opt_k, a, b) - 0.01
    return J
    

def Ying_2017_CAIP(img, mu=0.5, a=-0.3293, b=1.1258):
    lamda = 0.5
    sigma = 5
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Weight matrix estimation
    t_b = np.max(I, axis=2)
    t_our = cv2.resize(tsmooth(scipy.misc.imresize(t_b, 0.5, interp='bicubic', mode='F'), lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)
    #t_our = t_b
    # Apply camera model with k(exposure ratio)
    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad)

    # W: Weight Matrix
    t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
    for i in range(I.shape[2]):
        t[:,:,i] = t_our
    W = t**mu

    I2 = I*W
    J2 = J*(1-W)

    result = I2 + J2
    result = result * 255
    result[result > 255] = 255
    result[result<0] = 0
    return result.astype(np.uint8)

def Blur_Process(Path, Kernel_Size):
    Input_Image = cv2.imread(Path)
    Kernel = Kernel_Size
    Blur_Image = cv2.blur(Input_Image, (Kernel, Kernel))
    return(Blur_Image)

def GaussianBlur_Process(Path, Kernel_Size, Sigma_Size):
    Input_Image = cv2.imread(Path)
    Kernel = Kernel_Size
    Sigma = Sigma_Size
    GaussianBlur_Image = cv2.GaussianBlur(Input_Image, (Kernel, Kernel), Sigma)
    return(GaussianBlur_Image)

def MedianBlur_Process(Path, Kernel_Size):
    Input_Image = cv2.imread(Path)
    Kernel = Kernel_Size
    MedianBlur_Image = cv2.medianBlur(Input_Image, Kernel)
    return(MedianBlur_Image)

def BilateralFilter_Process(Path, Kernel_Size, SigmaColor_Size, SigmaSpac_Size):
    Input_Image = cv2.imread(Path)
    Kernel = Kernel_Size
    SigmaColor = SigmaColor_Size
    SigmaSpac = SigmaSpac_Size
    BilateralFilter_Image = cv2.bilateralFilter(Input_Image, Kernel, SigmaColor, SigmaSpac)
    return(BilateralFilter_Image)

################## MAIN ##################
Pass_Path = 'E:/OPEN DATASET/Eulgi Dataset/Spatial Resolution/PNG_pass'
Nonpass_Path = 'E:/OPEN DATASET/Eulgi Dataset/Spatial Resolution/PNG_nonpass'

ImageProcess(Pass_Path)
ImageProcess(Nonpass_Path)
################## END ##################
