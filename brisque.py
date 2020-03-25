import numpy as np
import cv2
from scipy.special import gamma



def estimate_GGD_parameters(vec):
    gam = np.arange(0.2, 10.0, 0.001)  # 产生候选的γ
    r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam))**2)  # 根据候选的γ计算r(γ)
    sigma_sq = np.mean((vec)**2)
    sigma = np.sqrt(sigma_sq)  # 方差估计
    E = np.mean(np.abs(vec))
    r = sigma_sq/(E**2)  # 根据sigma和E计算r(γ)
    diff = np.abs(r-r_gam)
    gamma_param = gam[np.argmin(diff, axis=0)]
    return [gamma_param, sigma_sq]


def estimate_AGGD_parameters(vec):
    alpha = np.arange(0.2, 10.0, 0.001)  # 产生候选的α
    r_alpha = ((gamma(2/alpha))**2)/(gamma(1/alpha)
                                     * gamma(3/alpha))  # 根据候选的γ计算r(α)
    sigma_l = np.sqrt(np.mean(vec[vec < 0]**2))
    sigma_r = np.sqrt(np.mean(vec[vec > 0]**2))
    gamma_ = sigma_l/sigma_r
    u2 = np.mean(vec**2)
    m1 = np.mean(np.abs(vec))
    r_ = m1**2/u2
    R_ = r_*(gamma_**3+1)*(gamma_+1)/((gamma_**2+1)**2)
    diff = (R_-r_alpha)**2
    alpha_param = alpha[np.argmin(diff, axis=0)]
    const1 = np.sqrt(gamma(1 / alpha_param) / gamma(3 / alpha_param))
    const2 = gamma(2 / alpha_param) / gamma(1 / alpha_param)
    eta = (sigma_r-sigma_l)*const1*const2
    return [alpha_param, eta, sigma_l**2, sigma_r**2]



def calculate_mscn(dis_image):
    dis_image = dis_image.astype(np.float32)  # 类型转换十分重要
    ux = cv2.GaussianBlur(dis_image, (7, 7), 7/6)
    ux_sq = ux*ux
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image**2, (7, 7), 7/6)-ux_sq))

    mscn = (dis_image-ux)/(1+sigma)

    return mscn



def brisque_feature(dis_image):
    mscn = calculate_mscn(dis_image)
    f1_2 = estimate_GGD_parameters(mscn)
    H = mscn*np.roll(mscn, 1, axis=1)
    V = mscn*np.roll(mscn, 1, axis=0)
    D1 = mscn*np.roll(np.roll(mscn, 1, axis=1), 1, axis=0)
    D2 = mscn*np.roll(np.roll(mscn, -1, axis=1), -1, axis=0)
    f3_6 = estimate_AGGD_parameters(H)
    f7_10 = estimate_AGGD_parameters(V)
    f11_14 = estimate_AGGD_parameters(D1)
    f15_18 = estimate_AGGD_parameters(D2)
    return f1_2+f3_6+f7_10+f11_14+f15_18


