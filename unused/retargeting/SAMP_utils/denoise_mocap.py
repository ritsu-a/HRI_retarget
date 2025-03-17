import smplx
import torch
import pickle


def denoise_samp(human_motion):

    denoised_full_poses = human_motion["full_poses"].clone()
    denoised_betas = human_motion["betas"].clone()
    denoised_full_trans = human_motion["full_trans"].clone()

    # denoise
    N_frame = denoised_full_poses.shape[0]
    v_trans = denoised_full_trans[1:] - denoised_full_trans[:-1]
    v_global_rots = denoised_full_poses[1:, 0:3] - denoised_full_poses[:-1, 0:3]
    v_local_rots = denoised_full_poses[1:, 3:66] - denoised_full_poses[:-1, 3:66]
    # for i in range(1560, 1570):
    #     print(i, ((v_trans[i]**2).sum()**0.5).item(), ((v_global_rots[i]**2).sum()**0.5).item(), ((v_local_rots[i]**2).sum()**0.5).item())
    #     print(i, denoised_full_trans[i], denoised_full_poses[i])
    # NOTE: no need for denoising

    denoised_human_motion = {
        "full_poses": denoised_full_poses,
        "betas": denoised_betas,
        "full_trans": denoised_full_trans,
    }
    return denoised_human_motion
