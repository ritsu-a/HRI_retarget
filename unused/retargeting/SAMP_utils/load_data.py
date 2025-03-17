import smplx
import torch
import pickle
from SAMP_utils.denoise_mocap import denoise_samp


def load_smplx_motion(samp_data_pkl_path, smplx_model_dir, denoise=False, start_frame=0, end_frame=-1, sampling_rate=3):
    """
    SAMP original frame rate: 30 FPS
    """

    with open(samp_data_pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
        betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1,10)
        full_trans = torch.tensor( data['pose_est_trans'], dtype=torch.float32)

        human_motion = {
            "full_poses": full_poses,
            "betas": betas,
            "full_trans": full_trans,
        }
        if denoise:
            human_motion = denoise_samp(human_motion)
    
    N_frame = human_motion["full_poses"].shape[0]
    if end_frame == -1:
        end_frame = N_frame
    body_model = smplx.create(model_path=smplx_model_dir, model_type='smplx', gender="male", use_pca=False, batch_size=N_frame)

    global_orient = human_motion["full_poses"][:, 0:3]  # (N_frame, 3)
    body_pose = human_motion["full_poses"][:, 3:66]  # (N_frame, 63)
    transl = human_motion["full_trans"]  # (N_frame, 3)
    output = body_model(global_orient=global_orient, body_pose=body_pose, betas=human_motion["betas"], transl=transl, return_verts=True)

    joint_positions = output.joints.detach().cpu().numpy()  # (N_frame, 127, 3)
    assert joint_positions.shape == (N_frame, 127, 3)

    result = {
        "joint_positions": joint_positions[start_frame:end_frame:sampling_rate],
    }
    return result
