import numpy as np

__all__ = [
    "H1_KP",
    "H1_KD",
]

# from https://humanoid4parkour.github.io/resources/humanoid-parkour-learning-paper.pdf (Table 8)
H1_KP_KD = {
    "shoulder_pitch": (30, 1),
    "shoulder_roll": (30, 1),
    "shoulder_yaw": (20, 0.5),
    "elbow": (20, 0.5),
    "torso": (200, 3),
    "hip_yaw": (60, 1.5),
    "hip_roll": (220, 4),
    "hip_pitch": (220, 4),
    "knee": (320, 4),
    "ankle": (40, 2),
}

SHOULDER_PITCH_KP = 30
SHOULDER_PITCH_KD = 1
SHOULDER_ROLL_KP = 30
SHOULDER_ROLL_KD = 1
SHOULDER_YAW_KP = 20
SHOULDER_YAW_KD = 0.5
ELBOW_KP = 20
ELBOW_KD = 0.5
TORSO_KP = 200
TORSO_KD = 3
HIP_YAW_KP = 60
HIP_YAW_KD = 1.5
HIP_ROLL_KP = 220
HIP_ROLL_KD = 4
HIP_PITCH_KP = 220
HIP_PITCH_KD = 4
KNEE_KP = 320
KNEE_KD = 4
ANKLE_KP = 40
ANKLE_KD = 2

H1_KP = np.float32([
    HIP_ROLL_KP, HIP_PITCH_KP, KNEE_KP,
    HIP_ROLL_KP, HIP_PITCH_KP, KNEE_KP,
    TORSO_KP, HIP_YAW_KP, HIP_YAW_KP,
    1,
    ANKLE_KP, ANKLE_KP,
    SHOULDER_PITCH_KP, SHOULDER_ROLL_KP, SHOULDER_YAW_KP, ELBOW_KP,
    SHOULDER_PITCH_KP, SHOULDER_ROLL_KP, SHOULDER_YAW_KP, ELBOW_KP,
])

H1_KD = np.float32([
    HIP_ROLL_KD, HIP_PITCH_KD, KNEE_KD,
    HIP_ROLL_KD, HIP_PITCH_KD, KNEE_KD,
    TORSO_KD, HIP_YAW_KD, HIP_YAW_KD,
    1,
    ANKLE_KD, ANKLE_KD,
    SHOULDER_PITCH_KD, SHOULDER_ROLL_KD, SHOULDER_YAW_KD, ELBOW_KD,
    SHOULDER_PITCH_KD, SHOULDER_ROLL_KD, SHOULDER_YAW_KD, ELBOW_KD,
])
