## Kinematic Retargeting from liuyun

The code is developed on PyTorch, and it is decoupled with IssacGym.

The forward kinematics (FK) is computed by ```pytorch_kinematics```.

### Environment Setup

Put SMPLX models under the folder ```SAMP_utils/models/smplx```, the file structure should be:

```
|-- ./SAMP_utils/models/smplx
    |-- SMPLX_FEMALE.npz
    |-- SMPLX_FEMALE.pkl
    |-- SMPLX_MALE.npz
    |-- SMPLX_MALE.pkl
    |-- SMPLX_NEUTRAL.npz
    |-- SMPLX_NEUTRAL.pkl
```

### Run Retargeting

(1) SMPLX -> H1

```python retarget_smplx_to_h1.py```

It will save the results to ```./pred.npz```, including 19 DoF joint angles, 3 DoF root global rotation, and 3 DoF root global translation for each frame.

(2) SMPL -> H1

TBD

### Visualize the Results

```
cd zhikai_code
python replay.py
```

This will replay the kinematic trajectory from ```./pred.npz``` in IssacGym.

### Contact

Please contact liuyun.
