import time
from dataclasses import dataclass, fields

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from cyclonedds.idl import IdlStruct

from real_robot_utils.driver.dualsense.dualsense_controller import (
    DualSenseLocomotionController, CommandCfg, DualSenseLocomotionCommand, ContinuousState, ControlMode)


from real_h1_walk_config import RealH1WalkConfig


@dataclass
class LocomotionCommand(IdlStruct, typename="LocomotionCommand"):
    state_start: bool
    state_terminate: bool
    state_move: bool
    state_work: bool
    x_lin_vel: float
    y_lin_vel: float
    yaw_ang_vel: float
    base_height: float
    base_pitch: float

    # a: float
    # def __init__(self, data: DualSenseLocomotionCommand):
    #     for field in fields(DualSenseLocomotionCommand):
    #         setattr(self, field.name, getattr(data, field.name))


def create_data(tar_class, src):
    param_dict = {}
    for field in fields(tar_class):
        assert hasattr(src, field.name)
        param_dict[field.name] = getattr(src, field.name)
    return tar_class(**param_dict)


class DualSenseCommandPublisher:
    def __init__(
        self,
        channel_id: int = 0,
        channel_network_interface: str = None,
        x_vel_command=CommandCfg(
            "x_vel", 0.0, -1.0, 1.0, ContinuousState.LY, 0.5, ControlMode.MOVE
        ),
        y_vel_command=CommandCfg(
            "y_vel", 0.0, -1.0, 1.0, ContinuousState.LX, 0.5, ControlMode.MOVE
        ),
        yaw_vel_command=CommandCfg(
            "yaw_vel", 0.0, -1.0, 1.0, ContinuousState.RX, 0.5, ControlMode.MOVE
        ),
        height_command=CommandCfg(
            "body_height", 1.0, 0.3, 1.3, ContinuousState.LY, 0.5, ControlMode.WORK
        ),
        pitch_command=CommandCfg(
            "body_pitch", 0.0, -0.5, 0.5, ContinuousState.RY, 0.5, ControlMode.WORK
        ),
        dt=0.002,
    ):
        ChannelFactoryInitialize(channel_id, channel_network_interface)
        self.remote_controller = DualSenseLocomotionController(
            x_vel_command=x_vel_command,
            y_vel_command=y_vel_command,
            yaw_vel_command=yaw_vel_command,
            height_command=height_command,
            pitch_command=pitch_command,
            dt=dt,
        )
        self.dt = dt
        self._pub = ChannelPublisher(
            RealH1WalkConfig.topics.remote_controller_topic_name, LocomotionCommand
        )

    def start(self):
        self._pub.Init()
        while True:
            try:
                self.remote_controller.refresh_state()
                command_lin_vel = self.remote_controller.get_command_lin_vel().tolist()
                command_ang_vel = self.remote_controller.get_command_ang_vel().tolist()
                command_height = self.remote_controller.get_command_height().tolist()
                command_pitch = self.remote_controller.get_command_pitch().tolist()
                command = LocomotionCommand(
                    state_start=self.remote_controller.get_state_start(),
                    state_terminate=self.remote_controller.get_state_terminate(),
                    state_move=self.remote_controller.get_state_move(),
                    state_work=self.remote_controller.get_state_work(),
                    x_lin_vel=command_lin_vel[0],
                    y_lin_vel=command_lin_vel[1],
                    yaw_ang_vel=command_ang_vel[0],
                    base_height=command_height[0],
                    base_pitch=command_pitch[0],
                )
                self._pub.Write(command)
                time.sleep(self.dt)

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    print("exit")
                print(e)
                break
        self.close()

    def close(self):
        self._pub.Close()


def run():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="real",
        choices=["real", "sim"],  # Add the available options here
        help="Driver mode {real, sim}",
    )

    args = parser.parse_args()
    if args.mode == "sim":
        channel_param = dict(channel_id=1, channel_network_interface="lo")
    else:
        channel_param = dict(channel_id=0, channel_network_interface=None)
    remote_controller = DualSenseCommandPublisher(**channel_param)
    remote_controller.start()


if __name__ == "__main__":
    run()
