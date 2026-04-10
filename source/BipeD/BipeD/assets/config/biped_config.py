import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/BD/BD.usd")

BD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),

        activate_contact_sensors=True,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "J_L0":   0.0,
            "J_L1":  0.08,
            "J_L2":  0.56,
            "J_L3":  -1.12,
            "J_L4_ankle": -0.57,

            "J_R0":   0.0,
            "J_R1":  -0.08,
            "J_R2":  -0.56,
            "J_R3":  1.12,
            "J_R4_ankle": 0.57
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "J_L0",
                "J_R0",
                "J_L1",
                "J_R1",
                "J_L2",
                "J_R2",
                "J_L3",
                "J_R3",
                "J_L4_ankle",
                "J_R4_ankle"
            ],
            effort_limit=20,
            velocity_limit=15.0,
            stiffness={
                "J_L0": 13,
                "J_R0": 13,
                "J_L1": 15,
                "J_R1": 15, 
                "J_L2": 15,
                "J_R2": 15,
                "J_L3": 15,
                "J_R3": 15, 
                "J_L4_ankle": 13,
                "J_R4_ankle": 13
            },
            damping={
                "J_L0": 0.3, 
                "J_R0": 0.3, 
                "J_L1": 0.65,
                "J_R1": 0.65,
                "J_L2": 0.65, 
                "J_R2": 0.65, 
                "J_L3": 0.65, 
                "J_R3": 0.65, 
                "J_L4_ankle": 0.3, 
                "J_R4_ankle": 0.3
            },
        ),
    },
)