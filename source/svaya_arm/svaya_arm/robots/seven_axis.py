import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

SevenAxis_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="data/seven_axis_Gripper/seven_axis_Gripper.usd",
        activate_contact_sensors=True,          # ← required for contact reporting
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,       # ← PhysX must track self contacts
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "joint_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            damping=None,
            stiffness=None,
        )
    },
)