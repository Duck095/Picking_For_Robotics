class EnvConfig:
    # Observation
    IMG_SIZE = 84
    FRAME_STACK = 4

    # Physics
    PHYSICS_HZ = 240
    RL_HZ = 20
    SUBSTEPS = PHYSICS_HZ // RL_HZ  # 12

    # Episode
    MAX_STEPS = 300

    # Action scale (delta EE per RL step)
    ACTION_SCALE_XY = 0.02
    ACTION_SCALE_Z = 0.02

    # Stage 1 success condition
    LIFT_HEIGHT = 0.1

    # Reward (Stage 1)
    TIME_PENALTY = 0.01
    DIST_WEIGHT = 0.2
    GRASP_REWARD = 2.0
    SUCCESS_BONUS = 2.0