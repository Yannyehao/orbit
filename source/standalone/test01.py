from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different single-arm manipulators.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import traceback

import carb
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.sensors import CameraCfg, Camera

##
# Pre-defined configs
##
from omni.isaac.orbit_assets import FRANKA_PANDA_CFG, UR10_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
    

    # cfg_cone.func("/World/Origin1/Cone", cfg_cone, translation=(0.0, 0.0, 0.1),orientation=(0.5, 0.0, 0.5, 0.0))
    # Origin 1 with Franka Panda
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Origin1/Table", cfg, translation=(0.55, 0.0, 1.05))
    # -- Robot
    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    robot_franka = Articulation(cfg=franka_arm_cfg)


    import random



    positions = [[0.55,0.1,1.05],[0.55,-0.1,1.05],[0.55,0.0,1.05]]
    for i,position in enumerate(positions):
        prim_utils.create_prim(f"/World/Origin1/Cube{i}", "Xform", translation=position)
        
    
    # cube_object = RigidObject(cfg=cube_cfg)
     
    # random.shuffle(positions)

    cfg_cuboid_rigid1 = RigidObjectCfg(
        prim_path="/World/Origin1/Cube0/CuboidRigid1",
        spawn = sim_utils.CuboidCfg(
        size= (0.05,0.05,0.05),
        visible=True, 
        semantic_tags=[("class", "cuboid"), ("color", "red")],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )



    cfg_cuboid_rigid2 = RigidObjectCfg(
        prim_path="/World/Origin1/Cube1/CuboidRigid2",
        spawn = sim_utils.CuboidCfg(
        size= (0.05,0.05,0.05),
        visible=True, 
        semantic_tags=[("class", "cuboid"), ("color", "green")],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )



    cfg_cuboid_rigid3 = RigidObjectCfg(
        prim_path="/World/Origin1/Cube2/CuboidRigid3",
        spawn = sim_utils.CuboidCfg(
        size= (0.05,0.05,0.05),
        visible=True, 
        semantic_tags=[("class", "cuboid"), ("color", "blue")],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    
    
     # sensors
    cfg_camera_side = CameraCfg(
        prim_path="/World/Side_cam",
        update_period=0.1,
        height=1000,
        width=1000,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.15, 1.7), rot=(0.0, 0.0,-0.5,-0.87), convention="opengl"),
    )  
    side_camera = Camera(cfg=cfg_camera_side)
    
    # sensors
    cfg_camera_top = CameraCfg(
        prim_path="/World/Top_cam",
        update_period=0.1,
        height=1000,
        width=1000,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, -0.95, 3.0), rot=(0.71, 0.0,0.0,-0.71), convention="opengl"),
    )  
    top_camera = Camera(cfg=cfg_camera_top) 
    
    
    # sensors
    cfg_camera_girpper = CameraCfg(
        prim_path="/World/Gripper_cam",
        update_period=0.1,
        height=1000,
        width=1000,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.3, -0.993, 1.8), rot=(0.69636, 0.12279,-0.12279,-0.69636), convention="opengl"),
    )  
    gripper_camera = Camera(cfg=cfg_camera_girpper) 
    
    
    
    # cfg_camera.func("/World/Orgin1/Robot/panda_hand/front_cam", cfg_camera)
    cube_object1 = RigidObject(cfg=cfg_cuboid_rigid1)
    cube_object2 = RigidObject(cfg=cfg_cuboid_rigid2)
    cube_object3 = RigidObject(cfg=cfg_cuboid_rigid3)

    
    
    scene_entities = {"robot_franka": robot_franka,  
                      "cube_object1": cube_object1, 
                      "cube_object2": cube_object2, 
                      "cube_object3": cube_object3,
                      "top_camera": top_camera}
    return scene_entities, origins 



import random

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    scene_entities_values = list(entities.values())  # convert dict_values to list
    cube_values = scene_entities_values[1:4] # exclude the robot
    origins = origins[0]
    top_camera = entities["top_camera"]
    image_counter = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # obtain the current positions of the cubes
            current_positions=[]
            current_positions = [cube_values[i].data.default_root_state[:,:3].clone() for i in range(3)]
            print(f"[INFO]: Current positions: {current_positions}")
            
            global_positions = [
                [0.55, -0.6, 1.05],  # Cube0
                [0.55, -1.0, 1.05],  # Cube1
                [0.55, -1.4, 1.05]   # Cube2
            ]
            
            for i in range(3):
                # convert the global_positions to tensor, make sure the device is the same as the current_positions[i]
                position_update = torch.tensor(global_positions[i], device=current_positions[i].device)
                # because the current_positions[i] is a tensor of shape (1, 3), we need to expand the position_update to (1, 3)
                current_positions[i][:, :3] = position_update
                
            # create indeces and shuffle them
            indices = list(range(3))
            random.shuffle(indices)
            
            # update the positions of the cubes
            for i in range(3):
                target_index = indices[i]
                cube_values[i].data.default_root_state[:, :3] = current_positions[target_index]
                
                # write the new state to the sim
                cube_values[i].write_root_state_to_sim(cube_values[i].data.default_root_state)
                cube_values[i].reset()
                
            print("-----------------")
            print("[INFO]: Resetting cuboid state")
        for i in range(3):
            cube_values[i].write_data_to_sim()        
            # reset the scene entities
        #     for index, robot in enumerate(entities.values()):
        #         # root state
        #         root_state = robot.data.default_root_state.clone()
        #         root_state[:, :3] += origins[index]
        #         robot.write_root_state_to_sim(root_state)
        #         # set joint positions
        #         joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        #         robot.write_joint_state_to_sim(joint_pos, joint_vel)
        #         # clear internal buffers
        #         robot.reset()
        #     print("[INFO]: Resetting robots state...")
            
            

        # # apply random actions to the robots
        # for robot in entities.values():
        #     # generate random joint positions
        #     joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
        #     joint_pos_target = joint_pos_target.clamp_(
        #         robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
        #     )
        #     # apply action to the robot
        #     robot.set_joint_position_target(joint_pos_target)
        #     # write data to sim
        #     robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        for i in range(3):
            cube_values[i].update(sim_dt)
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)
            
        print("-------------------------------")
        print("camera")
        rgb_tensor = top_camera.data.output["rgb"]
        rgb_tensor_normalized = rgb_tensor.float() / 255.0
        print("Received shape of rgb image: ",  top_camera.data.output["rgb"].shape)
        
        # # check the minimum and maximum values
        # min_val = rgb_tensor_normalized.min()
        # max_val = rgb_tensor_normalized.max()

        # # print the minimum and maximum values
        # print("Minimum value in the tensor: ", min_val.item())
        # print("Maximum value in the tensor: ", max_val.item())

        # # check if the tensor values are within the [0, 1] range
        # if min_val >= 0 and max_val <= 1:
        #     print("The tensor values are within the [0, 1] range.")
        # else:
        #     print("The tensor values are NOT within the [0, 1] range.")
        
        from PIL import Image
        import os
        # use PIL to save the top camera rgb output
        img_tensor = rgb_tensor_normalized
        img = img_tensor.squeeze()
        img = (img*255).byte()
        
        img_pil = Image.fromarray(img.cpu().numpy(), 'RGBA')
        
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        save_path = os.path.join(output_dir, f"top_camera_rgb_{image_counter}.png")
        img_pil.save(save_path)

        print(f"Saved top camera rgb image to {save_path}")
        image_counter += 1    
        
        







        

            
            
def main():
    """Main function."""
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()