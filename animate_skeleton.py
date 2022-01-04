import bpy
import os,sys
import copy
import numpy as np
import math
import time

sys.path.insert(0,"./")



import utils
from PFNN import PFNN
from mathutils import Matrix, Quaternion,Euler,Vector


bone_parent={}

extra_gait_smooth = 0.1
extra_joint_smooth = 0.5

def preprocess_blender_scene():
    allowed_objects = ['Camera']

    #Delete default initial objects
    #except camera and Lamp
    bpy.ops.object.select_all(action='DESELECT')
    current_objects = bpy.data.objects.keys()
    print(current_objects)

    for obj_ in current_objects:
        if obj_ not in allowed_objects:
            bpy.data.objects[obj_].select_set(True)
            bpy.ops.object.delete(use_global=False)
            # delete the default cube (which held the material)
            bpy.ops.object.select_all(action='DESELECT')
    current_objects = bpy.data.objects.keys()
    print(current_objects)






class Character:

    def __init__(self) -> None:
        self.old_joint_positions = []
        self.old_joint_rotations = []
        self.old_joint_velocities = []
        self.num_joints = 31
        self._joint_positions=[Vector((0,0,0))]*self.num_joints
        self._joint_velocities=[Vector((0,0,0))]*self.num_joints
        self._joint_rotations=[Euler((0,0,0)).to_matrix()]*self.num_joints
        self.phase=0
        
        self.bone_parent={}
        self.load_character()
        
        
        pass

    def load_character(self):
        bpy.ops.import_scene.fbx(
        filepath=os.path.join("./skeleton.fbx"),
        global_scale=1,
    )
       
        bpy.data.objects["Skeleton"].rotation_euler = (0,0,0)
        
        # bpy.ops.object.transform_apply(rotation=True)
        bpy.context.view_layer.update()


       
        self.arm_ob = bpy.data.objects["Skeleton"]

        for i in range(self.num_joints):
            if i==0:
                self.bone_parent[f"J{i+1}"]=None
            else:
                obj = bpy.data.objects[f"J{i+1}"]
                self.bone_parent[f"J{i+1}"]=obj.parent.name
            bone = bpy.data.objects[f"J{i+1}"]
            bone.keyframe_insert("location", frame=0)

    def get_bone_data(self):
        _joint_positions=[]
        _joint_rotations=[]
        _joint_velocities= np.zeros((self.num_joints,3))


        for i in range(self.num_joints):


            bone_name = f"J{i+1}"
            bone =  bpy.data.objects[bone_name]
            bone_location =bone.matrix_world.translation
            bone_location = Vector((-bone_location.x,bone_location.y,bone_location.z))*100

            _joint_positions.append(bone_location)
            _joint_rotations.append(bone.rotation_euler)


        
        if len(self.old_joint_positions)!=0:
            
            for i,_ in enumerate(_joint_positions):

                velocity = _joint_positions[i]-self.old_joint_positions[i]
                _joint_velocities[i] = Vector(velocity)


        self.old_joint_positions = _joint_positions.copy()


        return _joint_positions,_joint_rotations,_joint_velocities
    
    def set_bone_data(self,root_position,root_rotation,_joint_positions,_joint_rotations,_joint_velocities):


        # for i,bone_name in enumerate(self.arm_ob.pose.bones.keys()):


        #     # print(i,bone_name)
        #     bone =  self.arm_ob.pose.bones[bone_name]
        #     bone.rotation_mode = "XYZ"
        #     bone.rotation_euler = _joint_rotations[i].to_euler()

        # bpy.ops.object.mode_set(mode='EDIT',toggle=True)
        # bpy.ops.object.mode_set(mode='POSE')

        bone = bpy.data.objects['Skeleton']
        bone.location = Vector((-root_position.x,root_position.y,root_position.z))/100
        # print(bone.location)
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = root_rotation.to_quaternion()
        bone.keyframe_insert(data_path = 'location')
        bone.keyframe_insert(data_path = 'rotation_quaternion')

        bpy.context.view_layer.update()
        for i in range(self.num_joints):


            # print(i,bone_name)

            pos = _joint_positions[i]
            bone_name = f"J{i+1}"
           
            # pdb.set_trace()
            pos = pos/100
            pos =  Vector((-pos.x,pos.y,pos.z))
            # parent_name = ch.bone_parent[bone_name]
            # parent = bpy.data.objects[parent_name]
            # new_pos = parent.matrix_world.inverted()@ pos
        
            bone =  bpy.data.objects[bone_name] 
            bone.location = bone.parent.matrix_world.inverted()@ pos
            bone.keyframe_insert(data_path = 'location')

            bpy.context.view_layer.update()

        # bpy.ops.object.mode_set(mode='EDIT',toggle=True)





class Trajectory:

    def __init__(self) -> None:
        
        self.LENGTH = 120
        self.width = 25
        self.positions = [Vector((0,0,0))]*self.LENGTH
        self.directions = [Vector((0,0,1))]*self.LENGTH
        self.rotations = [Vector((0,0,0))]*self.LENGTH
        self.heights = np.zeros(self.LENGTH)
        self.gait_stand=np.zeros(self.LENGTH)
        self.gait_walk = np.zeros(self.LENGTH)
        self.gait_jog = np.zeros(self.LENGTH)
        self.gait_crouch = np.zeros(self.LENGTH)
        self.gait_jump = np.zeros(self.LENGTH)
        self.gait_bump = np.zeros(self.LENGTH)
       



        self.target_dir = Vector([0,0,1])
        self.target_vel = Vector([0,0,0])



    
    def reset(self,root_position,root_rotation):
        for i in range(self.LENGTH):
            self.positions.append(root_position)
            self.rotations.append(root_rotation)
            self.directions.append((0,0,1))
            self.heights.append(root_position.y)
            self.gait_stand.append(0.0)
            self.gait_walk.append(0.0)
            self.gait_jog.append(0.0)
            self.gait_crouch.append(0.0)
            self.gait_jump.append(0.0)
            self.gait_bump.append(0.0)
  



  


    def walk(self):

        self.gait_stand[self.LENGTH//2]  = utils.lerp(self.gait_stand[self.LENGTH//2],  0.0, extra_gait_smooth)
        self.gait_walk[self.LENGTH//2]   = utils.lerp(self.gait_walk[self.LENGTH//2],   1.0, extra_gait_smooth)
        self.gait_jog[self.LENGTH//2]    = utils.lerp(self.gait_jog[self.LENGTH//2],    0.0, extra_gait_smooth)
        self.gait_crouch[self.LENGTH//2] = utils.lerp(self.gait_crouch[self.LENGTH//2], 0.0, extra_gait_smooth)
        self.gait_jump[self.LENGTH//2]   = utils.lerp(self.gait_jump[self.LENGTH//2],   0.0, extra_gait_smooth)  
        self.gait_bump[self.LENGTH//2]   = utils.lerp(self.gait_bump[self.LENGTH//2],   0.0, extra_gait_smooth) 


    def stand(self,stand_amount=0):

        self.gait_stand[self.LENGTH//2]  = utils.lerp(self.gait_stand[self.LENGTH//2],  stand_amount, extra_gait_smooth)
        self.gait_walk[self.LENGTH//2]   = utils.lerp(self.gait_walk[self.LENGTH//2],   0.0, extra_gait_smooth)
        self.gait_jog[self.LENGTH//2]    = utils.lerp(self.gait_jog[self.LENGTH//2],    0.0, extra_gait_smooth)
        self.gait_crouch[self.LENGTH//2] = utils.lerp(self.gait_crouch[self.LENGTH//2], 0.0, extra_gait_smooth)
        self.gait_jump[self.LENGTH//2]   = utils.lerp(self.gait_jump[self.LENGTH//2],   0.0, extra_gait_smooth)  
        self.gait_bump[self.LENGTH//2]   = utils.lerp(self.gait_bump[self.LENGTH//2],   0.0, extra_gait_smooth)  






def reset_scene(pfnn,character,trajectory):

    Yp = pfnn
    pass


from dataclasses import dataclass
@dataclass
class PFNNSetting:
    weight_path="TensorFlow/trained_original/"
    XDIM = 342
    YDIM = 311
    HDIM = 512
    phase_index=3


if __name__=="__main__":

  
    trans_offset = 0
    start_pos = None
    end_pos = None
    velocity = 1
    #preprocess blender scene
    preprocess_blender_scene()
    #loadign_adam

    trajectory = Trajectory()

   
    pfnn = PFNN(PFNNSetting.weight_path,PFNNSetting.XDIM,PFNNSetting.YDIM,PFNNSetting.HDIM,Phase_index=PFNNSetting.phase_index)

    ch = Character()



    trajectory.target_vel = Vector((0,0,2))
    trajectory.target_dir = Vector((0,0,1))
    matrix = Euler((0,np.pi/2,0)).to_matrix()

   

    joint_positions,_,_ = ch.get_bone_data()

    ch._joint_positions = joint_positions


    for iter in range(600):

        #after 300 frames changing direction
        if iter>300:
            trajectory.target_dir = matrix@Vector((0,0,1))
            trajectory.target_vel = matrix@Vector((0,0,2))


       
        #directions

        if trajectory.target_vel.length<0.1:
            stand_amount = 1 - np.clip(np.linalg.norm(trajectory.target_vel)/0.1, 0.0, 1.0)
            trajectory.stand(stand_amount)
        else:
            trajectory.walk()
        # pdb.set_trace()
        #predict future trajectoru
        trajectory_positions_blend = [Vector((0,0,0))]*trajectory.LENGTH

        trajectory_positions_blend[trajectory.LENGTH//2] = trajectory.positions[trajectory.LENGTH//2]

        # pdb.set_trace()

        for i in range(trajectory.LENGTH//2+1,trajectory.LENGTH):
            bias_pos = 0.75
            bias_dir = 1.25
            scale_pos =   (1.0 - math.pow(1.0- ((float)(i - trajectory.LENGTH//2) / (trajectory.LENGTH//2)), bias_pos))
            scale_dir = (1.0 - math.pow(1.0- ((float)(i - trajectory.LENGTH//2) / (trajectory.LENGTH//2)), bias_dir))
            trajectory_positions_blend[i] = trajectory_positions_blend[i-1] +utils.lerp(trajectory.positions[i] - trajectory.positions[i-1], trajectory.target_vel,scale_pos)
            trajectory.directions[i] = utils.mix_directions(trajectory.directions[i],trajectory.target_dir,scale_dir)
            # trajectory->directions[i] = mix_directions(trajectory->directions[i], trajectory->target_dir, scale_dir);
            trajectory.heights[i] = trajectory.heights[trajectory.LENGTH//2]
            trajectory.gait_stand[i] = trajectory.gait_stand[trajectory.LENGTH//2]
            trajectory.gait_walk[i] = trajectory.gait_walk[trajectory.LENGTH//2]
            trajectory.gait_jog[i] = trajectory.gait_jog[trajectory.LENGTH//2]
            trajectory.gait_crouch[i] = trajectory.gait_crouch[trajectory.LENGTH//2]
            trajectory.gait_jump[i] = trajectory.gait_jump[trajectory.LENGTH//2]
            trajectory.gait_bump[i] = trajectory.gait_bump[trajectory.LENGTH//2]


        for i in range(trajectory.LENGTH//2+1,trajectory.LENGTH):
            trajectory.positions[i] = trajectory_positions_blend[i]


        for i in range(0,trajectory.LENGTH):

            dir1 = trajectory.directions[i]
            trajectory.rotations[i] = Quaternion((0.0, 1.0, 0.0),math.atan2(dir1.x, dir1.z) ).to_matrix().transposed()


        for i in range(trajectory.LENGTH//2,trajectory.LENGTH):
            #setting a plane
            trajectory.positions[i].y = 0
        

        trajectory.heights[trajectory.LENGTH//2] = 0.0


        for i in range(0,trajectory.LENGTH,10):
            trajectory.heights[trajectory.LENGTH//2] += (trajectory.positions[i].y / ((trajectory.LENGTH)//10))

       
        root_position = Vector((trajectory.positions[trajectory.LENGTH//2].x, 
                                    trajectory.heights[trajectory.LENGTH//2],
                                    trajectory.positions[trajectory.LENGTH//2].z))

        
        root_rotation = trajectory.rotations[trajectory.LENGTH//2]


        Xp = np.random.random(PFNNSetting.XDIM)


        for i in range(0,trajectory.LENGTH,10):
            w = (trajectory.LENGTH)//10

            pos = root_rotation.inverted() @ (trajectory.positions[i] - root_position)

            dir = root_rotation.inverted() @ trajectory.directions[i]



            Xp[(w*0)+i//10] = pos.x
            Xp[(w*1)+i//10] = pos.z
            Xp[(w*2)+i//10] = dir.x 
            Xp[(w*3)+i//10] = dir.z

      

        for i in range(0,trajectory.LENGTH,10):

            w = (trajectory.LENGTH)//10

            Xp[(w*4)+i//10] = trajectory.gait_stand[i]
            Xp[(w*5)+i//10] = trajectory.gait_walk[i]
            Xp[(w*6)+i//10] = trajectory.gait_jog[i]
            Xp[(w*7)+i//10] = trajectory.gait_crouch[i]
            Xp[(w*8)+i//10] = trajectory.gait_jump[i]
            Xp[(w*9)+i//10] = 0.0

         
        
        prev_root_position = Vector((
                    trajectory.positions[trajectory.LENGTH//2-1].x, 
                    trajectory.heights[trajectory.LENGTH//2-1],
                    trajectory.positions[trajectory.LENGTH//2-1].z))

        prev_root_rotation = trajectory.rotations[trajectory.LENGTH//2-1]

        # pdb.set_trace()

        # _joint_positions,_joint_rotations,_joint_velocities = ch.get_bone_data()

        for i in range(ch.num_joints):


            o = int(((trajectory.LENGTH)//10)*10)

            pos = prev_root_rotation.inverted() @ Vector((ch._joint_positions[i] - prev_root_position))

            prv = prev_root_rotation.inverted() @ Vector(ch._joint_velocities[i])


            Xp[o+(ch.num_joints*3*0)+i*3+0] = pos.x
            Xp[o+(ch.num_joints*3*0)+i*3+1] = pos.y
            Xp[o+(ch.num_joints*3*0)+i*3+2] = pos.z
            Xp[o+(ch.num_joints*3*1)+i*3+0] = prv.x
            Xp[o+(ch.num_joints*3*1)+i*3+1] = prv.y
            Xp[o+(ch.num_joints*3*1)+i*3+2] = prv.z


        for i in range(0,trajectory.LENGTH,10):
            o = (((trajectory.LENGTH)//10)*10)+ch.num_joints*3*2
            w = trajectory.LENGTH//10
            position_r = trajectory.positions[i] + trajectory.rotations[i] @ Vector(( trajectory.width, 0, 0))

            position_l =  trajectory.positions[i] + trajectory.rotations[i]  @ Vector(( -trajectory.width, 0, 0))

            Xp[o+(w*0)+(i//10)] = 0 - root_position.y
            Xp[o+(w*1)+(i//10)] = trajectory.positions[i].y - root_position.y
            Xp[o+(w*2)+(i//10)] = 0 - root_position.y


    

        timephase = ch.phase
        stand_amount = math.pow(1.0-trajectory.gait_stand[trajectory.LENGTH//2], 0.25)
        pfnn.setdamping( 1-(stand_amount * 0.9 + 0.1))
        Yp =pfnn.predict(Xp)

  
      

        for i in range(ch.num_joints):
        
            opos = 8+(((trajectory.LENGTH//2)//10)*4)+(ch.num_joints*3*0)
            ovel = 8+(((trajectory.LENGTH//2)//10)*4)+(ch.num_joints*3*1)
            orot = 8+(((trajectory.LENGTH//2)//10)*4)+(ch.num_joints*3*2)

            pos = (root_rotation @ Vector((Yp[opos+i*3+0], Yp[opos+i*3+1], Yp[opos+i*3+2]))) + root_position
            vel = (root_rotation @ Vector((Yp[ovel+i*3+0], Yp[ovel+i*3+1], Yp[ovel+i*3+2])))
            qq = utils.quat_exp(Vector((Yp[orot+i*3+0], Yp[orot+i*3+1], Yp[orot+i*3+2]))).to_matrix().transposed()
            rot = (root_rotation @ qq ).to_quaternion()



            
            
            # pdb.set_trace()
 
            ch._joint_positions[i]=utils.lerp(ch._joint_positions[i]+vel,pos,extra_joint_smooth)
            ch._joint_velocities[i]=vel
            ch._joint_rotations[i]=rot


            bone_name = f"J{i+1}"
      
        
        for i in range(0, trajectory.LENGTH//2):

            trajectory.positions[i]  = trajectory.positions[i+1]
            trajectory.directions[i] = trajectory.directions[i+1]
            trajectory.rotations[i] = trajectory.rotations[i+1]
            trajectory.heights[i] = trajectory.heights[i+1]
            trajectory.gait_stand[i] = trajectory.gait_stand[i+1]
            trajectory.gait_walk[i] = trajectory.gait_walk[i+1]
            trajectory.gait_jog[i] = trajectory.gait_jog[i+1]
            trajectory.gait_crouch[i] = trajectory.gait_crouch[i+1]
            trajectory.gait_jump[i] = trajectory.gait_jump[i+1]
            trajectory.gait_bump[i] = trajectory.gait_bump[i+1]


        rootIndex = trajectory.LENGTH//2
        # trajectory.positions[rootIndex] = 

        # pdb.set_trace()

        stand_amount = math.pow(1.0-trajectory.gait_stand[trajectory.LENGTH//2], 0.25)
    
        trajectory_update = trajectory.rotations[trajectory.LENGTH//2] @ Vector((Yp[0], 0,Yp[1]))

        trajectory.positions[trajectory.LENGTH//2]  = trajectory.positions[trajectory.LENGTH//2] + stand_amount * trajectory_update

        trajectory.directions[trajectory.LENGTH//2] = Euler((0,stand_amount * -Yp[2],0)).to_matrix() @ trajectory.directions[trajectory.LENGTH//2]

        trajectory.rotations[trajectory.LENGTH//2] = Euler((0,math.atan2(trajectory.directions[trajectory.LENGTH//2].x,trajectory.directions[trajectory.LENGTH//2].z,),0)).to_matrix().transposed()
        

        # pdb.set_trace()

        for i in range(trajectory.LENGTH//2+1, trajectory.LENGTH):


            w = (trajectory.LENGTH//2)//10

            m = math.fmod((i - (trajectory.LENGTH//2)) / 10.0, 1.0)


            
            # pdb.set_trace()

            trajectory.positions[i].x  = ((1-m) * Yp[8+(w*0)+(i//10)-w] + m * Yp[8+(w*0)+(i//10)-w+1])
            trajectory.positions[i].z  = ((1-m) * Yp[8+(w*1)+(i//10)-w] + m * Yp[8+(w*1)+(i//10)-w+1])
            trajectory.directions[i].x = (1-m) * Yp[8+(w*2)+(i//10)-w] + m * Yp[8+(w*2)+(i//10)-w+1]
            trajectory.directions[i].z = (1-m) * Yp[8+(w*3)+(i//10)-w] + m * Yp[8+(w*3)+(i//10)-w+1]
            trajectory.positions[i]    = (trajectory.rotations[trajectory.LENGTH//2] @ trajectory.positions[i]) + trajectory.positions[trajectory.LENGTH//2]
            traj_dir = (trajectory.rotations[trajectory.LENGTH//2] @ trajectory.directions[i])
            trajectory.directions[i]   = traj_dir.normalized()
            trajectory.rotations[i]    = Euler((0,math.atan2(trajectory.directions[i].x, trajectory.directions[i].z),0 )).to_matrix().transposed()


    

    
        scene = bpy.context.scene
        
        scene.frame_set(iter)
      
        print("-----")
        ch.set_bone_data(root_position,root_rotation,ch._joint_positions,ch._joint_rotations,[])



    # pdb.set_trace()



    
