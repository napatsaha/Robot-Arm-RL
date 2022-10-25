using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class RobotController : MonoBehaviour
{
    [System.Serializable]
    public struct Joint
    {
        public string inputAxis;
        public GameObject robotPart;
    }
    public Joint[] joints;


    // CONTROL

    public void StopAllJointRotations()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            GameObject robotPart = joints[i].robotPart;
            //ArticulationJointController articulation = robotPart.GetComponent<ArticulationJointController>();
            //ArticulationBody articulationBody = articulation.GetComponent<ArticulationBody>();
            //articulationBody.jointVelocity = new ArticulationReducedSpace(0.0f);

            UpdateRotationState(RotationDirection.None, robotPart);
            //UpdateRotationState(0f, robotPart);
        }
    }

    public void ResetRotation(float[] rotations)
    {
        for (int i = 0; i < joints.Length; i++)
        {
            Joint joint = joints[i];
            ArticulationJointController articulation = joint.robotPart.GetComponent<ArticulationJointController>();
            articulation.ForceToRotation(rotations[i]);
        }

        //StopAllJointRotations();
    }

    public void RotateJoint(int jointIndex, RotationDirection direction)
    {
        //StopAllJointRotations();
        Joint joint = joints[jointIndex];
        UpdateRotationState(direction, joint.robotPart);
    }
    public void RotateJoint(int jointIndex, float movement)
    {
        //StopAllJointRotations();
        Joint joint = joints[jointIndex];
        UpdateRotationState(movement, joint.robotPart);
    }

    // HELPERS

    static void UpdateRotationState(RotationDirection direction, GameObject robotPart)
    {
        ArticulationJointController jointController = robotPart.GetComponent<ArticulationJointController>();
        jointController.rotationState = direction;
    }

    static void UpdateRotationState(float movement, GameObject robotPart)
    {
        ArticulationJointController jointController = robotPart.GetComponent<ArticulationJointController>();
        jointController.RotateJoint(movement);
    }



}
