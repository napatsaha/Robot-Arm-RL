﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RobotManualInput : MonoBehaviour
{
    public GameObject robot;


    void Update()
    {
        RobotController robotController = robot.GetComponent<RobotController>();
        for (int i = 0; i < robotController.joints.Length; i++)
        {
            float inputVal = Input.GetAxis(robotController.joints[i].inputAxis);
            // Debug.Log(robotController.joints[i].inputAxis+" = "+inputVal);
            if (Mathf.Abs(inputVal) > Mathf.Epsilon)
            {
                RotationDirection direction = GetRotationDirection(inputVal);
                robotController.RotateJoint(i, inputVal);
                return;
            }
        }
        robotController.StopAllJointRotations();

    }


    // HELPERS

    static RotationDirection GetRotationDirection(float inputVal)
    {
        if (inputVal > 0)
        {
            return RotationDirection.Positive;
        }
        else if (inputVal < 0)
        {
            return RotationDirection.Negative;
        }
        else
        {
            return RotationDirection.None;
        }
    }
}
