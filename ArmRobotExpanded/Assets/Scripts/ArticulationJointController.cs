using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum RotationDirection { None = 0, Positive = 1, Negative = -1 };

public class ArticulationJointController : MonoBehaviour
{
    public RotationDirection rotationState = RotationDirection.None;
    public float rotationValue = 0f;
    public float speed = 300f;

    public float speed2 = 1f;

    public ArticulationBody articulation;


    // LIFE CYCLE

    void Start()
    {
        articulation = GetComponent<ArticulationBody>();
    }

    void FixedUpdate() 
    {
        // if (rotationState != RotationDirection.None)
        // {
        //     float rotationChange = (float)rotationState * speed * Time.deltaTime;
        //     float rotationGoal = CurrentPrimaryAxisRotation() + rotationChange;
        //     RotateTo(rotationGoal);
        // }
        // if (rotationValue > 0f)
        // {
        //     RotateJoint(rotationValue);
        // }
    }

    public void RotateJoint(float value)
    {
        float rotationChange = (float)value * speed * Time.deltaTime;
        // if (Mathf.Abs(rotationChange) > Mathf.Epsilon)
        // {
        float rotationGoal = CurrentPrimaryAxisRotation() + rotationChange;
        // Debug.Log("Rotation Change = "+rotationChange);
        // Debug.Log("CurrentPrimaryAxisRotation = " + CurrentPrimaryAxisRotation());
        RotateTo(rotationGoal);            
        // }

    }

    public void RotateJoint(RotationDirection value)
    {
        float rotationChange = (float)value * speed * Time.deltaTime;
        float rotationGoal = CurrentPrimaryAxisRotation() + rotationChange;
        RotateTo(rotationGoal);
    }    

    public void ForceToRotation(float rotation)
    {
        RotateTo(rotation);

        float radians = Mathf.Deg2Rad * rotation;
        ArticulationReducedSpace newPosition = new ArticulationReducedSpace(radians);
        articulation.jointPosition = newPosition;

        ArticulationReducedSpace newVelocity = new ArticulationReducedSpace(0.0f);
        articulation.jointPosition = newVelocity;
    }

    // MOVEMENT HELPERS

    public float CurrentPrimaryAxisRotation()
    {
        float currentRotationRads = articulation.jointPosition[0];
        float currentRotation = Mathf.Rad2Deg * currentRotationRads;
        // float currentRotation = currentRotationRads;
        return currentRotation;
    }

    public float CurrentJointVelocity()
    {
        float velocity = articulation.jointVelocity[0];
        velocity = Mathf.Rad2Deg * velocity;
        return velocity;
    }

    public List<float> CollectionExtraInformation()
    {
        List<float> obs = new List<float>();
        obs.Add(articulation.jointVelocity[0]);
        obs.Add(articulation.jointForce[0]);
        obs.Add(articulation.jointAcceleration[0]);
        return obs;
    }
    

    public void RotateTo(float primaryAxisRotation)
    {

        var drive = articulation.xDrive;
        drive.target = primaryAxisRotation;
        articulation.xDrive = drive;
    }



}
