using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum RotationDirection { None = 0, Positive = 1, Negative = -1 };

public class ArticulationJointController : MonoBehaviour
{
    public RotationDirection rotationState = RotationDirection.None;
    public float speed = 300f;

    public float speed2 = 1f;

    private ArticulationBody articulation;


    // LIFE CYCLE

    void Start()
    {
        articulation = GetComponent<ArticulationBody>();
    }

    void FixedUpdate() 
    {
        if (rotationState != RotationDirection.None)
        {
            float rotationChange = (float)rotationState * speed * Time.deltaTime;
            float rotationGoal = CurrentPrimaryAxisRotation() + rotationChange;
            RotateTo(rotationGoal);
        }
    }

    public void RotateJoint(float value)
    {
        float rotationChange = (float)value * speed2 * Time.deltaTime;
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

    public void RotateTo(float primaryAxisRotation)
    {
        var drive = articulation.xDrive;
        drive.target = primaryAxisRotation;
        articulation.xDrive = drive;
    }



}
