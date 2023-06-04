using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PositionRandomizer : MonoBehaviour
{
    [SerializeField] GameObject table;
    [SerializeField] GameObject arm;
    [SerializeField] GameObject armTip;
    // [SerializeField] GameObject activeJoint;

    public int jointIndex;
    Rigidbody targetRigidBody;
    RobotController controller;
    GameObject activeJoint;
    GameObject handJoint;

    // float radius = Mathf.Sqrt(Mathf.Pow(0.485f,2) + Mathf.Pow(-0.17f, 2));
    float radius = 1f;
    public float radiusError = 0f;
    public float minLength = 1f;
    float minRadius;
    float targetRadius;
    float height;
    Vector3 center;

    Bounds tableBounds;
    Bounds armBase;

    private void Start() {
        targetRigidBody = transform.GetComponent<Rigidbody>();
        targetRadius = transform.GetComponent<Collider>().bounds.extents.x;
        tableBounds = table.GetComponent<Collider>().bounds;
        controller = arm.GetComponent<RobotController>();
        // activeJoint = controller.joints[jointIndex].robotPart;
        handJoint = controller.joints[6].robotPart;
        armBase = arm.GetComponent<Collider>().bounds;
        height = table.transform.position.y;
        // center = activeJoint.transform.position;
        // height = activeJoint.transform.position.y;
        // radius = tableBounds.extents.z;
        // height = armTip.transform.position - arm.transform.position;
        // radius = GetChildJointSize();
        // Debug.Log("Radius = "+radius);
        // radius = handJoint.GetComponent<Collider>().bounds.size.z + activeJoint.GetComponent<Collider>().bounds.extents.z;
        // minRadius = minLength*radius;
        // Debug.Log(radius);
        // Debug.Log(height);
        // Debug.Log(activeJoint.gameObject.name + " " + childJoint.gameObject.name);
    }

    public void Randomize()
    {
        activeJoint = controller.joints[jointIndex].robotPart;
        // handJoint = controller.joints[6].robotPart;
        // armBase = arm.GetComponent<Collider>().bounds;
        center = activeJoint.transform.position;
        // height = activeJoint.transform.position.y;

        float handLength = handJoint.GetComponent<Collider>().bounds.size.z;
        radius = GetChildJointSize() + handLength;
        minRadius = minLength*radius;
        // Debug.Log("Radius = "+radius);

        // Reset Target Motion
        targetRigidBody.angularVelocity = Vector3.zero;
        targetRigidBody.velocity = Vector3.zero;

        // Randomize Target Position

        Vector3 offset = GetRandomOffset(center);
        Vector3 point = center + offset;
        // Debug.Log("Center = "+center+", Offset = "+offset);
        // Debug.Log("Offset = "+offset);
        transform.position = point;

        // Randomize Target Rotation
        Vector3 randomRotation = new Vector3(
            Random.value * 360.0f, 
            Random.value * 360.0f, 
            Random.value * 360.0f);
        transform.rotation = Quaternion.Euler(randomRotation);

        // 2D
        //Vector2 center = Vector2.zero;
        //Vector2 randomPoint = center + Random.insideUnitCircle * radius;
        //target.transform.position = new Vector3(randomPoint.x, 0.778f, randomPoint.y);

        //Debug.Log("New Position" + target.transform.position);

        // 3D
        // Vector3 center = new Vector3(0f, 0.778f, 0f);

    }
    private float GetChildJointSize()
    {
        float length = 0f;
        for (int i = jointIndex; i < controller.joints.Length - 1; i++)
        {
            // controller.joints[i].robotPart.GetComponent<ArticulationBody>().anchorRotatio
            Vector3 box = controller.joints[i].robotPart.GetComponent<Collider>().bounds.extents;
            float longestDimension = Mathf.Max(box.x, box.y, box.z);
            length += longestDimension;
        }
        return length;
    }

    private Vector3 GetRandomOffset(Vector3 center)
    {
        // Vector2 offset2D = Random.insideUnitCircle * radius;
        // float offsetY = Random.value * armHeight.y;
        // Vector3 offset = new Vector3(offset2D.x, offsetY, offset2D.y);
        // offset.y = Mathf.Abs(offset.y);

        Quaternion anchorRotation = activeJoint.GetComponent<ArticulationBody>().anchorRotation;
        while (true)
        {
            Vector2 offset2D = Random.insideUnitCircle * (radiusError + radius);
            Vector3 offset;
            if(anchorRotation.z > 0){
                // Active joint spins in y axis
                // Possible area is a horizontal Arc around joint
                offset = new Vector3(offset2D.x, 0f, offset2D.y);
            } else
            {
                // Active Joint spins in z axis
                // Possible area is a vertical Arc around joint
                offset = new Vector3(offset2D.x, offset2D.y, 0f);
            }
            if (((center+offset).y > (height + targetRadius)) && // To make sure position is above table height
                (Mathf.Abs(offset.magnitude - minRadius) < 2*targetRadius) && // To make sure position is in periphery of circle
                (Mathf.Abs(offset.x) > 2*(armBase.extents.x + targetRadius)) // To make sure position far away from the base
                ) {
                return offset;
            }
            
            // if ((Mathf.Abs(offset2D.x) > minRadius) && (Mathf.Abs(offset2D.y) > minRadius))
            // {
            //     Vector3 offset = new Vector3(offset2D.x, height, offset2D.y);
            //     return offset;
            // }
            
        }

    }

}
