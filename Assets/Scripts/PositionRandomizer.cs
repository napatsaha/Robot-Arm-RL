using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PositionRandomizer : MonoBehaviour
{
    [SerializeField] GameObject table;
    [SerializeField] GameObject arm;
    [SerializeField] GameObject armTip;
    Rigidbody targetRigidBody;

    // float radius = Mathf.Sqrt(Mathf.Pow(0.485f,2) + Mathf.Pow(-0.17f, 2));
    float radius;
    float minRadius;
    float targetRadius;
    Vector3 armHeight;
    Vector3 center;

    Bounds tableBounds;
    Bounds armBase;

    private void Start() {
        targetRigidBody = transform.GetComponent<Rigidbody>();
        targetRadius = transform.GetComponent<Collider>().bounds.extents.x;
        tableBounds = table.GetComponent<Collider>().bounds;
        armBase = arm.GetComponent<Collider>().bounds;
        center = armBase.center;
        // radius = tableBounds.extents.z;
        armHeight = armTip.transform.position - arm.transform.position;
        radius = armHeight.y;
        minRadius = armBase.extents.x + targetRadius;
    }

    public void Randomize()
    {
        // Reset Target Motion
        targetRigidBody.angularVelocity = Vector3.zero;
        targetRigidBody.velocity = Vector3.zero;

        // Randomize Target Position

        Vector3 offset = GetRandomOffset();
        Vector3 point = center + offset;
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

    private Vector3 GetRandomOffset()
    {
        // Vector2 offset2D = Random.insideUnitCircle * radius;
        // float offsetY = Random.value * armHeight.y;
        // Vector3 offset = new Vector3(offset2D.x, offsetY, offset2D.y);
        // offset.y = Mathf.Abs(offset.y);

        while (true)
        {
            Vector3 offset = Random.insideUnitSphere * radius;
            if ((offset.y > targetRadius) && (Mathf.Abs(offset.x) > minRadius) && (Mathf.Abs(offset.z) > minRadius))
            {
                return offset;
            }
            
        }

    }

}
