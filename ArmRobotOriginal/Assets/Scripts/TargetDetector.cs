using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TargetDetector : MonoBehaviour
{
    public bool hasTouchedTarget;
    public bool hasTouchedTable;
    public string targetTag = "Target";
    public string tableTag = "Table";

    public void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            hasTouchedTarget = true;
        }

        if (collision.gameObject.CompareTag(tableTag))
        {
            hasTouchedTable = true;
        }
    }

    public void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.CompareTag(tableTag))
        {
            hasTouchedTable = false;
        }
        if (collision.gameObject.CompareTag(targetTag))
        {
            hasTouchedTarget = false;
        }
    }
}
