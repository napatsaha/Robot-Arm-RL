using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandDetector : MonoBehaviour
{
    public bool hasTouched;
    public bool badCollision;

    public string targetTag = "Pincher";
    public string negativeTag = "Body";

    public void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            hasTouched = true;
        }
        if (collision.gameObject.CompareTag(negativeTag))
        {
            badCollision = true;
            // Debug.Log("Bad Collision");
        }


    }

    public void OnCollisionExit(Collision collision)
    {

        if (collision.gameObject.CompareTag(targetTag))
        {
            hasTouched = false;
        }

        if (collision.gameObject.CompareTag(negativeTag))
        {
            badCollision = false;
            // Debug.Log("Averted Collision");
        }
    }
}
