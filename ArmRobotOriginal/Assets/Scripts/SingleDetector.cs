using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SingleDetector : MonoBehaviour
{
    public bool hasTouched;

    // public string targetTag;

    public List<string> targetTags;


    public void OnCollisionEnter(Collision collision)
    {
        foreach (string tag in targetTags)
        {
            if (collision.gameObject.CompareTag(tag))
            {
                hasTouched = true;
                // Debug.Log(gameObject.name + " has touched the object " + collision.gameObject.name);
            }            
        }
    }

    public void OnCollisionExit(Collision collision)
    {
        foreach (string tag in targetTags)
        {
            if (collision.gameObject.CompareTag(tag))
            {
                hasTouched = false;
            }            
        }

    }
}
