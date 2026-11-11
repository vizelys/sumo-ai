using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// A sumo fighter. It can drive forward/backward and turn.
/// Goal: push the opponent out of the ring without falling out itself.
/// Rewards are given by SumoArena (+1 win, -1 loss, 0 draw).
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class SumoAgent : Agent
{
    [Header("Movement")]
    [Tooltip("Push force applied when driving (Newtons).")]
    public float moveForce = 25f;
    [Tooltip("Turning speed in degrees per second.")]
    public float turnSpeed = 200f;
    [Tooltip("Top speed in m/s.")]
    public float maxSpeed = 6f;

    // Wired automatically by SumoArena.
    [HideInInspector] public SumoArena arena;
    [HideInInspector] public SumoAgent opponent;
    [HideInInspector] public Rigidbody rb;

    BehaviorParameters behaviorParams;

    public int TeamId => behaviorParams != null ? behaviorParams.TeamId : 0;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        behaviorParams = GetComponent<BehaviorParameters>();
        // Stay upright: the fighter only turns around the vertical axis.
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        // The arena decides when a round ends, not the agent.
        MaxStep = 0;
    }

    // ---------------------------------------------------------------- senses
    // 13 numbers. Everything is expressed relative to the agent ("in front of me",
    // "to my left"), which makes learning much easier than world coordinates.
    public override void CollectObservations(VectorSensor sensor)
    {
        if (arena == null || opponent == null)
        {
            for (int i = 0; i < 13; i++) sensor.AddObservation(0f);
            return;
        }

        float r = arena.ringRadius;
        Vector3 center = arena.ringCenter.position;

        Vector3 toCenter = ToLocal(center - transform.position) / r;
        Vector3 myVel = ToLocal(GetVelocity(rb)) / maxSpeed;
        Vector3 toOpponent = ToLocal(opponent.transform.position - transform.position) / (2f * r);
        Vector3 oppVel = ToLocal(GetVelocity(opponent.rb)) / maxSpeed;
        Vector3 oppFacing = ToLocal(opponent.transform.forward);

        sensor.AddObservation(toCenter.x);     // where is the middle of the ring?
        sensor.AddObservation(toCenter.z);
        sensor.AddObservation(myVel.x);        // how am I moving?
        sensor.AddObservation(myVel.z);
        sensor.AddObservation(toOpponent.x);   // where is the opponent?
        sensor.AddObservation(toOpponent.z);
        sensor.AddObservation(oppVel.x);       // how is the opponent moving?
        sensor.AddObservation(oppVel.z);
        sensor.AddObservation(oppFacing.x);    // is the opponent facing me?
        sensor.AddObservation(oppFacing.z);
        sensor.AddObservation(arena.EdgeDistance01(transform.position));          // 1 = center, 0 = edge
        sensor.AddObservation(arena.EdgeDistance01(opponent.transform.position));
        sensor.AddObservation(rb.angularVelocity.y / 10f);                        // am I spinning?
    }

    // --------------------------------------------------------------- actions
    // Action 0: drive (-1 = backward, +1 = forward)
    // Action 1: turn  (-1 = left,     +1 = right)
    public override void OnActionReceived(ActionBuffers actions)
    {
        float move = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float turn = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        Quaternion spin = Quaternion.Euler(0f, turn * turnSpeed * Time.fixedDeltaTime, 0f);
        rb.MoveRotation(rb.rotation * spin);

        float forwardSpeed = Vector3.Dot(GetVelocity(rb), transform.forward);
        bool canAccelerate = (move > 0f && forwardSpeed < maxSpeed) ||
                             (move < 0f && forwardSpeed > -maxSpeed);
        if (canAccelerate)
        {
            rb.AddForce(transform.forward * (move * moveForce), ForceMode.Force);
        }
    }

    // ------------------------------------------------------ manual controls
    // Set "Behavior Type" to "Heuristic Only" to drive the fighters yourself.
    // Team 0: W A S D    Team 1: arrow keys
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var a = actionsOut.ContinuousActions;
        bool playerOne = TeamId == 0;
        float move = 0f, turn = 0f;

#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb != null)
        {
            if (playerOne)
            {
                move = (kb.wKey.isPressed ? 1f : 0f) - (kb.sKey.isPressed ? 1f : 0f);
                turn = (kb.dKey.isPressed ? 1f : 0f) - (kb.aKey.isPressed ? 1f : 0f);
            }
            else
            {
                move = (kb.upArrowKey.isPressed ? 1f : 0f) - (kb.downArrowKey.isPressed ? 1f : 0f);
                turn = (kb.rightArrowKey.isPressed ? 1f : 0f) - (kb.leftArrowKey.isPressed ? 1f : 0f);
            }
        }
#else
        if (playerOne)
        {
            move = (Input.GetKey(KeyCode.W) ? 1f : 0f) - (Input.GetKey(KeyCode.S) ? 1f : 0f);
            turn = (Input.GetKey(KeyCode.D) ? 1f : 0f) - (Input.GetKey(KeyCode.A) ? 1f : 0f);
        }
        else
        {
            move = (Input.GetKey(KeyCode.UpArrow) ? 1f : 0f) - (Input.GetKey(KeyCode.DownArrow) ? 1f : 0f);
            turn = (Input.GetKey(KeyCode.RightArrow) ? 1f : 0f) - (Input.GetKey(KeyCode.LeftArrow) ? 1f : 0f);
        }
#endif

        a[0] = move;
        a[1] = turn;
    }

    // --------------------------------------------------------------- helpers
    Vector3 ToLocal(Vector3 worldVector)
    {
        worldVector.y = 0f;
        Vector3 local = transform.InverseTransformDirection(worldVector);
        local.y = 0f;
        return local;
    }

    // Unity 6 renamed Rigidbody.velocity to linearVelocity.
    public static Vector3 GetVelocity(Rigidbody body)
    {
#if UNITY_6000_0_OR_NEWER
        return body.linearVelocity;
#else
        return body.velocity;
#endif
    }

    public static void StopRigidbody(Rigidbody body)
    {
#if UNITY_6000_0_OR_NEWER
        body.linearVelocity = Vector3.zero;
#else
        body.velocity = Vector3.zero;
#endif
        body.angularVelocity = Vector3.zero;
    }
}
