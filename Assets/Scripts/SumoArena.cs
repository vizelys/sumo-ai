using UnityEngine;

/// <summary>
/// Referee for one ring. Detects ring-outs, hands out rewards and resets the round.
/// Put one on each arena; duplicate the whole arena to train faster.
/// </summary>
public class SumoArena : MonoBehaviour
{
    [Header("Ring")]
    [Tooltip("Empty object placed at the centre of the ring's TOP surface.")]
    public Transform ringCenter;
    [Tooltip("Radius of the ring in metres (a cylinder scaled 10 on X/Z = radius 5).")]
    public float ringRadius = 5f;

    [Header("Fighters")]
    public SumoAgent agentA;   // Behavior Parameters -> Team Id 0
    public SumoAgent agentB;   // Behavior Parameters -> Team Id 1

    [Header("Rounds")]
    [Tooltip("Physics steps before a round is a draw (50 steps = 1 second).")]
    public int maxEnvironmentSteps = 2000;
    [Tooltip("Distance of each fighter from the centre at the start.")]
    public float spawnDistance = 2.5f;
    [Tooltip("Height of the fighter's centre above the ring surface (0.5 for a 1m cube).")]
    public float spawnHeight = 0.5f;
    [Tooltip("Random rotation added at spawn, in degrees, so rounds differ.")]
    public float spawnAngleJitter = 30f;
    [Tooltip("Total penalty over a full round, shared out per step. Discourages stalling.")]
    public float timePenalty = 0.5f;

    [Header("Scoreboard (for recording)")]
    public bool showScoreboard = false;
    public int winsA, winsB, draws;

    int stepCount;

    void Awake()
    {
        agentA.arena = this;
        agentB.arena = this;
        agentA.opponent = agentB;
        agentB.opponent = agentA;
    }

    void Start()
    {
        ResetRound();
    }

    void FixedUpdate()
    {
        stepCount++;

        bool aOut = IsOut(agentA);
        bool bOut = IsOut(agentB);

        if (aOut && bOut)
        {
            // Both fell out together: draw.
            agentA.SetReward(0f);
            agentB.SetReward(0f);
            draws++;
            FinishRound();
            return;
        }
        if (aOut) { Win(agentB, agentA); winsB++; return; }
        if (bOut) { Win(agentA, agentB); winsA++; return; }

        if (maxEnvironmentSteps > 0 && stepCount >= maxEnvironmentSteps)
        {
            // Time is up: nobody won. "Interrupted" tells the trainer the
            // round was cut short rather than truly finished.
            draws++;
            agentA.EpisodeInterrupted();
            agentB.EpisodeInterrupted();
            ResetRound();
            return;
        }

        // Small cost for every step so that waiting is never the best strategy.
        if (maxEnvironmentSteps > 0 && timePenalty > 0f)
        {
            float p = -timePenalty / maxEnvironmentSteps;
            agentA.AddReward(p);
            agentB.AddReward(p);
        }
    }

    void Win(SumoAgent winner, SumoAgent loser)
    {
        // Self-play reads the LAST reward of a round: +1 win, -1 loss, 0 draw.
        winner.SetReward(1f);
        loser.SetReward(-1f);
        FinishRound();
    }

    void FinishRound()
    {
        agentA.EndEpisode();
        agentB.EndEpisode();
        ResetRound();
    }

    bool IsOut(SumoAgent agent)
    {
        Vector3 offset = agent.transform.position - ringCenter.position;
        if (offset.y < -1f) return true;            // fell below the ring
        offset.y = 0f;
        return offset.magnitude > ringRadius;       // centre went past the edge
    }

    /// <summary>1 at the centre of the ring, 0 at the edge (negative outside).</summary>
    public float EdgeDistance01(Vector3 position)
    {
        Vector3 offset = position - ringCenter.position;
        offset.y = 0f;
        return 1f - offset.magnitude / ringRadius;
    }

    void ResetRound()
    {
        stepCount = 0;
        float angle = Random.Range(0f, 360f);
        Vector3 dir = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;

        Place(agentA, ringCenter.position + dir * spawnDistance, -dir);
        Place(agentB, ringCenter.position - dir * spawnDistance, dir);
    }

    void Place(SumoAgent agent, Vector3 position, Vector3 facing)
    {
        position.y = ringCenter.position.y + spawnHeight;
        float jitter = Random.Range(-spawnAngleJitter, spawnAngleJitter);
        Quaternion rotation = Quaternion.LookRotation(facing, Vector3.up) * Quaternion.Euler(0f, jitter, 0f);

        Rigidbody body = agent.rb != null ? agent.rb : agent.GetComponent<Rigidbody>();
        SumoAgent.StopRigidbody(body);
        agent.transform.SetPositionAndRotation(position, rotation);
        body.position = position;
        body.rotation = rotation;
    }

    void OnGUI()
    {
        if (!showScoreboard) return;
        var style = new GUIStyle(GUI.skin.label) { fontSize = 28, fontStyle = FontStyle.Bold };
        style.normal.textColor = Color.white;
        GUI.Label(new Rect(20, 20, 800, 40), $"BLUE {winsA}  —  {winsB} RED    (draws {draws})", style);
    }
}
