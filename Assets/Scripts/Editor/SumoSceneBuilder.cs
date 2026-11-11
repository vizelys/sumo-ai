#if UNITY_EDITOR
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Adds a "Sumo" menu to the Unity menu bar that builds the whole scene in one click:
/// ring, two fighters with eyes, all ML-Agents components configured, camera and scoreboard.
///
///   Sumo > Build Scene (1 arena)    -> to test with the keyboard and to record footage
///   Sumo > Build Scene (12 arenas)  -> to train faster (12 fights at the same time)
///
/// The scene is saved to Assets/Scenes/Sumo.unity (or Sumo Training.unity).
/// </summary>
public static class SumoSceneBuilder
{
    const float ArenaSpacing = 20f;
    const float RingRadius = 5f;

    [MenuItem("Sumo/Build Scene (1 arena - test and record)")]
    static void BuildSingle() => Build(1, "Sumo");

    [MenuItem("Sumo/Build Scene (12 arenas - training)")]
    static void BuildTraining() => Build(12, "Sumo Training");

    static void Build(int arenaCount, string sceneName)
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        var mats = new Materials
        {
            ring = MakeMaterial("Sumo Ring", new Color(0.93f, 0.82f, 0.60f)),
            blue = MakeMaterial("Sumo Blue", new Color(0.15f, 0.45f, 0.95f)),
            red = MakeMaterial("Sumo Red", new Color(0.92f, 0.22f, 0.20f)),
            white = MakeMaterial("Sumo Eye White", Color.white),
            black = MakeMaterial("Sumo Eye Pupil", new Color(0.05f, 0.05f, 0.05f)),
        };

        int columns = Mathf.CeilToInt(Mathf.Sqrt(arenaCount));
        for (int i = 0; i < arenaCount; i++)
        {
            var position = new Vector3((i % columns) * ArenaSpacing, 0f, (i / columns) * ArenaSpacing);
            var arena = BuildArena(i, position, mats);
            arena.showScoreboard = i == 0;   // only one scoreboard on screen
        }

        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(0f, 8f, -11f);
            cam.transform.LookAt(new Vector3(0f, 0.5f, 0f));
            cam.fieldOfView = 50f;
        }

        if (!AssetDatabase.IsValidFolder("Assets/Scenes")) AssetDatabase.CreateFolder("Assets", "Scenes");
        string path = $"Assets/Scenes/{sceneName}.unity";
        EditorSceneManager.SaveScene(scene, path);
        AssetDatabase.SaveAssets();

        Debug.Log($"[Sumo] Built {arenaCount} arena(s) and saved the scene to {path}. Press Play!");
    }

    // ------------------------------------------------------------------ arena
    static SumoArena BuildArena(int index, Vector3 position, Materials mats)
    {
        var root = new GameObject($"Arena {index + 1}");
        root.transform.position = position;

        // Ring: a flat disc of radius 5 whose top surface is at y = 0.5.
        var ring = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        ring.name = "Ring";
        ring.transform.SetParent(root.transform, false);
        ring.transform.localScale = new Vector3(RingRadius * 2f, 0.5f, RingRadius * 2f);
        Object.DestroyImmediate(ring.GetComponent<Collider>());      // the default capsule collider is rounded
        ring.AddComponent<MeshCollider>().convex = true;               // flat top, real edge
        ring.GetComponent<Renderer>().sharedMaterial = mats.ring;

        var center = new GameObject("RingCenter");
        center.transform.SetParent(root.transform, false);
        center.transform.localPosition = new Vector3(0f, 0.5f, 0f);

        var blue = BuildFighter("Fighter Blue", root.transform, new Vector3(-2.5f, 1f, 0f), 0, mats.blue, mats);
        var red = BuildFighter("Fighter Red", root.transform, new Vector3(2.5f, 1f, 0f), 1, mats.red, mats);

        var arena = root.AddComponent<SumoArena>();
        arena.ringCenter = center.transform;
        arena.ringRadius = RingRadius;
        arena.agentA = blue;
        arena.agentB = red;
        return arena;
    }

    // ---------------------------------------------------------------- fighter
    static SumoAgent BuildFighter(string name, Transform parent, Vector3 localPosition, int teamId,
                                  Material bodyMaterial, Materials mats)
    {
        var body = GameObject.CreatePrimitive(PrimitiveType.Cube);
        body.name = name;
        body.transform.SetParent(parent, false);
        body.transform.localPosition = localPosition;
        Vector3 toCenter = new Vector3(-localPosition.x, 0f, -localPosition.z);
        body.transform.localRotation = Quaternion.LookRotation(toCenter.normalized, Vector3.up);
        body.GetComponent<Renderer>().sharedMaterial = bodyMaterial;

        var rb = body.AddComponent<Rigidbody>();
        rb.mass = 1f;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        // Adding the agent also adds Behavior Parameters automatically.
        var agent = body.AddComponent<SumoAgent>();
        var bp = body.GetComponent<BehaviorParameters>();
        if (bp == null) bp = body.AddComponent<BehaviorParameters>();
        bp.BehaviorName = "Sumo";                                   // must match config/sumo.yaml
        bp.TeamId = teamId;                                         // 0 = blue, 1 = red (self-play)
        bp.BehaviorType = BehaviorType.Default;
        bp.BrainParameters.VectorObservationSize = 13;
        bp.BrainParameters.NumStackedVectorObservations = 3;
        bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);

        var requester = body.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = 5;
        requester.TakeActionsBetweenDecisions = true;

        AddEye(body.transform, -0.22f, mats);
        AddEye(body.transform, 0.22f, mats);
        return agent;
    }

    static void AddEye(Transform body, float x, Materials mats)
    {
        var eye = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        eye.name = "Eye";
        Object.DestroyImmediate(eye.GetComponent<Collider>());   // visual only, no physics
        eye.transform.SetParent(body, false);
        eye.transform.localPosition = new Vector3(x, 0.2f, 0.5f);
        eye.transform.localScale = Vector3.one * 0.28f;
        eye.GetComponent<Renderer>().sharedMaterial = mats.white;

        var pupil = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        pupil.name = "Pupil";
        Object.DestroyImmediate(pupil.GetComponent<Collider>());
        pupil.transform.SetParent(eye.transform, false);
        pupil.transform.localPosition = new Vector3(0f, 0f, 0.38f);
        pupil.transform.localScale = Vector3.one * 0.5f;
        pupil.GetComponent<Renderer>().sharedMaterial = mats.black;
    }

    // -------------------------------------------------------------- materials
    struct Materials { public Material ring, blue, red, white, black; }

    static Material MakeMaterial(string name, Color color)
    {
        if (!AssetDatabase.IsValidFolder("Assets/Materials")) AssetDatabase.CreateFolder("Assets", "Materials");
        string path = $"Assets/Materials/{name}.mat";

        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (mat == null)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            mat = new Material(shader);
            AssetDatabase.CreateAsset(mat, path);
        }

        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);   // URP
        if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);           // Built-in
        EditorUtility.SetDirty(mat);
        return mat;
    }
}
#endif
