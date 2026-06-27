/// Standalone BATRunner — no Unity dependencies.
/// Reads gesture JSON, extracts 73 plumb-line features per frame,
/// runs 4 ONNX models, writes gesture-level CSV.
///
/// Build:  dotnet build -c Release
/// NuGet:  Newtonsoft.Json, Microsoft.ML.OnnxRuntime, ShellProgressBar

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;
using ShellProgressBar;

public static class BATRunner
{
    private static readonly string[] ModelFileNames =
    {
        "Models/best_model_Space_multilabel.onnx",
        "Models/best_model_Spine_multilabel.onnx",
        "Models/best_model_LE_multilabel.onnx",
        "Models/best_model_FS_multilabel.onnx"
    };

    private const int ExpectedFeatureCount = 73;
    private static bool DumpModelOutput, WholeFolder, AccuracyTest, Verbose = true, Cuda;
    private static string ResultsDirectory;
    private static InferenceSession _spaceSession, _spineSession, _limbSession, _floorSession;

    public static async Task Main(string[] args)
    {
        string gesturePath = null, directoryPath = null;
        bool verboseArg = false;

        foreach (string arg in args)
        {
            switch (arg)
            {
                case "-D": DumpModelOutput = true; break;
                case "-F": WholeFolder = true; break;
                case "-A": AccuracyTest = true; break;
                case "-V": verboseArg = true; break;
                case "-C": Cuda = true; break;
                default:
                    if (File.Exists(arg) && arg.Contains(".json")) gesturePath = arg;
                    else if (Directory.Exists(arg)) directoryPath = arg;
                    break;
            }
        }

        if (WholeFolder && directoryPath != null)
            gesturePath = Path.Combine(directoryPath, "bat_gestures.json");
        else if (string.IsNullOrEmpty(gesturePath))
            gesturePath = Path.Combine(DefaultGestureLocation, "bat_gestures.json");

        Verbose = !WholeFolder || verboseArg;
        ResultsDirectory = Path.GetDirectoryName(Path.GetFullPath(gesturePath));

        string exeDir = Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName);
        string[] modelPaths = ModelFileNames.Select(m => Path.Combine(exeDir, m)).ToArray();
        foreach (var mp in modelPaths)
            if (!File.Exists(mp)) throw new FileNotFoundException($"Model not found: {mp}");

        var opts = Cuda ? SessionOptions.MakeSessionOptionWithCudaProvider() : new SessionOptions();
        using (_spaceSession = new InferenceSession(modelPaths[0], opts))
        using (_spineSession  = new InferenceSession(modelPaths[1], opts))
        using (_limbSession   = new InferenceSession(modelPaths[2], opts))
        using (_floorSession  = new InferenceSession(modelPaths[3], opts))
        {
            if (!WholeFolder) RunOnFile(gesturePath);
            else
            {
                string[] files = Directory.GetFiles(ResultsDirectory, "*.json");
                using (var pbar = new ProgressBar(files.Length, "Processing..."))
                    foreach (string f in files) { RunOnFile(f); pbar.Tick(); }
            }
        }
    }

    private static string DefaultGestureLocation =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                     "AppData", "LocalLow", "EML", "LuminAI", "model");

    private static void RunOnFile(string gesturePath)
    {
        LogLow($"Processing: {gesturePath}");
        float[] tensorArray = PreprocessGestureFile(gesturePath);
        float[][] results = RunModels(tensorArray).Result;
        string outPath = WholeFolder
            ? Path.Combine(ResultsDirectory, Path.GetFileNameWithoutExtension(gesturePath) + "_bat_results.csv")
            : Path.Combine(ResultsDirectory, "bat_result.csv");
        WriteResults(outPath, results);
    }

    // ══════════════════════════════════════════════════════════════
    // JSON → 73 plumb-line features per frame
    // ══════════════════════════════════════════════════════════════

    private static float[] PreprocessGestureFile(string path)
    {
        string json = File.ReadAllText(path);
        json = Regex.Replace(json, @"ISODate\(""(.+?)""\)", @"""$1""");
        json = Regex.Replace(json, @"ObjectId\(""(.+?)""\)", @"""$1""");
        json = json.Replace("NaN", "null");

        JObject data = JObject.Parse(json);
        JArray bodyFrames = (JArray)data["bodyFrames"];
        if (bodyFrames == null || bodyFrames.Count == 0)
            throw new InvalidOperationException("No bodyFrames found.");

        var rows = new List<float[]>();
        foreach (var bf in bodyFrames)
        {
            float[] features = ProcessBodyFrame((JObject)bf);
            if (features != null) rows.Add(features);
        }

        var tensor = new List<float>(rows.Count * ExpectedFeatureCount);
        foreach (var row in rows) tensor.AddRange(row);
        return tensor.ToArray();
    }

    // COG weights
    private static readonly Dictionary<int, float> CogWeights = new()
    {
        {0,0.07f}, {7,0.25f}, {8,0.25f}, {1,0.075f}, {2,0.075f},
        {3,0.075f}, {4,0.075f}, {5,0.035f}, {6,0.035f}, {11,0.03f},
        {12,0.03f}, {13,0.03f}, {14,0.03f}, {15,0.02f}, {16,0.02f},
        {17,0.02f}, {18,0.02f}
    };
    private const float CogDefaultWeight = 0.01f;

    // Distance names → bone keys
    private static readonly Dictionary<int, string> DistanceMapping = new()
    {
        {0,"Hips_to_plumbline"},{1,"LeftUpperLeg_to_plumbline"},{2,"RightUpperLeg_to_plumbline"},
        {3,"LeftLowerLeg_to_plumbline"},{4,"RightLowerLeg_to_plumbline"},{5,"LeftFoot_to_plumbline"},
        {6,"RightFoot_to_plumbline"},{7,"Spine_to_plumbline"},{8,"Chest_to_plumbline"},
        {9,"Neck_to_plumbline"},{10,"Head_to_plumbline"},{11,"LeftShoulder_to_plumbline"},
        {12,"RightShoulder_to_plumbline"},{13,"LeftUpperArm_to_plumbline"},{14,"RightUpperArm_to_plumbline"},
        {15,"LeftLowerArm_to_plumbline"},{16,"RightLowerArm_to_plumbline"},{17,"LeftHand_to_plumbline"},
        {18,"RightHand_to_plumbline"},{19,"LeftToes_to_plumbline"},{20,"RightToes_to_plumbline"},
        {21,"UpperChest_to_plumbline"}
    };

    private static readonly string[] DistanceOrder =
    {
        "Chest_to_plumbline","Head_to_plumbline","Hips_to_plumbline","LeftFoot_to_plumbline",
        "LeftHand_to_plumbline","LeftLowerArm_to_plumbline","LeftLowerLeg_to_plumbline",
        "LeftShoulder_to_plumbline","LeftToes_to_plumbline","LeftUpperArm_to_plumbline",
        "LeftUpperLeg_to_plumbline","Neck_to_plumbline","RightFoot_to_plumbline",
        "RightHand_to_plumbline","RightLowerArm_to_plumbline","RightLowerLeg_to_plumbline",
        "RightShoulder_to_plumbline","RightToes_to_plumbline","RightUpperArm_to_plumbline",
        "RightUpperLeg_to_plumbline","Spine_to_plumbline"
    };

    private static float[] ProcessBodyFrame(JObject bf)
    {
        JArray posArray = (JArray)bf["bodyFrameHumanPos"];
        if (posArray == null) return null;

        var positions = new Dictionary<int, Vector3>();
        foreach (var e in posArray)
        {
            int k = (int)e["k"];
            JArray v = (JArray)e["v"];
            positions[k] = new Vector3((float)v[0], (float)v[1], (float)v[2]);
        }

        Vector3 cog = Vector3.Zero;
        float totalW = 0f;
        foreach (var kv in positions)
        {
            float w = CogWeights.GetValueOrDefault(kv.Key, CogDefaultWeight);
            cog += kv.Value * w;
            totalW += w;
        }
        if (totalW > 0f) cog /= totalW;

        Vector3 up = Vector3.UnitY;
        var plumbDist = new Dictionary<int, float>();
        var angles = new Dictionary<int, float>();

        foreach (var kv in positions)
        {
            Vector3 jtc = kv.Value - cog;
            plumbDist[kv.Key] = Vector3.Cross(jtc, up).Length();

            float len = jtc.Length();
            float angle = 0f;
            if (len > 1e-10f)
            {
                float cosA = Math.Clamp(Vector3.Dot(jtc, up) / len, -1f, 1f);
                angle = MathF.Acos(cosA) * (180f / MathF.PI);
            }
            angles[kv.Key] = angle;
        }

        var namedDist = new Dictionary<string, float>();
        foreach (var kv in DistanceMapping)
            if (plumbDist.TryGetValue(kv.Key, out float d))
                namedDist[kv.Value] = d;

        var row = new List<float>(ExpectedFeatureCount);
        foreach (string name in DistanceOrder)
            row.Add(namedDist.GetValueOrDefault(name, 0f));
        foreach (int bone in angles.Keys.OrderBy(k => k))
            row.Add(angles[bone]);

        if (row.Count != ExpectedFeatureCount)
            throw new InvalidOperationException($"Expected {ExpectedFeatureCount} features, got {row.Count}");
        return row.ToArray();
    }

    // ══════════════════════════════════════════════════════════════
    // ONNX inference
    // ══════════════════════════════════════════════════════════════

    private static async Task<float[][]> RunModels(float[] tensorArray)
    {
        int rows = tensorArray.Length / ExpectedFeatureCount;
        var tensor = new DenseTensor<float>(tensorArray, new[] { rows, ExpectedFeatureCount, 1 });

        var t0 = RunOnnxModel(_spaceSession, tensor);
        var t1 = RunOnnxModel(_spineSession, tensor);
        var t2 = RunOnnxModel(_limbSession, tensor);
        var t3 = RunOnnxModel(_floorSession, tensor);

        float[][] results = await Task.WhenAll(t0, t1, t2, t3);
        return new[] { results[0], results[1], results[2], results[3] };
    }

    private static async Task<float[]> RunOnnxModel(InferenceSession session, DenseTensor<float> input)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(session.InputNames[0], input)
        };

        using (var results = session.Run(inputs))
        {
            var output = results[0].AsTensor<float>();
            int nCodes = output.Dimensions[1];
            LogLow($"Model output: (1, {nCodes})");
            return output.ToArray();
        }
    }

    // ══════════════════════════════════════════════════════════════
    // CSV output  —  gesture-level, single row per category
    // ══════════════════════════════════════════════════════════════

    public static void WriteResults(string resultsPath, float[][] results)
    {
        string[] categories = { "space", "spine", "limb", "floor" };
        using (var sw = new StreamWriter(resultsPath))
            for (int i = 0; i < categories.Length; i++)
                sw.WriteLine(categories[i] + "," + string.Join(",", results[i].Select(f => f.ToString("G9"))));
        LogLow("Results → " + resultsPath);
    }

    // ══════════════════════════════════════════════════════════════
    // Logging
    // ══════════════════════════════════════════════════════════════

    public static void Log(string s) => Console.WriteLine(s);
    public static void LogLow(string s) { if (Verbose) Console.WriteLine(s); }
    public static void LogWarning(string s) => Console.WriteLine("WARNING: " + s);
    public static void LogError(string s) => Console.WriteLine("ERROR: " + s);
}
