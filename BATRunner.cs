// Decompiled with JetBrains decompiler
// Type: BATRunner
// Assembly: BATRunner, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null
// MVID: 80EBE4E7-EE3F-4407-911E-4871BEB8C488
// Assembly location: BATRunner.dll inside D:\Documents\GitHub\LuminAI-4\BATRunner\Release\BATRunner.exe)

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MongoDB.Bson.Serialization;
using MongoDB.Bson.Serialization.Attributes;
using MongoDB.Bson.Serialization.Serializers;
using MovementTheory;
using ShellProgressBar;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Quaternion = UnityEngine.Quaternion;
using Vector3 = UnityEngine.Vector3;

public class BATRunner
{
    private static string[] modelFileNames = new string[4]
    {
    "Models/best_model_Space_multilabel.onnx",
    "Models/best_model_Spine_multilabel.onnx",
    "Models/best_model_LE_multilabel.onnx",
    "Models/best_model_FS_multilabel.onnx"
    };
    private const int EXPECTED_MODEL_FEATURE_COUNT = 73;

    static bool DumpModelOutput = false; // -D flag to dump raw model outputs to CSV for debugging/analysis
    static string resultsDirectory;
    static bool WholeFolder = false; // -F flag to indicate whether to process a whole folder of gesture files instead of just one
    static bool AccuracyTest = false; // -A flag, trun in accuracy test mode, comparing outputs to BAT codes in the gesture files
    static bool Verbose = true; // -V flag for very verbose logging
    static bool Cuda = false; // -C flag to attempt to use CUDA for ONNX inference (will fall back to CPU if CUDA is not available or fails to load)


    static InferenceSession spaceSession;
    static InferenceSession spineSession;
    static InferenceSession limbSession;
    static InferenceSession floorSession;

    private static string DefaultGestureLocation
    {
        get
        {
            return Path.GetFullPath(((FileSystemInfo)Directory.GetParent(Path.GetFullPath(Environment.GetFolderPath((Environment.SpecialFolder)28)))).FullName) + "\\LocalLow\\EML\\LuminAI\\model";
        }
    }

    private static async Task Main(string[] args)
    {
        BsonSerializer.RegisterSerializer(typeof(Quaternion), new QuaternionSerializer());
        BsonSerializer.RegisterSerializer(typeof(Vector3), new Vector3Serializer());


        string gesturePath = null;
        string directoryPath = null;
        bool verboseArg = false;
        foreach (string arg in args)
        {
            if (arg == "-D")
            {
                DumpModelOutput = true;
                Log("DumpModelOutput enabled: model outputs will be dumped to CSV in the results directory.");
            }
            else if (arg == "-F")
            {
                WholeFolder = true;
                Log("WholeFolder enabled: all JSON files in the gesture file directory will be processed.");
            }
            else if (arg == "-A")
            {
                AccuracyTest = true;
                Log("AccuracyTest enabled: model outputs will be compared to BAT codes in the gesture files and accuracy will be logged.");
            }
            else if (arg == "-V")
            {
                verboseArg = true;
            }
            else if (arg == "-C")
            {
                Cuda = true;
                Log("Cuda enabled: the program will attempt to use CUDA for ONNX inference. If CUDA is not available or fails to load, it will fall back to CPU.");
            }
            else if (System.IO.File.Exists(arg) && arg.Contains(".json"))
            {
                gesturePath = arg;
                Log("Gesture path argument provided: " + gesturePath);
            }
            else if (System.IO.Directory.Exists(arg))
            {
                directoryPath = arg;
                Log("Valid directory provided, using if WholeFolder is enabled: " + directoryPath);
            }
        }

        if (WholeFolder && directoryPath != null)
        {
            gesturePath = Path.Combine(directoryPath, "bat_gestures.json");
            Log("WholeFolder enabled and directory argument provided, looking for gesture files in: " + directoryPath);
        }
        else if (string.IsNullOrEmpty(gesturePath))
        {

            gesturePath = Path.Combine(BATRunner.DefaultGestureLocation, "bat_gestures.json");
            BATRunner.LogWarning("No gesture path provided as argument. Attempting to use default path: " + gesturePath);
        }


        // verbose is on by default in single file mode, but in whole folder mode we disable it by default
        if (WholeFolder && !verboseArg)
        {
            Verbose = false;
        }
        else if (verboseArg)
        {
            Verbose = true;
        }

        if (Verbose)
        {
            Log("Verbose logging is enabled. This may produce a large amount of output.");
        }

        resultsDirectory = Path.GetDirectoryName(gesturePath);

        // start the model sessions




        string spaceModel = Path.Combine(Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName), BATRunner.modelFileNames[0]);
        string spineModel = Path.Combine(Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName), BATRunner.modelFileNames[1]);
        string limbModel = Path.Combine(Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName), BATRunner.modelFileNames[2]);
        string floorModel = Path.Combine(Path.GetDirectoryName(Process.GetCurrentProcess().MainModule.FileName), BATRunner.modelFileNames[3]);

        LogLow("Model paths:");
        if (!File.Exists(spaceModel))
        {
            throw new FileNotFoundException("Space model not found at " + spaceModel);
        }
        if (!IsModelValid(spaceModel))
        {
            throw new Exception("Space model is not valid.");
        }
        LogLow("Space: " + spaceModel);

        if (!File.Exists(spineModel))
        {
            throw new FileNotFoundException("Spine model not found at " + spineModel);
        }
        if (!IsModelValid(spineModel))
        {
            throw new Exception("Spine model is not valid.");
        }
        LogLow("Spine: " + spineModel);

        if (!File.Exists(limbModel))
        {
            throw new FileNotFoundException("Limb model not found at " + limbModel);
        }
        if (!IsModelValid(limbModel))
        {
            throw new Exception("Limb model is not valid.");
        }
        LogLow("Limb: " + limbModel);


        if (!File.Exists(floorModel))
        {
            throw new FileNotFoundException("Floor model not found at " + floorModel);
        }
        if (!IsModelValid(floorModel))
        {
            throw new Exception("Floor model is not valid.");
        }
        LogLow("Floor: " + floorModel);

        int maxTasksRunning = 32;
        var semaphore = new System.Threading.SemaphoreSlim(maxTasksRunning, maxTasksRunning);
        using (spaceSession = new InferenceSession(spaceModel, Cuda ? SessionOptions.MakeSessionOptionWithCudaProvider() : new SessionOptions()))
        using (spineSession = new InferenceSession(spineModel, Cuda ? SessionOptions.MakeSessionOptionWithCudaProvider() : new SessionOptions()))
        using (limbSession = new InferenceSession(limbModel, Cuda ? SessionOptions.MakeSessionOptionWithCudaProvider() : new SessionOptions()))
        using (floorSession = new InferenceSession(floorModel, Cuda ? SessionOptions.MakeSessionOptionWithCudaProvider() : new SessionOptions()))
        {

            if (!WholeFolder)
            {
                RunOnFile(gesturePath);
            }
            else
            {

                string[] files = Directory.GetFiles(resultsDirectory, "*.json");
                Log("Found " + files.Length + " JSON files in directory. Processing all files in directory due to WholeFolder flag:");

                using (var pbar = new ProgressBar(files.Length, "Processing gesture files in directory...", new ProgressBarOptions
                {
                    ProgressCharacter = '─'
                }))
                {
                    foreach (string file in files)
                    {
                        RunOnFile(file);
                        pbar.Tick();
                    }
                }
                
            }
        }
    }

    static void RunOnFile(string gesturePath)
    {
        float[][] resultArrays = null;

        var result = CategorizeGestureFile(gesturePath);

        result.Wait();

        if (result.IsFaulted)
        {
            LogError("Error: " + result.Exception);
            throw result.Exception;
        }
        else
        {
            LogLow("Results:");
            foreach (var category in result.Result)
            {
                LogLow(string.Join(",", category));
            }
        }

        resultArrays = result.Result;

        string resultsPath;
        if (WholeFolder)
        {
            resultsPath = Path.Combine(resultsDirectory, Path.GetFileNameWithoutExtension(gesturePath) + "_bat_results.csv");
        }
        else
        {
            resultsPath = Path.Combine(resultsDirectory, "bat_results.csv");
        }

        WriteResults(Path.Combine(resultsDirectory, "bat_result.csv"), resultArrays);
    }


    public static async Task<float[][]> CategorizeGestureFile(string pathToGesture)
    {
        /*
        string tensorPath;
        if (args.Length != 1)
        {
            BATRunner.LogWarning("No tensor file path provided as argument. Attempting to use default path: " + BATRunner.DefaultTensorLocation);
            tensorPath = BATRunner.DefaultTensorLocation;
        }
        else
            tensorPath = args[0];
        */

        string g = File.ReadAllText(pathToGesture);
        string tensorPath = BATRunner.DefaultGestureLocation;

        float[] tensor_array = BATPreprocess.GetTensorArray(g);
        Task<float[][]> results = BATRunner.RunModels(tensor_array);
        await results;
        return results.Result;
    }

    private static async Task<float[][]> RunModels(float[] tensor_array)
    {
        //float[] tensor_array = Enumerable.ToArray<float>(Enumerable.Select<string, float>((IEnumerable<string>)tensorString.Split(','), (Func<string, float>)(s => float.Parse(s))));
        
        

        int cols = EXPECTED_MODEL_FEATURE_COUNT;
        int rows = tensor_array.Length / cols;

        var input_dimensions = new int[] { rows, cols, 1 };

        LogLow($"input dimensions: ({string.Join(",", input_dimensions)})");

        var sample_features = new DenseTensor<float>(tensor_array, input_dimensions);

        Task<float[]> space = BATRunner.RunOnnxModel(spaceSession, sample_features);
        Task<float[]> spine = BATRunner.RunOnnxModel(spineSession, sample_features);
        Task<float[]> limb = BATRunner.RunOnnxModel(limbSession, sample_features);
        Task<float[]> floor = BATRunner.RunOnnxModel(floorSession, sample_features);

        Task<float[][]> all = Task.WhenAll<float[]>(space, spine, limb, floor);

        await all;

        if (((Task)all).IsFaulted)
            throw ((Task)all).Exception;

        float[][] results = new float[4][]
        {
          space.Result,
          spine.Result,
          limb.Result,
          floor.Result
        };
        return results;
    }

    /// <summary>
    /// Runs a single ONNX model.  Model takes (N, 73, 1) as input and
    /// produces (1, num_codes) as gesture-level output (temporal pooling
    /// is done inside the model).
    /// </summary>
    private static async Task<float[]> RunOnnxModel(InferenceSession session, DenseTensor<float> sample_features)
    {
        var name = session.InputNames[0];
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(name, sample_features) };

        using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
        {
            var outputTensor = results[0].AsTensor<float>();

            // Model outputs (1, num_codes) — squeeze batch dim to get (num_codes,)
            int numCodes = outputTensor.Dimensions[1];
            LogLow($"Model output: (1, {numCodes}) — gesture-level vector");

            var resultsArray = outputTensor.ToList<float>();

            if (DumpModelOutput)
            {
                DumpOutput(resultsDirectory, resultsArray, numCodes);
            }

            return resultsArray.ToArray();
        }
    }

    /// <summary>
    /// Writes gesture-level results CSV.  Each row: category,N values (one per code).
    /// Expected code counts: space=5, spine=6, limb=8, floor=4
    /// </summary>
    public static void WriteResults(string resultsPath, float[][] results)
    {
        using (StreamWriter streamWriter = new StreamWriter(resultsPath))
        {
            string[] categories = { "space", "spine", "limb", "floor" };
            for (int i = 0; i < categories.Length; i++)
            {
                streamWriter.Write(categories[i] + ",");
                streamWriter.Write(string.Join(",", results[i].Select(f => f.ToString("G9"))));
                streamWriter.WriteLine();
            }
        }
        BATRunner.LogLow("Results written to " + resultsPath);
    }

    static void DumpOutput(string modelPath, List<float> flatArray, int columnCount)
    {
        int rowCount = flatArray.Count / columnCount;
        LogLow($"Model output dimensions: ({rowCount},{columnCount})");
        string modelName = Path.GetFileNameWithoutExtension(modelPath);
        using (StreamWriter streamWriter = new StreamWriter(Path.Combine(resultsDirectory, $"{modelName}_output_dump.csv")))
        {
            string[] row = new string[columnCount];
            for (int i = 0; i < flatArray.Count; i++)
            {
                int c = i % columnCount;
                row[c] = flatArray[i].ToString();
                if (c == columnCount - 1)
                {
                     ((TextWriter)streamWriter).WriteLine(string.Join(",", row));
                }
            }
        }
        LogLow("Model output dumped to " + Path.Combine(resultsDirectory, $"{modelName}_output_dump.csv"));
    }

    static float[] AverageColumns(List<float> flatArray, int columnCount)
    {
        int rowCount = flatArray.Count / columnCount;

        var averages = new float[columnCount];

        LogLow($"pre-average dimensions: ({rowCount},{columnCount})");

        // flatten all frames into one row by averaging each column across all rows
        for (int i = 0; i < flatArray.Count; i++)
        {
            int r = (int)(Math.Floor((double)i / (double)rowCount)) - 1;
            int c = i % columnCount;

            float value = flatArray[i];

            float averagedValue = value / rowCount;

            averages[c] += averagedValue;
        }

        return averages;
    }

    static bool IsModelValid(string modelPath)
    {
        try
        {
            // Creating the session attempts to load and validate the model structure.
            using (var session = new InferenceSession(modelPath))
            {
                // If it creates without error, it's generally valid for ORT.
                // You can optionally check input/output metadata here.
                // For example: session.InputMetadata["input_name"].ElementType
                return true;
            }
        }
        catch (TypeInitializationException ex) when (ex.InnerException is EntryPointNotFoundException)
        {
            throw new InvalidProgramException("Failed to load entry point, likely because .exe was not compiled as x64. Please ensure the project is set to target x64 architecture.", ex);
        }
        catch (OnnxRuntimeException ex)
        {
            // Catches loading/structural errors specific to ORT.
            LogError($"Model loading failed: ");
            throw;
        }
        catch (Exception ex)
        {
            // Catches other general errors.
            LogError($"An unexpected error occurred:");
            LogError(ex.ToString());
            throw;
        }
    }

    public static void Log(string s)
    {
        System.Console.WriteLine(s);
    }

    public static void LogLow(string s) // for very verbose logging
    {
        if (Verbose)
            System.Console.WriteLine(s);
    }
    public static void LogWarning(string s)
    {
        System.Console.WriteLine("WARNING: " + s);
    }

    public static void LogError(string s)
    {
        System.Console.WriteLine("ERROR: " + s);
    }

    private enum ModelType
    {
        Space,
        Spine,
        Limb,
        Floor,
    }

    #region Serializers
    // serializer classes for Quaternion and Vector3
    [BsonSerializer(typeof(Quaternion))]
    public class QuaternionSerializer : StructSerializerBase<Quaternion>
    {
        public override Quaternion Deserialize(BsonDeserializationContext context, BsonDeserializationArgs args)
        {
            context.Reader.ReadStartArray();
            var x = (float)context.Reader.ReadDouble();
            var y = (float)context.Reader.ReadDouble();
            var z = (float)context.Reader.ReadDouble();
            var w = (float)context.Reader.ReadDouble();
            context.Reader.ReadEndArray();
            return new Quaternion(x, y, z, w);
        }

        public override void Serialize(BsonSerializationContext context, BsonSerializationArgs args, Quaternion value)
        {
            context.Writer.WriteStartArray();
            context.Writer.WriteDouble(value.x);
            context.Writer.WriteDouble(value.y);
            context.Writer.WriteDouble(value.z);
            context.Writer.WriteDouble(value.w);
            context.Writer.WriteEndArray();
        }

        public Type ValueType
        {
            get { return typeof(Quaternion); }
        }
    }

    [BsonSerializer(typeof(Vector3))]
    public class Vector3Serializer : StructSerializerBase<Vector3>
    {
        public override Vector3 Deserialize(BsonDeserializationContext context, BsonDeserializationArgs args)
        {
            context.Reader.ReadStartArray();
            var x = (float)context.Reader.ReadDouble();
            var y = (float)context.Reader.ReadDouble();
            var z = (float)context.Reader.ReadDouble();
            context.Reader.ReadEndArray();
            return new Vector3(x, y, z);
        }

        public override void Serialize(BsonSerializationContext context, BsonSerializationArgs args, Vector3 value)
        {

            context.Writer.WriteStartArray();
            context.Writer.WriteDouble(value.x);
            context.Writer.WriteDouble(value.y);
            context.Writer.WriteDouble(value.z);
            context.Writer.WriteEndArray();
        }

        public Type ValueType
        {
            get { return typeof(Vector3); }
        }
    }
        #endregion

}
