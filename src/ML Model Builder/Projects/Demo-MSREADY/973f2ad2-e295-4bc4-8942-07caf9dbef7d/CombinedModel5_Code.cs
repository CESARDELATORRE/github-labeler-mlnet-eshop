using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace MLGeneratedCode
{
public class Program
{
/// <summary>
/// This is the input to the trained model.
///
/// In most pipelines, not all columns that are used in training are also used in scoring. Namely, the label
/// and weight columns are almost never required at scoring time. Since we don't know which columns
/// are 'optional' in this sense, all the columns are listed below.
///
/// You are free to remove any fields from the below class. If the fields are not required for scoring, the model
/// will continue to work. Otherwise, the exception will be thrown when a prediction engine is created.
///
/// </summary>
public class InputData
{
            public Single ID;

            public string Area;

            public string Title;

            public string Description;
}

/// <summary>
/// This is the output of the scored model, the prediction.
///
///</summary>
public class ScoredOutput
{
            [KeyType(Count=22, Min=0, Contiguous=true)]
            public UInt32 PredictedLabel;

            [VectorType(22)]
            public Single[] Score;
}

/*public static void Main(string[] args)
{
string modelPath;
modelPath = "D:\\DevSpikes\\GitHub-Issues-Labeler-eShopOnContainers\\ML Model Builder\\Projects\\De" +
    "mo-MSREADY\\973f2ad2-e295-4bc4-8942-07caf9dbef7d\\CombinedModel5.zip";
PredictAsync(modelPath);
}*/

/// <summary>
/// This method demonstrates how to run prediction.
///
///</summary>
public static void Predict(string modelPath)
{
    var model = await PredictionModel.ReadAsync<InputData, ScoredOutput>(modelPath);

    var inputData = new InputData();
    // TODO: populate the example's features.

    var score = model.Predict(inputData);
    // TODO: consume the resulting score.

    var scores = model.Predict(new List<InputData> { inputData, inputData });
    // TODO: consume the resulting scores.
  }
} 
}
