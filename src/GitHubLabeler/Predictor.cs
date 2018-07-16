using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace GitHubLabeler
{
    internal class Predictor
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string TrainDataPath => Path.Combine(AppPath, "datasets", "corefx-issues-train-data.tsv");
        //private static string TrainDataPath => Path.Combine(AppPath, "datasets", "corefx-issues-full-data.tsv");
        private static string EvalDataPath => Path.Combine(AppPath, "datasets", "corefx-issues-eval-data.tsv");

        private static string ModelPath => Path.Combine(AppPath, "TrainedModels", "GitHubLabelerModel.zip");
        private static string PreTrainedModelPath => Path.Combine(AppPath, 
                                                                  "TrainedModels",
                                                                  //"GitHubLabelerPreTrainedModel_FromModelBuilder.zip");
                                                                  "GitHubLabelerPreTrainedModel_FromCode.zip");
        
        private static PredictionModel<GitHubIssue, GitHubIssuePrediction> _modelFromFile;

        public static async Task TrainAsync()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(TrainDataPath).CreateFrom<GitHubIssue>());

            pipeline.Add(new Dictionarizer(("Area", "Label")));

            pipeline.Add(new TextFeaturizer("Title", "Title"));

            pipeline.Add(new TextFeaturizer("Description", "Description"));
            
            pipeline.Add(new ColumnConcatenator("Features", "Title", "Description"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            Console.WriteLine("=============== Training the model ===============");

            var model = pipeline.Train<GitHubIssue, GitHubIssuePrediction>();

            await model.WriteAsync(ModelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", ModelPath);
            Console.WriteLine("============================================");
        }

        public static async Task<ClassificationMetrics> EvaluateAccuracyAsync()
        {
            if (_modelFromFile == null)
            {
                _modelFromFile = await PredictionModel.ReadAsync<GitHubIssue, GitHubIssuePrediction>(ModelPath);
                //_model = await PredictionModel.ReadAsync<GitHubIssue, GitHubIssuePrediction>(PreTrainedModelPath);
            }

            var testData = new TextLoader(EvalDataPath).CreateFrom<GitHubIssue>();

            var evaluator = new ClassificationEvaluator();
            ClassificationMetrics metrics = evaluator.Evaluate(_modelFromFile, testData);

            Console.WriteLine(" ");
            Console.WriteLine("=============================================================");
            
            return metrics;
        }

        public static async Task<string> PredictAsync(GitHubIssue issue)
        {
            if (_modelFromFile == null)
            {
                //_model = await PredictionModel.ReadAsync<GitHubIssue, GitHubIssuePrediction>(ModelPath);
                _modelFromFile = await PredictionModel.ReadAsync<GitHubIssue, GitHubIssuePrediction>(PreTrainedModelPath);
            }

            var prediction = _modelFromFile.Predict(issue);

            return prediction.Area;
        }
    }
}
