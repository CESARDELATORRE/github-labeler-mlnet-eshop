using System;
using System.Configuration;
using System.Threading.Tasks;

// Requires following NuGet packages
// NuGet: Microsoft.Extensions.Configuration
// NuGet: Microsoft.Extensions.Configuration.Json
using Microsoft.Extensions.Configuration;
using System.IO;



namespace GitHubLabeler
{
    internal static class Program
    {
        public static IConfiguration Configuration { get; set; }
        private static async Task Main(string[] args)
        {
            var builder = new ConfigurationBuilder()
                                        .SetBasePath(Directory.GetCurrentDirectory())
                                        .AddJsonFile("appsettings.json");

            Configuration = builder.Build();

            //Build and Train a new model
            await Predictor.TrainAsync();

            //Evaluate/test accuracy of the just trained model
            var accuracyMetrics = await Predictor.EvaluateAccuracyAsync();
            Console.WriteLine("Micro-Accuracy is {0}", accuracyMetrics.AccuracyMicro);
            
            //Test single issue categorization
            await TestCategorizeIssue();
            
            //Set predicted labels to Issues in GitHub repo
            //await Label();

            Console.WriteLine("Press a key to finish...");
            Console.ReadKey();
        }

        private static async Task<string> TestCategorizeIssue()
        {
            GitHubIssue gitHubIssue = new GitHubIssue
            {
                ID = "99999",
                Title = "Issue with Entity Framework Core and when load testing my app",
                Description = "I'm experiencing and issue with Entity Framework Core and the internal System.Data.SqlConnection. The database is a SQL Server database and it happens only when running on Linux, either as a container or standalone. The connection is failing randomly when accessing a database after scaling out and load testing with more than 100 instances"
            };

            var categorizedLabel = await Predictor.PredictAsync(gitHubIssue);

            Console.WriteLine("============================================================");
            Console.WriteLine(" ");
            Console.WriteLine("=============== Single Test of Categorization ===============");
            Console.WriteLine("The label/area predicted for the single hard-coded issue is '{0}'", categorizedLabel);
            Console.WriteLine("=============================================================");
            Console.WriteLine(" ");

            // Area/Label should be System.Data
            return categorizedLabel;
        }

        private static async Task Label()
        {
            //(CDLTLL-2)
            var token = Configuration["GitHubToken"];
            var repoOwner = Configuration["GitHubRepoOwner"];
            var repoName = Configuration["GitHubRepoName"];

            Console.WriteLine($"GitHub Token = {Configuration["GitHubToken"]}");
            Console.WriteLine($"GitHub Repo Owner = {Configuration["GitHubRepoOwner"]}");
            Console.WriteLine($"GitHub Repo Name = {Configuration["GitHubRepoName"]}");
            Console.WriteLine();

            if (string.IsNullOrEmpty(token) ||
                string.IsNullOrEmpty(repoOwner) ||
                string.IsNullOrEmpty(repoName))
            {
                Console.Error.WriteLine();

                //(CDLTLL-2)
                Console.Error.WriteLine("Error: please configure the credentials in the appsettings.json file");
                Console.ReadLine();
                return;
            }

            //(CDLTLL)
            var labeler = new Labeler(repoOwner, repoName, token);

            await labeler.LabelAllNewIssues();

            Console.WriteLine("Labeling completed");
            Console.ReadLine();
        }
    }
}