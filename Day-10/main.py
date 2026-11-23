from src.data_loader import load_data, display_dataset_info
from src.data_quality import analyze_data_quality, save_quality_report
from src.eda import perform_eda, save_eda_observations
from src.visualization import create_visualizations
from src.train import train_and_compare_models
from src.evaluate import evaluate_best_model
from src.error_analysis import perform_error_analysis


def main():

    print("\n========================================")
    print(" TITANIC SURVIVAL PREDICTION PROJECT")
    print(" COMPLETE MACHINE LEARNING WORKFLOW")
    print("========================================")

    # Step 1: Load dataset
    print("\n[1/7] Loading Dataset...")
    df = load_data()
    display_dataset_info(df)

    # Step 2: Data Quality
    print("\n[2/7] Data Quality Analysis...")
    missing_report = analyze_data_quality(df)
    save_quality_report(df, missing_report)

    # Step 3: EDA
    print("\n[3/7] Exploratory Data Analysis...")
    perform_eda(df)
    save_eda_observations(df)

    # Step 4: Visualizations
    print("\n[4/7] Creating Visualizations...")
    create_visualizations(df)

    # Step 5: Model Training
    print("\n[5/7] Training and Comparing Models...")
    train_and_compare_models()

    # Step 6: Final Evaluation
    print("\n[6/7] Evaluating Best Model...")
    evaluate_best_model()

    # Step 7: Error Analysis
    print("\n[7/7] Performing Error Analysis...")
    perform_error_analysis()

    print("\n========================================")
    print(" PROJECT EXECUTION COMPLETED!")
    print("========================================")

    print("\nGenerated Outputs:")
    print("- models/best_model.joblib")
    print("- reports/data_quality_report.md")
    print("- reports/eda_observations.md")
    print("- reports/model_comparison.csv")
    print("- reports/metrics.json")
    print("- reports/error_analysis.csv")
    print("- reports/model_review.md")
    print("- reports/figures/")


if __name__ == "__main__":
    main()
