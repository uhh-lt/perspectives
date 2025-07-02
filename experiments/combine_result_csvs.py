import typer
from pathlib import Path
import pandas as pd

app = typer.Typer()


@app.command()
def combine_result_csvs(
    results_dir: Path = typer.Option(..., help="Path to the results directory"),
    output_dir: Path = typer.Option(
        ..., help="Path to directory to save the combined results"
    ),
):
    """
    Combine multiple CSV files from the results directory into a single CSV file.
    """
    combined_results = pd.DataFrame()

    csv_files = list(results_dir.glob("*.csv"))
    csv_files = [csv for csv in csv_files if csv.name != "combined_results.csv"]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_results = pd.concat([combined_results, df], ignore_index=True)

    combined_results_path = output_dir / "combined_results.csv"

    if combined_results_path.exists():
        # If the file exists, read it and append the new results
        existing_results = pd.read_csv(combined_results_path)
        combined_results = pd.concat(
            [existing_results, combined_results], ignore_index=True
        )

    # Save the combined results to the file
    combined_results.to_csv(combined_results_path, index=False)
    print(f"Combined results saved to {combined_results_path}")

    # # Remove the individual CSV files
    # for csv_file in csv_files:
    #     csv_file.unlink()


if __name__ == "__main__":
    app()
