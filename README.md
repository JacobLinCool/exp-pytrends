# Google Trends Data Fetcher

This tool allows you to easily fetch and save Google Trends data for multiple keywords over a specified time period.

## Prerequisites

Before you begin, make sure you have Python installed on your computer. This script works with Python 3.10 or newer.

## Setup

1. Download all the files in this project to your computer.

2. Open a terminal or command prompt and navigate to the folder containing these files.

3. Install the required packages by running: `pip install -r requirements.txt`

## How to Use

> To run as web app, use `python app.py` and open `http://127.0.0.1:7860`.

To use this tool as command line tool, you'll run it from the command line with specific arguments. Here's the basic structure:

### Required Arguments

- `[keywords]`: One keyword you want to search for, separated by spaces.
- `--start-date`: The start date for your data in YYYY-MM-DD format.
- `--end-date`: The end date for your data in YYYY-MM-DD format.

### Optional Arguments

- `--geo`: The geographic location for the trends data.
- `--output`: Specify a custom output file name.
- `--verbose`: Add this flag to see more detailed output during the process.
- `--overlap`: Add this flag to set overlapping days. (default is 1)

### Examples

1. Fetch data for a keyword:

```sh
python google_trends_fetcher.py "bitcoin" --start-date 2023-01-01 --end-date 2023-12-31
```

2. Fetch data for a specific geographic location (e.g., Taiwan):

```sh
python google_trends_fetcher.py "股票" --start-date 2024-09-01 --end-date 2024-09-30 --geo TW
```

## Output

The script will save the fetched data as a CSV file in the `data` folder. The file name will include the keywords, date range, and geographic location unless you specify a custom output file name.

## Troubleshooting

- If you encounter any errors, make sure all required packages are installed correctly.
- Check that your date format is correct (YYYY-MM-DD).
- Ensure you have a stable internet connection.

For any other issues, please contact the script maintainer.

Happy trend hunting!
