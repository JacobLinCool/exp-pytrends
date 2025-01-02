import os
import pytz
import argparse
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from joblib import Memory
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib import rcParams
import time


cache_dir = os.path.join(os.path.dirname(__file__), "cache")
memory = Memory(cache_dir, verbose=0)


@memory.cache(ignore=["pytrends"])
def fetch_data(
    pytrends: TrendReq, keywords: list, start_date: datetime, days: int, geo: str
):
    start_time = time.time()

    start = start_date.strftime("%Y-%m-%d")
    end = (start_date + timedelta(days=days)).strftime("%Y-%m-%d")
    timeframe = f"{start}T00 {end}T00"
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="")
    res = pytrends.interest_over_time()

    if res.empty:
        return res

    res.index = pd.to_datetime(res.index)
    res.index = res.index.tz_localize(pytz.utc).tz_convert(pytz.timezone("Asia/Taipei"))

    # Throttling to ensure the request takes at least 0.5 second
    elapsed_time = time.time() - start_time
    if elapsed_time < 0.5:
        time.sleep(0.5 - elapsed_time)

    return res


def calculate_rescale_factor(last_df, current_df):
    """
    Calculate rescale factor based on the keyword with highest average in overlapping period.
    Returns a rescale_factor to be applied to all keywords.
    """
    # Find overlapping period
    overlap_start = max(last_df.index.min(), current_df.index.min())
    overlap_end = min(last_df.index.max(), current_df.index.max())

    # Get overlapping data using boolean indexing
    last_overlap = last_df[
        (last_df.index >= overlap_start) & (last_df.index <= overlap_end)
    ]
    current_overlap = current_df[
        (current_df.index >= overlap_start) & (current_df.index <= overlap_end)
    ]

    if last_overlap.empty or current_overlap.empty:
        return None

    # Calculate average values for each keyword in overlapping period
    keywords = [col for col in last_overlap.columns if col != "isPartial"]
    keyword_avgs = {k: current_overlap[k].mean() for k in keywords}

    # Find keyword with highest average
    best_keyword = max(keyword_avgs.items(), key=lambda x: x[1])[0]

    # Calculate rescale factor using the best keyword
    last_values = last_overlap[best_keyword]
    current_values = current_overlap[best_keyword]

    # Find indices where both series have non-zero values
    valid_indices = (last_values > 0) & (current_values > 0)

    if not valid_indices.any():
        return None

    # Calculate ratio using the best keyword
    ratios = last_values[valid_indices] / current_values[valid_indices]
    rescale_factor = ratios.mean()

    # Return the same factor for all keywords
    return rescale_factor


def fetch_google_trends(keywords, start_date, end_date, geo="TW", verbose=False):
    if isinstance(keywords, str):
        keywords = [keywords]

    pytrends = TrendReq(geo=geo, retries=10, backoff_factor=1)

    fetch_head = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=pytz.utc)
    end_date = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=pytz.utc)

    all_data = []
    last_dataframe = None
    total_days = (end_date - fetch_head).days
    days_fetched = 0

    progress_bar = tqdm(
        total=total_days, desc=f"Fetching data for {keywords}", unit="days"
    )

    try:
        while fetch_head < end_date:
            segment_data = fetch_data(pytrends, keywords, fetch_head, 5, geo)
            if segment_data.empty:
                raise ValueError(
                    f"No data fetched for keywords on {fetch_head.date()}, try to add a keyword that always has data"
                )
            last_date = segment_data[segment_data[keywords].sum(axis=1) > 0].index[-1]

            # Update progress bar with days fetched in this iteration
            days_in_segment = (last_date - fetch_head).days
            days_fetched += days_in_segment
            progress_bar.update(days_in_segment)

            if verbose:
                print(f"Data fetched from {fetch_head} to {last_date}")

            if last_dataframe is not None:
                rescale_factor = calculate_rescale_factor(last_dataframe, segment_data)
                if verbose:
                    print(f"Rescale factor: {rescale_factor}")
                if rescale_factor is None:
                    raise ValueError(
                        "No overlapping data found, try to add a keyword that always has data"
                    )

                for keyword in keywords:
                    segment_data[keyword] *= rescale_factor

            fetch_head = last_date - timedelta(hours=23)
            last_dataframe = segment_data[segment_data.index >= fetch_head]
            all_data.append(segment_data)

    finally:
        progress_bar.close()

    if all_data:
        df = pd.concat(all_data)

        # Sort index and remove duplicates, keeping the last value
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Ensure hourly data points by resampling
        date_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="h", tz=df.index.tz
        )
        df = df.reindex(date_range)

        # Interpolate missing values (limit to 24 hours gap)
        df = df.interpolate(method="time", limit=24)

        # Fill remaining NaN with 0
        df = df.fillna(0)

        # Normalize all keywords together
        max_value = df.max().max()
        if max_value > 0:
            df = df / max_value * 100
        return df
    else:
        return pd.DataFrame()


def save_to_csv(data, filename):
    data.to_csv(filename)
    print(f"Data saved to {filename}")


def setup_chinese_font():
    """Setup matplotlib to use a font that supports Chinese characters."""
    windows_font = "Microsoft YaHei"
    macos_font = "Arial Unicode MS"
    linux_font = "Noto Sans CJK TC"

    # Try different fonts based on platform
    for font_name in [windows_font, macos_font, linux_font]:
        if any(f.name == font_name for f in fm.fontManager.ttflist):
            rcParams["font.family"] = font_name
            return True

    # If none of the preferred fonts are found, try to find any font that supports Chinese
    chinese_fonts = [
        f.name
        for f in fm.fontManager.ttflist
        if ("Chinese" in f.name or "CJK" in f.name)
    ]

    if chinese_fonts:
        rcParams["font.family"] = chinese_fonts[0]
        return True

    print(
        "Warning: No suitable Chinese font found. Chinese characters may not display correctly."
    )
    return False


def plot_trends(data, filename_base, columns=None):
    """Plot the trends data and save to a file.

    Args:
        data: DataFrame containing the trends data
        filename_base: Base filename for saving the plot
        columns: List of column names to plot. If None, plot all columns
    """
    setup_chinese_font()

    plt.figure(figsize=(12, 6))

    if "isPartial" in data.columns:
        data = data.drop("isPartial", axis=1)

    # If columns is specified, only plot those columns
    plot_columns = columns if columns else data.columns
    plot_columns = [col for col in plot_columns if col in data.columns]

    if not plot_columns:
        print("Warning: No valid columns to plot")
        return

    for column in plot_columns:
        plt.plot(data.index, data[column], label=column, marker=".", markersize=2)

    plt.gca().xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d", tz=data.index.tz)
    )
    plt.gcf().autofmt_xdate()  # Rotation and alignment of tick labels

    plt.title("Google Trends Data", fontsize=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Relative Interest", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plot_filename = f"{filename_base}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Google Trends data for keywords"
    )
    parser.add_argument("keywords", nargs="+", help="Keywords to search for")
    parser.add_argument(
        "--start-date", required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument("--geo", default="", help="Geographic location")
    parser.add_argument("--output", help="Output CSV file name")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--overlap", type=int, default=1, help="Number of overlapping days (default: 1)"
    )
    parser.add_argument(
        "--plot",
        nargs="*",
        metavar="COLUMN",
        help="Generate a plot of the trends data. Optionally specify column names to plot",
    )

    args = parser.parse_args()

    if args.verbose:
        print(
            f"Fetching data for {args.keywords} in {args.geo} from {args.start_date} to {args.end_date}"
        )

    data = fetch_google_trends(
        args.keywords, args.start_date, args.end_date, args.geo, args.verbose
    )

    if "isPartial" in data.columns:
        data = data.drop("isPartial", axis=1)

    print(data.describe())

    if not data.empty:
        if args.verbose:
            print(f"Data fetched successfully for {args.keywords}")

        keywords_str = "_".join(args.keywords)
        start_date = args.start_date.replace("-", "")
        end_date = args.end_date.replace("-", "")
        geo = args.geo if args.geo else ""
        filename = f"data/google_trends_{keywords_str}_{start_date}_{end_date}_{geo}"

        if args.output:
            filename = args.output.rsplit(".", 1)[0]  # Remove extension if present

        csv_filename = f"{filename}.csv"
        save_to_csv(data, csv_filename)

        if args.plot is not None:
            plot_trends(data, filename, args.plot if args.plot else None)
    else:
        print("No data fetched for keywords")


if __name__ == "__main__":
    main()
