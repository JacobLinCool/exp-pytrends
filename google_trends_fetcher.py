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


def fetch_google_trends(keywords, start_date, end_date, geo="TW", overlap=1):
    if isinstance(keywords, str):
        keywords = [keywords]

    pytrends = TrendReq(geo=geo, retries=10, backoff_factor=1)

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    each_step = 5
    nonoverlap = each_step - overlap

    all_data = []
    total_days = (end_date - current_date).days
    steps = (total_days + nonoverlap - 1) // nonoverlap

    progress_bar = tqdm(total=steps, desc=f"Fetching data for {keywords}", unit="step")

    for _ in range(steps):
        progress_bar.write(
            f"Fetching data from {current_date.date()} to {current_date.date() + timedelta(days=nonoverlap)}"
        )
        daily_data = fetch_data(pytrends, keywords, current_date, each_step, geo)

        if not daily_data.empty:
            if len(all_data) > 0:
                # Calculate rescale factors for each keyword
                rescale_factors = {}
                for keyword in keywords:
                    # Get first date from current sequence
                    first_date = daily_data.index[0]

                    # Find matching date range in previous sequence
                    prev_data = all_data[-1][keyword]
                    overlap_prev = prev_data[prev_data.index >= first_date]
                    overlap_curr = daily_data[keyword].iloc[: len(overlap_prev)]

                    # Only rescale if both overlap sections have non-zero values
                    if overlap_prev.sum() > 0 and overlap_curr.sum() > 0:
                        rescale_factor = overlap_prev.sum() / overlap_curr.sum()
                    else:
                        progress_bar.write(
                            f"Warning: Zero values in overlap period for '{keyword}'. Using factor 1.0"
                        )
                        rescale_factor = 1.0

                    if pd.isna(rescale_factor) or rescale_factor == 0:
                        progress_bar.write(
                            f"Warning: Invalid rescale factor for '{keyword}'. Using factor 1.0"
                        )
                        rescale_factor = 1.0

                    rescale_factors[keyword] = rescale_factor
                    daily_data[keyword] *= rescale_factor

                progress_bar.write(f"Applied rescale factors: {rescale_factors}")

                # Remove the overlapped points from the current data before concatenating
                daily_data = daily_data.iloc[len(overlap_prev) :]

            all_data.append(daily_data)

        current_date += timedelta(days=nonoverlap)
        progress_bar.update(1)
        if current_date > end_date:
            break

    progress_bar.close()

    if all_data:
        df = pd.concat(all_data)

        # Assertion to check for consecutive timestamps
        timestamps = df.index
        time_diffs = timestamps[1:] - timestamps[:-1]
        expected_diff = pd.Timedelta(hours=1)  # Google Trends data is hourly
        is_consecutive = all(diff == expected_diff for diff in time_diffs)
        assert is_consecutive, (
            "Error: Data points are not consecutive. There might be overlaps or gaps in the data. "
            f"Time differences: {[str(diff) for diff in time_diffs if diff != expected_diff]}"
        )

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


def plot_trends(data, filename_base):
    """Plot the trends data and save to a file."""
    # Setup Chinese font support
    setup_chinese_font()

    plt.figure(figsize=(12, 6))

    if "isPartial" in data.columns:
        data = data.drop("isPartial", axis=1)

    for column in data.columns:
        plt.plot(data.index, data[column], label=column, marker=".", markersize=2)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
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
        "--plot", action="store_true", help="Generate a plot of the trends data"
    )

    args = parser.parse_args()

    if args.verbose:
        print(
            f"Fetching data for {args.keywords} in {args.geo} from {args.start_date} to {args.end_date}"
        )

    data = fetch_google_trends(
        args.keywords, args.start_date, args.end_date, args.geo, args.overlap
    )

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

        if args.plot:
            plot_trends(data, filename)
    else:
        print("No data fetched for keywords")


if __name__ == "__main__":
    main()
