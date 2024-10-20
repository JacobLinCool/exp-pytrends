import os
import pytz
import argparse
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from joblib import Memory
import pandas as pd
from tqdm import tqdm


cache_dir = os.path.join(os.path.dirname(__file__), "cache")
memory = Memory(cache_dir, verbose=0)


@memory.cache(ignore=["pytrends"])
def fetch_data(
    pytrends: TrendReq, keyword: str, start_date: datetime, days: int, geo: str
):
    start = start_date.strftime("%Y-%m-%d")
    end = (start_date + timedelta(days=days)).strftime("%Y-%m-%d")
    timeframe = f"{start}T00 {end}T00"
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop="")
    res = pytrends.interest_over_time()
    if res.empty:
        return res

    res.index = pd.to_datetime(res.index)
    res.index = res.index.tz_localize(pytz.utc).tz_convert(pytz.timezone("Asia/Taipei"))
    return res


def fetch_google_trends(keyword, start_date, end_date, geo="TW", overlap=1):
    pytrends = TrendReq(geo=geo, retries=5, backoff_factor=1)

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    each_step = 5
    nonoverlap = each_step - overlap

    all_data = []
    total_days = (end_date - current_date).days
    steps = (total_days + nonoverlap - 1) // nonoverlap

    progress_bar = tqdm(total=steps, desc=f"Fetching data for '{keyword}'", unit="step")

    for _ in range(steps):
        progress_bar.write(
            f"Fetching data from {current_date.date()} to {current_date.date() + timedelta(days=nonoverlap)}"
        )
        daily_data = fetch_data(pytrends, keyword, current_date, each_step, geo)
        if not daily_data.empty:
            points = 24 * overlap + 1
            rescale_factor = (
                1.0
                if len(all_data) == 0
                else (
                    all_data[-1].iloc[-points:].sum() / daily_data.iloc[:points].sum()
                ).values[0]
            )
            if rescale_factor == 0 or pd.isna(rescale_factor):
                progress_bar.write(
                    f"Warning: rescale factor is {rescale_factor}, meaning no overlap between two consecutive data. Will use 1.0 instead."
                )
                rescale_factor = 1.0
            daily_data[keyword] *= rescale_factor
            all_data.append(daily_data)
        current_date += timedelta(days=nonoverlap)
        progress_bar.update(1)
        if current_date > end_date:
            break

    progress_bar.close()

    if all_data:
        df = pd.concat(all_data)
        df[keyword] = df[keyword] / df[keyword].max() * 100
        df[keyword] = df[keyword].apply(lambda x: round(x, 3))
        return df
    else:
        return pd.DataFrame()


def save_to_csv(data, filename):
    data.to_csv(filename)
    print(f"Data saved to {filename}")


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

    args = parser.parse_args()

    all_data = []

    for keyword in args.keywords:
        if args.verbose:
            print(
                f"Fetching data for '{keyword}' in {args.geo} from {args.start_date} to {args.end_date}"
            )

        data = fetch_google_trends(
            keyword, args.start_date, args.end_date, args.geo, args.overlap
        )

        if not data.empty:
            all_data.append(data)
            if args.verbose:
                print(f"Data fetched successfully for '{keyword}'")
        else:
            print(f"No data found for '{keyword}'")

    if all_data:
        combined_data = pd.concat(all_data, axis=1)

        keywords_str = "_".join(args.keywords)
        start_date = args.start_date.replace("-", "")
        end_date = args.end_date.replace("-", "")
        geo = args.geo if args.geo else ""
        filename = (
            f"data/google_trends_{keywords_str}_{start_date}_{end_date}_{geo}.csv"
        )

        if args.output:
            filename = args.output
        save_to_csv(combined_data, filename)
    else:
        print("No data fetched for any keywords")


if __name__ == "__main__":
    main()
