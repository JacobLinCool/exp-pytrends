import gradio as gr
from datetime import datetime, timedelta
import os
from google_trends_fetcher import fetch_google_trends, plot_trends, save_to_csv

# Ensure data directory exists
os.makedirs("data", exist_ok=True)


def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def process_keywords(keywords_str):
    """Convert comma-separated keywords string to list and clean whitespace"""
    return [k.strip() for k in keywords_str.split(",") if k.strip()]


def fetch_trends(keywords_str, start_date, end_date, geo, overlap):
    # Input validation
    if not keywords_str.strip():
        return "Please enter at least one keyword", None, None

    if not validate_date(start_date) or not validate_date(end_date):
        return "Invalid date format. Please use YYYY-MM-DD", None, None

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_dt >= end_dt:
        return "Start date must be before end date", None, None

    if (end_dt - start_dt).days > 365 * 15:
        return "Date range cannot exceed 15 years", None, None

    # Process keywords
    keywords = process_keywords(keywords_str)
    if not keywords:
        return "Please enter valid keywords", None, None

    try:
        # Fetch data
        data = fetch_google_trends(
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            geo=geo,
            overlap=overlap,
        )

        if data.empty:
            return "No data found for the specified parameters", None, None

        # Generate filenames
        keywords_str = "_".join(keywords)
        start_date_str = start_date.replace("-", "")
        end_date_str = end_date.replace("-", "")
        geo_str = geo if geo else "GLOBAL"
        base_filename = f"data/google_trends_{keywords_str}_{start_date_str}_{end_date_str}_{geo_str}"

        # Save CSV
        csv_filename = f"{base_filename}.csv"
        save_to_csv(data, csv_filename)

        # Generate plot
        plot_trends(data, base_filename)
        plot_filename = f"{base_filename}.png"

        return "Data fetched successfully!", csv_filename, plot_filename

    except Exception as e:
        return f"Error: {str(e)}", None, None


# Create Gradio interface
with gr.Blocks(title="Google Trends Data Fetcher") as demo:
    gr.Markdown(
        """
    # Google Trends Data Fetcher
    
    Fetch and visualize Google Trends data for multiple keywords over a specified time period.
    
    ### Instructions:
    1. Enter keywords separated by commas
    2. Select date range
    3. Optionally specify geographic location (e.g., 'US', 'JP', 'TW')
    4. Click 'Fetch Data' to start
    """
    )

    with gr.Row():
        with gr.Column():
            keywords_input = gr.Textbox(
                label="Keywords (comma-separated)",
                placeholder="Enter keywords, e.g.: bitcoin, ethereum, dogecoin",
            )
            start_date = gr.Textbox(
                label="Start Date (YYYY-MM-DD)",
                value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            )
            end_date = gr.Textbox(
                label="End Date (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d")
            )
            geo = gr.Textbox(
                label="Geographic Location (optional)",
                placeholder="e.g., US, JP, TW",
                value="TW",
            )
            overlap = gr.Slider(
                minimum=1, maximum=4, value=2, step=1, label="Overlap Days"
            )
            fetch_button = gr.Button("Fetch Data")

        with gr.Column():
            output_message = gr.Textbox(label="Status")
            csv_output = gr.File(label="Download CSV")
            plot_output = gr.Image(label="Trends Visualization")

    fetch_button.click(
        fn=fetch_trends,
        inputs=[keywords_input, start_date, end_date, geo, overlap],
        outputs=[output_message, csv_output, plot_output],
    )

if __name__ == "__main__":
    demo.launch()
