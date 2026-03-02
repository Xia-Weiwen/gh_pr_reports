import os
import sys
import re
import requests
from datetime import datetime, timedelta, timezone
import time
import json
from google import genai


def get_recent_pulls_of_repo(owner="pytorch", repo="pytorch", since_time=None, end_time=None):
    """
    Get pull requests from the given repository in the last week.
    """

    GITHUB_TOKEN = os.getenv("GH_TOKEN")
    if GITHUB_TOKEN is None:
        print("Warning: GH_TOKEN environment variable is not set. You may hit rate limits.")

    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if since_time is None:
        since_time = end_time - timedelta(weeks=1)
    since_iso = since_time.isoformat().replace('+00:00', 'Z')

    # API endpoint
    base_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

    # Request parameters
    params = {
        'state': 'all',  # Can be 'open', 'closed', or 'all'
        'sort': 'created',  # Sort by creation time
        'direction': 'desc',  # Descending order (newest first)
        'per_page': 50,  # Maximum number per page
        'page': 1
    }

    headers = {
        'Accept': 'application/vnd.github.v3+json',
    }

    # Add authentication (optional, but helps avoid rate limits)
    # You need to create a Personal Access Token on GitHub
    if GITHUB_TOKEN is not None:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'

    all_pulls = []

    retries = 5
    try:
        while True:
            print(f"Fetching page {params['page']}...")
            response = requests.get(base_url, params=params, headers=headers)

            # Check response status
            if response.status_code != 200:
                print(f"Request failed, status code: {response.status_code}")
                print(f"Error message: {response.text}")
                retries -= 1
                if retries > 0:
                    print(f"Retrying... ({retries} attempts left)")
                    time.sleep(5)
                    continue
                else:
                    print("Max retries reached. Exiting.")
                    break
            retries = 5  # Reset retries on success

            # Parse JSON response
            pulls = response.json()

            if not pulls:
                print("No more data")
                break

            print(f"Fetched {len(pulls)} pull requests on page {params['page']}")

            # Filter PRs from the last week
            recent_pulls = []
            found_older = False
            for pull in pulls:
                created_at = datetime.fromisoformat(pull['created_at'].replace('Z', '+00:00'))

                # Add to results if created within the last week
                if created_at >= since_time and created_at <= end_time:
                    recent_pulls.append({
                        'number': pull['number'],
                        'title': pull['title'],
                        'state': pull['state'],
                        'created_at': pull['created_at'],
                        'updated_at': pull['updated_at'],
                        'user': pull['user']['login'],
                        'labels': [label['name'] for label in pull.get('labels', [])],
                        'body': pull.get('body', ''),
                    })
                elif created_at < since_time:
                    # Since the list is sorted by creation time, stop when encountering older PRs
                    found_older = True
                    break

            print(f"  -> {len(recent_pulls)} pull requests created since {since_iso}")
            all_pulls.extend(recent_pulls)

            # Stop fetching if current page has PRs older than a week
            if found_older:
                break

            params['page'] += 1
            time.sleep(8 + len(all_pulls)/100)  # Avoid hitting rate limits

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []

    return all_pulls


def clean_up_body(body: str):
    cleaned_body = re.sub(r'<!--.*?-->', '', body)
    cleaned_body = re.sub(r'\n+', '\n', cleaned_body)
    cleaned_body = re.sub(r'(\r\n)+', r'\r\n', cleaned_body)
    tail_idx = cleaned_body.find("## Accuracy Tests")
    if tail_idx > 0:
        cleaned_body = cleaned_body[:tail_idx]
    tail_idx = cleaned_body.find("## Checklist")
    if tail_idx > 0:
        cleaned_body = cleaned_body[:tail_idx]
    return cleaned_body


def get_pulls_summary_as_string(pulls):
    """
    Get a summary of pull requests as a string.
    """
    summary_lines = [f"\nThere are {len(pulls)} pull requests recently:"]

    for i, pull in enumerate(pulls, 1):
        summary_lines.append(f"{i}. PR #{pull['number']}: {pull['title']}")
        summary_lines.append(f"   Author: {pull['user']}")
        summary_lines.append(f"   Created at: {pull['created_at']}")
        summary_lines.append(f"   Labels: {', '.join(pull['labels']) if pull['labels'] else 'None'}")
        summary_lines.append(f"   Description: {clean_up_body(pull['body'])}")
        summary_lines.append("\n\n")

    return "\n".join(summary_lines)


def summarize_prs_by_ai(pr_summary_string, owner, repo):
    """
    Send the PR summary to an AI model for analysis.
    """
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()
    print("Sending PR summary to AI model for analysis...")
    prompt = (
f"""
Act as a Senior AI Software Architect. Analyze the following list of Pull Requests from {owner}/{repo} and provide a high-signal technical digest for a team of framework developers. Be Professional, technical and objective. Keep concise. Skip the "Here is your report" intro.

Start the report with a key takeaways: A one-sentence summary of the most impactful trend this week, with total number of PRs and top active areas (with inline numbering for these areas). Then summarize these PRs by their categories (features, components, hardware backends, etc.)

**Formatting Instructions:**
- Use numbering for the first level, bullet points for other levels
- Use backticks for code symbols (e.g., `DTensor`).
- Give number and link for mentioned PRs (e.g., [PR#1234](URL))

**Input Data:**
{pr_summary_string}
"""
    )
    summary = ""
    try:
        print("Trying GEMINI...")
        response = client.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        )
        summary = response.text
    except Exception as e:
        print("Request failed!")
        print(e)
        raise e
    return summary


if __name__ == "__main__":
    # Get repo from command line args or default to pytorch/pytorch
    if len(sys.argv) >= 3:
        owner = sys.argv[1]
        repo = sys.argv[2]
    else:
        owner = "pytorch"
        repo = "pytorch"
    print("=" * 20)
    print(owner, "/", repo)
    print("=" * 20)
    # Get pull requests
    end_time = datetime.now(timezone.utc)
    # end_time = datetime(year=2026, month=3, day=1, hour=0, minute=0, second=0, tzinfo=timezone.utc)
    since_time = end_time - timedelta(weeks=1)
    recent_pulls = get_recent_pulls_of_repo(owner, repo, since_time, end_time)
    if not recent_pulls:
        print("No recent pull requests found.")
        sys.exit(0)

    # Print summary
    print(f"Number of recent pulls: {len(recent_pulls)}")
    summary_string = get_pulls_summary_as_string(recent_pulls)

    # Send to AI for analysis
    ai_analysis = summarize_prs_by_ai(summary_string, owner, repo)
    print("AI Analysis of Recent Pull Requests:")
    print(ai_analysis)

    # write to file in reports/onwer_repo/report-{date_in_YYYYMMDD}.md
    os.makedirs("reports", exist_ok=True)
    report_dir = os.path.join("reports", f"{owner}_{repo}")
    os.makedirs(report_dir, exist_ok=True)
    date_str = end_time.strftime("%Y%m%d")
    start_date = since_time.strftime("%Y%m%d")
    report_path = os.path.join(report_dir, f"report-{date_str}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# AI Analysis of Pull Requests for {owner}/{repo} during {start_date}-{date_str}\n\n")
        f.write(ai_analysis)
        f.write("\n")
    print(f"Report saved to {report_path}")
