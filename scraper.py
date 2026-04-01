"""
scraper.py - YouTube Comment Scraper

Scrapes comments from a YouTube video using YouTube's internal API.
No API key required. Includes retry logic and rate limit handling.

Usage:
    from scraper import scrape_comments
    comments = scrape_comments("https://www.youtube.com/watch?v=VIDEO_ID", max_comments=50)
"""

import re
import json
import time
import requests


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _make_request(method, url, max_retries=4, **kwargs):
    """Make HTTP request with retry logic for rate limiting."""
    resp = None
    for attempt in range(max_retries):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=15, **kwargs)
            else:
                resp = requests.post(url, timeout=15, **kwargs)

            if resp.status_code == 429:
                wait_time = (attempt + 1) * 3
                time.sleep(wait_time)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.HTTPError:
            if resp is not None and resp.status_code == 429 and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
                continue
            raise
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise RuntimeError("Request timed out")

    raise RuntimeError(
        "YouTube is rate-limiting requests. Please wait a minute and try again, "
        "or use the 'Paste Comments' tab to analyze comments manually."
    )


def scrape_comments(url, max_comments=50):
    """
    Scrape comments from a YouTube video.

    Args:
        url: YouTube video URL or video ID
        max_comments: Maximum number of comments to fetch

    Returns:
        list of dicts with keys: 'author', 'text', 'votes', 'time'
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    # Step 1: Fetch the video page
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    resp = _make_request("GET", video_url, headers=session.headers.copy())
    html = resp.text

    # Extract API key
    api_key = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"
    key_match = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
    if key_match:
        api_key = key_match.group(1)

    # Extract initial data
    data_match = re.search(r'var ytInitialData\s*=\s*({.+?});\s*</script>', html)
    if not data_match:
        data_match = re.search(r'window\["ytInitialData"\]\s*=\s*({.+?});\s*', html)
    if not data_match:
        raise RuntimeError("Could not parse YouTube page. The video may be unavailable.")

    initial_data = json.loads(data_match.group(1))

    # Step 2: Find comment section continuation token
    continuation_token = _find_continuation(initial_data)
    if not continuation_token:
        raise RuntimeError(
            "Could not find comments section. "
            "Comments may be disabled for this video."
        )

    # Step 3: Fetch comments
    all_comments = []
    pages = 0
    max_pages = 5

    while continuation_token and len(all_comments) < max_comments and pages < max_pages:
        payload = {
            "context": {
                "client": {
                    "clientName": "WEB",
                    "clientVersion": "2.20241201.00.00",
                    "hl": "en",
                    "gl": "US",
                }
            },
            "continuation": continuation_token,
        }

        try:
            api_url = f"https://www.youtube.com/youtubei/v1/next?key={api_key}"
            resp = _make_request(
                "POST", api_url,
                json=payload,
                headers={
                    **session.headers,
                    "Content-Type": "application/json",
                    "X-YouTube-Client-Name": "1",
                    "X-YouTube-Client-Version": "2.20241201.00.00",
                },
            )
            data = resp.json()
        except Exception as e:
            if all_comments:
                break
            raise RuntimeError(f"Failed to fetch comments: {str(e)}")

        # Extract comments from response
        new_comments, next_token = _parse_comments_response(data)
        all_comments.extend(new_comments)
        continuation_token = next_token
        pages += 1

        # Small delay between pages
        if continuation_token and len(all_comments) < max_comments:
            time.sleep(0.5)

    return all_comments[:max_comments]


def _find_continuation(data, depth=0):
    """Recursively find the comment section continuation token."""
    if depth > 15:
        return None

    if isinstance(data, dict):
        # Direct continuation token patterns
        for key in ("token", "continuation"):
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]

        # Check continuationCommand
        if "continuationCommand" in data:
            token = data["continuationCommand"].get("token")
            if token:
                return token

        # Check itemSectionRenderer for comment section specifically
        if "itemSectionRenderer" in data:
            section = data["itemSectionRenderer"]
            section_id = section.get("sectionIdentifier", "")
            if "comment" in section_id.lower():
                contents = section.get("contents", [])
                for item in contents:
                    token = _find_continuation(item, depth + 1)
                    if token:
                        return token

        # Recurse into known container keys first
        priority_keys = [
            "contents", "tabs", "twoColumnWatchNextResults",
            "results", "secondaryResults", "itemSectionRenderer",
            "sectionListRenderer", "continuationItemRenderer",
            "continuationEndpoint", "continuationCommand",
        ]
        for key in priority_keys:
            if key in data:
                token = _find_continuation(data[key], depth + 1)
                if token:
                    return token

        # Then try everything else
        for key, value in data.items():
            if key not in priority_keys:
                token = _find_continuation(value, depth + 1)
                if token:
                    return token

    elif isinstance(data, list):
        for item in data:
            token = _find_continuation(item, depth + 1)
            if token:
                return token

    return None


def _parse_comments_response(data):
    """Parse YouTube API response and extract comments + next continuation."""
    comments = []
    next_token = None

    endpoints = data.get("onResponseReceivedEndpoints", [])

    for endpoint in endpoints:
        # Get continuation items
        items = (
            endpoint.get("reloadContinuationItemsCommand", {}).get("continuationItems", [])
            or endpoint.get("appendContinuationItemsAction", {}).get("continuationItems", [])
        )

        for item in items:
            # Extract comment
            thread = item.get("commentThreadRenderer", {})
            if thread:
                renderer = thread.get("comment", {}).get("commentRenderer", {})
                comment = _extract_comment(renderer)
                if comment:
                    comments.append(comment)

            # Check for next page continuation
            cont_item = item.get("continuationItemRenderer", {})
            if cont_item:
                token = (
                    cont_item
                    .get("continuationEndpoint", {})
                    .get("continuationCommand", {})
                    .get("token")
                )
                if token:
                    next_token = token

    return comments, next_token


def _extract_comment(renderer):
    """Extract comment data from a commentRenderer."""
    if not renderer:
        return None

    # Get text
    text_runs = renderer.get("contentText", {}).get("runs", [])
    text = "".join(run.get("text", "") for run in text_runs)

    if not text.strip():
        return None

    # Get author
    author = renderer.get("authorText", {}).get("simpleText", "Unknown")

    # Get time
    published = renderer.get("publishedTimeText", {}).get("runs", [])
    time_text = published[0].get("text", "") if published else ""

    # Get votes
    votes = renderer.get("voteCount", {}).get("simpleText", "0")

    return {
        "author": author,
        "text": text.strip(),
        "time": time_text,
        "votes": votes,
    }


if __name__ == "__main__":
    import sys

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"Scraping comments from: {test_url}\n")

    try:
        results = scrape_comments(test_url, max_comments=10)
        print(f"Found {len(results)} comments:\n")
        for i, c in enumerate(results, 1):
            print(f"  {i}. [{c['author']}] {c['text'][:120]}")
            print(f"     {c['time']} | {c['votes']} votes\n")
    except Exception as e:
        print(f"Error: {e}")
